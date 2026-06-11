import contextlib
import io
import datetime
import logging
import threading
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

# Graphics Imports for Vehicle Diagram
from reportlab.graphics.shapes import Drawing, Line, String, Polygon, Group
from reportlab.graphics import renderPDF

# Internal Modules
import bricos_solver as solver
import bricos_viz as viz
import bricos_data as data_mod

_logger = logging.getLogger("bricos.report")


@contextlib.contextmanager
def persistent_image_export():
    """Keep ONE kaleido export process alive for all plot exports.

    With kaleido 1.x, every fig.write_image spawns and tears down a fresh
    headless browser (~3-4 s per image; the v0.53 test report spent ~210 s
    on its ~58 exports). kaleido's sync server is a singleton that
    calc_fig_sync - plotly's write_image backend - reuses when running, so
    starting it once around report generation reduces each export to the
    actual render time. kaleido 0.2.x keeps a persistent global scope of
    its own and has no sync server; export then works exactly as before.
    """
    try:
        import kaleido
        start = getattr(kaleido, 'start_sync_server', None)
        stop = getattr(kaleido, 'stop_sync_server', None)
    except Exception:
        start = stop = None
    if start is None or stop is None:
        yield
        return
    try:
        start(silence_warnings=True)
    except Exception:
        _logger.exception("Could not start the kaleido sync server; "
                          "plot exports fall back to one browser per image.")
        yield
        return
    try:
        yield
    finally:
        try:
            stop(silence_warnings=True)
        except Exception:
            _logger.exception("Could not stop the kaleido sync server.")


# ==========================================
# CUSTOM CANVAS FOR PAGE NUMBERING
# ==========================================

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        """add page info to each page (page x of y)"""
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        # Draw "Page x of y" in the bottom right corner
        self.setFont("Helvetica", 8)
        self.drawRightString(200*mm, 10*mm, 
            "Page %d of %d" % (self._pageNumber, page_count))

# ==========================================
# REPORT GENERATION MODULE
# ==========================================

class BricosReportGenerator:
    def __init__(self, buffer, meta_data, session_state, raw_res_A, raw_res_B, nodes_A, nodes_B, version=None, progress_callback=None):
        self.buffer = buffer
        self.meta = meta_data
        self.state = session_state
        self.version = version or data_mod.APP_VERSION
        self.progress_callback = progress_callback
        
        # Validity Check
        self.valid_B = (nodes_B is not None)
        
        self.styles = getSampleStyleSheet()
        self.elements = []
        self.chapter_count = 1
        
        # Define Custom Styles
        self.styles.add(ParagraphStyle(name='SwecoHeader', parent=self.styles['Heading1'], fontSize=16, spaceAfter=12))
        self.styles.add(ParagraphStyle(name='SwecoSubHeader', parent=self.styles['Heading2'], fontSize=14, spaceAfter=10, textColor=colors.darkblue))
        self.styles.add(ParagraphStyle(name='SwecoTableHead', parent=self.styles['Normal'], fontSize=9, fontName='Helvetica-Bold'))
        self.styles.add(ParagraphStyle(name='SwecoBody', parent=self.styles['Normal'], fontSize=9, leading=11))
        self.styles.add(ParagraphStyle(name='SwecoSmall', parent=self.styles['Normal'], fontSize=8, leading=10))
        self.styles.add(ParagraphStyle(name='SwecoCell', parent=self.styles['Normal'], fontSize=8, leading=9))
        self.styles.add(ParagraphStyle(name='SwecoMath', parent=self.styles['Normal'], fontSize=10, leading=12, alignment=TA_CENTER, spaceAfter=6, spaceBefore=6))
        
        # New style for logs with increased leading to prevent overlap
        self.styles.add(ParagraphStyle(name='SwecoLog', parent=self.styles['Normal'], fontSize=8, leading=14))

        # Formula lines with sub/superscripts: the markup descends below the
        # 10 pt line box of SwecoSmall and collides with whatever flowable
        # follows (e.g. the Phi_SLS line ran into the phi table).
        self.styles.add(ParagraphStyle(name='SwecoFormula', parent=self.styles['Normal'], fontSize=8, leading=12, spaceBefore=2, spaceAfter=4))

        # Use pre-calculated results passed from Main UI to avoid redundant Numba execution
        self.params_A = self.state['sysA']
        self.params_B = self.state['sysB']
        
        # Retrieve Model Properties (E-modulus) passed from Main
        self.props_A = self.state.get('model_props_A', {'Spans':{}, 'Walls':{}})
        self.props_B = self.state.get('model_props_B', {'Spans':{}, 'Walls':{}})
        
        self.raw_A = raw_res_A
        self.nodes_A = nodes_A
        self.raw_B = raw_res_B
        self.nodes_B = nodes_B
        
        # Initialize ThreadPool for Parallel Rendering
        self.executor = ThreadPoolExecutor(max_workers=4)
        # Serializes fig.write_image across worker threads; see
        # _render_plot_task for why the export step must not run in parallel.
        self._image_export_lock = threading.Lock()
        # combine_results memo: every component chapter needs the same
        # 'Unfactored' combination - recombining per chapter was pure waste.
        # Cached results are read-only, like the solver cache contract.
        self._combine_cache = {}

    def _combined(self, sys_id, res_mode):
        key = (sys_id, res_mode)
        if key not in self._combine_cache:
            raw = self.raw_A if sys_id == 'A' else self.raw_B
            params = self.params_A if sys_id == 'A' else self.params_B
            self._combine_cache[key] = solver.combine_results(raw, params, res_mode)
        return self._combine_cache[key]

    def _update_progress(self, val):
        if self.progress_callback:
            self.progress_callback(val)

    def _match_vehicle_class(self, current_loads, current_spacing):
        """
        Checks if the current load/spacing configuration matches a standard vehicle
        defined in vehicles.csv. Returns the name if found, else 'Custom'.
        Delegates to the cached vehicle library instead of re-reading the CSV
        for every vehicle in the report.
        """
        try:
            return data_mod.identify_vehicle_class(current_loads, current_spacing)
        except Exception:
            return "Custom"

    @staticmethod
    def _vehicle_has_loads(params, vehicle_key):
        return bool(params.get(vehicle_key, {}).get('loads'))

    def _has_any_vehicle(self):
        systems = [self.params_A]
        if self.valid_B:
            systems.append(self.params_B)
        return any(
            self._vehicle_has_loads(p, 'vehicle') or self._vehicle_has_loads(p, 'vehicleB')
            for p in systems
        )

    @staticmethod
    def _surcharge_interaction_text(params):
        if params.get('combine_surcharge_vehicle', False):
            return "Simultaneous: vehicle traffic + surcharge"
        return "Exclusive: envelope of vehicle traffic or surcharge"

    @staticmethod
    def _has_vehicle_loads(params):
        return bool(params.get('vehicle', {}).get('loads')) or bool(params.get('vehicleB', {}).get('loads'))

    @staticmethod
    def _has_surcharge_loads(params):
        return bool(params.get('surcharge'))

    @staticmethod
    def _characteristic_formula_text(params):
        has_vehicle = BricosReportGenerator._has_vehicle_loads(params)
        has_surcharge = BricosReportGenerator._has_surcharge_loads(params)

        sls_mode = params.get('phi_sls_mode', 'Same')
        phi_sym = "Phi_SLS" if sls_mode in ('Reduced', 'Manual') else "Phi"

        sls_g = params.get('sls_g', 1.0)
        sls_j = params.get('sls_j', 1.0)
        sls_veh = params.get('sls_veh', 1.0)
        sls_vehB = params.get('sls_vehB', 1.0)
        has_A = bool(params.get('vehicle', {}).get('loads'))
        has_B = bool(params.get('vehicleB', {}).get('loads'))

        veh_terms = []
        if has_A: veh_terms.append(f"{sls_veh} · {phi_sym} · VehA")
        if has_B: veh_terms.append(f"{sls_vehB} · {phi_sym} · VehB")
        veh_txt = " + ".join(veh_terms)
        surch_txt = f"{sls_veh} · Surcharge"

        if has_vehicle and has_surcharge:
            if params.get('combine_surcharge_vehicle', False):
                variable = f"{veh_txt} + {surch_txt}"
            else:
                variable = f"Envelope({veh_txt}, {surch_txt})"
        elif has_vehicle:
            variable = veh_txt
        elif has_surcharge:
            variable = surch_txt
        else:
            variable = ""

        if data_mod.udl_line_load(params) > 0.0:
            udl_term = f"{params.get('sls_udl', 0.40)} · Traffic UDL (no Phi)"
            variable = f"{variable} + {udl_term}" if variable else udl_term

        perm = f"{sls_g} · SW"
        if params.get('soil'): perm += f" + {sls_j} · Soil"
        if not variable:
            return perm
        eq = f"{perm} + {variable}"
        if has_vehicle and sls_mode == 'Reduced':
            eq += (" , where Phi_SLS = 1 + (Phi_ULS - 1)/2 per Vejledning til "
                   "belastnings- og beregningsgrundlag for broer, 5.4.2")
        elif has_vehicle and sls_mode == 'Manual':
            eq += f" , where Phi_SLS = {params.get('phi_sls', 1.0):.3f} (user-defined)"
        return eq

    @staticmethod
    def _phi_display_text(p, raw):
        """Formatted phi value(s) for equations and summaries.

        Returns a single number, or a min-max range when per-member values
        exist (calculated span-based or manual per-span).
        """
        members = (raw.get('Phi Members') or {}) if raw else {}
        vals = sorted(set(round(v, 4) for v in members.values()))
        if len(vals) > 1:
            return f"Phi[{vals[0]:.2f}-{vals[-1]:.2f}]"
        if vals:
            return f"{vals[0]:.2f}"
        if p.get('phi_mode') == 'Calculate' and raw:
            return f"{raw.get('phi_calc', 1.0):.2f}"
        return f"{p.get('phi', 1.0):.2f}"

    def _build_characteristic_formula_text(self):
        eqA = self._characteristic_formula_text(self.params_A)
        if not self.valid_B:
            return eqA
        eqB = self._characteristic_formula_text(self.params_B)
        if eqA == eqB:
            return eqA
        return f"SysA: {eqA} <br/> SysB: {eqB}"

    @staticmethod
    def _vehicle_step_keys_for_direction(params, base_key):
        direction = params.get('vehicle_direction', 'Forward')
        if direction == "Reverse":
            return [(f"{base_key}_Rev", "Reverse")]
        if direction == "Both":
            return [(base_key, "Forward"), (f"{base_key}_Rev", "Reverse")]
        return [(base_key, "Forward")]

    def _iter_vehicle_step_sources(self):
        combos = [
            (self.params_A, self.raw_A, "System A", self.nodes_A, 'Vehicle Steps A', "Vehicle A", 'vehicle'),
            (self.params_A, self.raw_A, "System A", self.nodes_A, 'Vehicle Steps B', "Vehicle B", 'vehicleB')
        ]

        if self.valid_B:
            combos.extend([
                (self.params_B, self.raw_B, "System B", self.nodes_B, 'Vehicle Steps A', "Vehicle A", 'vehicle'),
                (self.params_B, self.raw_B, "System B", self.nodes_B, 'Vehicle Steps B', "Vehicle B", 'vehicleB')
            ])

        for params, raw, sys_label, nodes, base_key, veh_label, veh_param_key in combos:
            if not self._vehicle_has_loads(params, veh_param_key):
                continue
            for step_key, direction_label in self._vehicle_step_keys_for_direction(params, base_key):
                if raw and raw.get(step_key):
                    yield params, raw, sys_label, nodes, step_key, veh_label, direction_label

    def generate(self):
        # All plot exports share one persistent kaleido process; see
        # persistent_image_export for why this matters on kaleido 1.x.
        with persistent_image_export():
            self._generate_content()

    def _generate_content(self):
        self._update_progress(0.05)
        
        # 1. Cover / Metadata
        self._add_header_section()
        self.elements.append(PageBreak())
        
        # 2. Background Theory & Methodology
        self.elements.append(Paragraph(f"{self.chapter_count}. Basis of Analysis & Methodology", self.styles['SwecoSubHeader']))
        self._add_theory_section()
        self.elements.append(Spacer(1, 0.5*cm))
        self._add_conventions_text(self.params_A)
        self.elements.append(PageBreak())
        self.chapter_count += 1
        
        # 3. Global Analysis Settings
        self.elements.append(Paragraph(f"{self.chapter_count}. Global Analysis Settings", self.styles['SwecoSubHeader']))
        self._add_global_settings_summary()
        self.elements.append(PageBreak())
        self.chapter_count += 1
        
        # 4. Input Summary
        self.elements.append(Paragraph(f"{self.chapter_count}. System Configuration & Geometry", self.styles['SwecoSubHeader']))
        
        self._add_system_input_summary("System A", self.params_A, self.raw_A, self.props_A, "sysA")
        self.elements.append(PageBreak())
        
        if self.valid_B:
            self._add_system_input_summary("System B", self.params_B, self.raw_B, self.props_B, "sysB")
            self.elements.append(PageBreak())
            
        self.chapter_count += 1
        
        self._update_progress(0.15)
        
        # 5./6. Total envelopes per the selected limit states. The unfactored
        # combination chapter appears when neither limit state is analyzed.
        analyze_uls = bool(self.params_A.get('analyze_uls', True))
        analyze_sls = bool(self.params_A.get('analyze_sls', True))

        if analyze_uls:
            self.elements.append(Paragraph(f"{self.chapter_count}. Design Results (ULS) - Total Envelope", self.styles['SwecoSubHeader']))
            eq_txt = self._build_uls_equation_text()
            self.elements.append(Paragraph(f"<b>Formula:</b> {eq_txt}", self.styles['SwecoSmall']))
            self.elements.append(Spacer(1, 0.2*cm))
            self._add_results_section("Design (ULS)", prog_range=(0.15, 0.35))
            self.elements.append(PageBreak())
            self.chapter_count += 1
        else:
            self._update_progress(0.35)

        if analyze_sls:
            self.elements.append(Paragraph(f"{self.chapter_count}. Characteristic Results (SLS), including dynamic factor Phi where applicable", self.styles['SwecoSubHeader']))
            self.elements.append(Paragraph(f"<b>Formula:</b> {self._build_characteristic_formula_text()}", self.styles['SwecoSmall']))
            self.elements.append(Paragraph('Unfactored results (all loads at factor 1.0, no dynamic factor) are available in the interactive UI as "Unfactored".', self.styles['SwecoSmall']))
            self.elements.append(Spacer(1, 0.2*cm))
            self._add_results_section("Characteristic (SLS)", prog_range=(0.35, 0.50))
            self.elements.append(PageBreak())
            self.chapter_count += 1
        else:
            self._update_progress(0.50)

        if not analyze_uls and not analyze_sls:
            self.elements.append(Paragraph(f"{self.chapter_count}. Total Results (Unfactored)", self.styles['SwecoSubHeader']))
            self.elements.append(Paragraph(
                "<b>Formula:</b> all load cases combined with factor 1.0 and no dynamic factor "
                "(no limit state selected under Analysis & Result Settings).",
                self.styles['SwecoSmall']))
            self.elements.append(Spacer(1, 0.2*cm))
            self._add_results_section("Unfactored", prog_range=(0.15, 0.50))
            self.elements.append(PageBreak())
            self.chapter_count += 1
        
        # 7. Load Components (Unfactored)
        # Check active components across BOTH systems if valid_B, else just A
        has_sw = any(v > 0 for v in self.params_A['sw_list'])
        has_soil = len(self.params_A.get('soil', [])) > 0
        has_surch = len(self.params_A.get('surcharge', [])) > 0
        has_udl = data_mod.udl_line_load(self.params_A) > 0.0

        if self.valid_B:
            if any(v > 0 for v in self.params_B['sw_list']): has_sw = True
            if len(self.params_B.get('soil', [])) > 0: has_soil = True
            if len(self.params_B.get('surcharge', [])) > 0: has_surch = True
            if data_mod.udl_line_load(self.params_B) > 0.0: has_udl = True

        active_comps = sum([has_sw, has_soil, has_surch, has_udl])
        prog_start = 0.50
        prog_total_span = 0.25
        prog_step = prog_total_span / max(1, active_comps)
        
        if has_sw:
            self.elements.append(Paragraph(f"{self.chapter_count}. Load Case: Selfweight (Unfactored)", self.styles['SwecoSubHeader']))
            self._add_component_section("Selfweight", prog_range=(prog_start, prog_start + prog_step))
            self.elements.append(PageBreak())
            self.chapter_count += 1
            prog_start += prog_step
            
        if has_soil:
            self.elements.append(Paragraph(f"{self.chapter_count}. Load Case: Soil (Unfactored)", self.styles['SwecoSubHeader']))
            self._add_component_section("Soil", prog_range=(prog_start, prog_start + prog_step))
            self.elements.append(PageBreak())
            self.chapter_count += 1
            prog_start += prog_step
            
        if has_surch:
            self.elements.append(Paragraph(f"{self.chapter_count}. Load Case: Surcharge (Unfactored)", self.styles['SwecoSubHeader']))
            self._add_component_section("Surcharge", prog_range=(prog_start, prog_start + prog_step))
            self.elements.append(PageBreak())
            self.chapter_count += 1
            prog_start += prog_step

        if has_udl:
            self.elements.append(Paragraph(f"{self.chapter_count}. Load Case: Traffic UDL (Unfactored)", self.styles['SwecoSubHeader']))
            self.elements.append(Paragraph(
                "Adverse-only envelope of the uniformly distributed traffic load (traffic UDL), applied "
                "only in the unfavourable parts of the influence surface per EN 1991-2:2003, 4.3.2(1)(b). "
                "The load intensity includes the dynamic increment (DS/EN 1991-2 DK NA:2017, A.2.3.2), so the "
                "dynamic factor Phi is not applied. Being point-wise patterned, this case has no single "
                "applied load total and is therefore not part of the global equilibrium check.",
                self.styles['SwecoSmall']))
            self._add_component_section("Traffic UDL", prog_range=(prog_start, prog_start + prog_step))
            self.elements.append(PageBreak())
            self.chapter_count += 1
            prog_start += prog_step

        # 8. Critical Vehicle Steps
        if self._has_any_vehicle():
            self.elements.append(Paragraph(f"{self.chapter_count}. Critical Vehicle Steps (Unfactored)", self.styles['SwecoSubHeader']))

            self.elements.append(Paragraph(f"<b>Table {self.chapter_count}.1: Critical Vehicle Effects (Raw Step Values)</b>", self.styles['SwecoSmall']))
            self.elements.append(Paragraph("Values represent raw forward and/or reverse moving-load step effects before application of Partial Factors (Gamma, KFI) and Dynamic Factor (Phi). Vehicle A and Vehicle B are combined by independent moving-load envelope superposition.", self.styles['SwecoCell']))
            self._add_unfactored_vehicle_table()
            self.elements.append(Spacer(1, 0.4*cm))

            self.elements.append(Paragraph("<b>Vehicle Step Plots</b>", self.styles['SwecoBody']))
            self.elements.append(Paragraph(
                "Vehicle positions causing peak effects per span. Plotted values are the "
                "unfactored vehicle effects; where a Traffic UDL is defined, the shaded "
                "band marks the deck regions carrying the UDL for that step (the window "
                "around the vehicle is left clear). Step titles state the lead-axle "
                "chainage measured from the left end of the deck; values outside the "
                "deck range occur while the vehicle enters or leaves it.",
                self.styles['SwecoSmall']))
            self.elements.append(Spacer(1, 0.2*cm))
            self._add_smart_vehicle_steps(prog_range=(0.75, 0.95))
            self.chapter_count += 1
        else:
            self._update_progress(0.95)

        self._update_progress(0.98)

        self.executor.shutdown(wait=True)

        doc = SimpleDocTemplate(
            self.buffer, pagesize=A4,
            rightMargin=1.5*cm, leftMargin=1.5*cm,
            topMargin=2*cm, bottomMargin=2*cm
        )
        doc.build(self.elements, canvasmaker=NumberedCanvas)
        self._update_progress(1.0)

    def _add_conventions_text(self, params):
        """Adds audit-required conventions text."""
        conventions_text = f"""
        <b>Model Assumptions & Conventions:</b><br/>
        • <b>Coordinate System:</b> 2D Plane Frame (X: Horizontal, Y: Vertical, M: Counter-clockwise positive).<br/>
        • <b>Effective Width:</b> Analysis properties are calculated based on the effective width <i>b<sub>eff</sub></i>. Area <i>A = b<sub>eff</sub> · h</i>. Inertia <i>I = b<sub>eff</sub> · h<sup>3</sup> / 12</i>.<br/>
        • <b>Shear Area:</b> The shear area <i>A<sub>s</sub></i> is assumed to be <i>5/6 · A</i> (Rectangular section).<br/>
        • <b>Material Stiffness:</b> Shear Modulus <i>G = E / (2·(1+&nu;))</i>. <br/>
        • <b>Elastic Modulus:</b> The Modulus of Elasticity <i>E</i> (or <i>E<sub>cm</sub></i>) is calculated in accordance with DS/EN 1992-1-1 Table 3.1: <i>E = 22 · ((f<sub>ck</sub> + 8) / 10)<sup>0.3</sup></i> (where <i>f<sub>ck</sub></i> is in MPa and <i>E</i> is in GPa).<br/>
        • <b>Loads:</b> Gravity <i>g = 9.81 m/s²</i>. Vehicle loads (tonnes) are converted to kN using this factor.
        Gravity actions, including selfweight and vertical vehicle axle loads, are applied in the global vertical direction and transformed into local member axial and transverse components according to element orientation.
        Soil and surcharge loads are applied as line loads (kN/m) acting on the analysis strip.<br/>
        """
        self.elements.append(Paragraph(conventions_text, self.styles['SwecoSmall']))

    def _add_theory_section(self):
        """Adds standard background theory text, condensed to fit on one page."""
        styles = self.styles
        def add_sub(title, text):
            self.elements.append(Paragraph(f"<b>{title}</b>", styles['SwecoBody']))
            self.elements.append(Paragraph(text, styles['SwecoBody']))
            self.elements.append(Spacer(1, 0.2*cm))

        # Condensed 1.1 & 1.2
        add_sub("1.1 Calculation Method & Element Formulation",
            f"The analysis is performed using <b>BriCoS v{self.version}</b>, a 2D Matrix Stiffness FEM tool. "
            "For prismatic members, BriCoS uses the selected <b>Euler-Bernoulli</b> or "
            "<b>Timoshenko</b> formulation (with shear deformation via the parameter "
            "<i>&Phi;<sub>s</sub></i> = 12<i>EI</i> / (<i>GA<sub>s</sub>L</i><sup>2</sup>)). "
            "Non-prismatic members are handled using displacement-based Euler-Bernoulli "
            "integration of <i>EI(x)</i>; full non-prismatic Timoshenko shear deformation is "
            "not included in the current implementation. "
            "Material behavior is assumed Linear Elastic. "
            "Internal forces (M, V, N) include the exact contributions of member loads "
            "(fixed-end corrections and load discontinuities). For prismatic members they are "
            "independent of the mesh size. For non-prismatic members the section variation is "
            "represented per mesh sub-element with cubic displacement interpolation, so internal "
            "forces converge with mesh refinement rather than being mesh-independent; the default "
            "0.5 m mesh keeps this discretization effect small. "
            "Deflections are interpolated between nodes from the nodal displacements using Hermite "
            "shape functions; the local deflection of loads acting between nodes is not added. The "
            "resulting underestimate is bounded by approximately <i>P&middot;L<sub>mesh</sub></i><sup>3</sup>/(192<i>EI</i>) "
            "per point load, vanishes with mesh refinement, and is negligible at the default mesh "
            "size of 0.5 m.")

        # Condensed 1.3
        add_sub("1.2 Boundary Conditions", 
            "Supports are modeled using the Penalty Method with high-stiffness springs for Fixed/Pinned supports (<i>k</i> &approx; 10<sup>14</sup>) "
            "and discrete springs for elastic foundations based on user-specified stiffness (<i>K<sub>x</sub>, K<sub>y</sub>, K<sub>&theta;</sub></i>).")

        # Condensed 1.4
        add_sub("1.3 Moving Load Analysis",
            "Traffic actions are evaluated using a Quasi-Static algorithm, stepping the vehicle model across the structure to compute absolute maximum and minimum envelopes for forces and displacements. "
            "The Dynamic Amplification Factor (<i>&Phi;</i>) is calculated automatically based on the influence length (compliant with <b>DS/EN 1991-2 DK NA</b>) or defined manually. "
            "Where both Vehicle A and Vehicle B are defined, BriCoS uses independent moving-load envelope superposition: maxima/minima from Vehicle A and Vehicle B may be combined even when they occur at different moving-load positions.")

        # Condensed 1.5
        add_sub("1.4 Load Combinations",
            "Design values (<i>E<sub>d</sub></i>) are computed by superposition of factored envelopes: "
            "<i>E<sub>d</sub></i> = <i>K<sub>FI</sub></i> &middot; (<i>&gamma;<sub>G</sub>E<sub>SW</sub></i> + <i>&gamma;<sub>Soil</sub>E<sub>Soil</sub></i> + <i>&gamma;<sub>Q</sub>&Phi;E<sub>Veh</sub></i> + <i>&gamma;<sub>Q</sub>E<sub>Surch</sub></i>). "
            "Partial factors (<i>&gamma;</i>) and Consequence Class factor (<i>K<sub>FI</sub></i>) are applied as defined in settings. "
            "Traffic surcharge interaction is applied according to user selection (Exclusive or Simultaneous with vehicle load). "
            "Note that a single partial factor per permanent action is applied to both the maximum and minimum "
            "envelope values; favorable/unfavorable permanent-load combinations "
            "(e.g. <i>&gamma;<sub>G,inf</sub></i> / <i>&gamma;<sub>G,sup</sub></i>) are not evaluated automatically "
            "and should be verified with a separate analysis using the favorable factor.")

        # Condensed 1.6
        add_sub("1.5 Member Connectivity & Local Axes",
            "All element connections are modeled as fully rigid (no releases). Elements are defined along cross-section centerlines without eccentricities. "
            "<b>Local Coordinate Systems</b> for interpreting N and V:")
        
        bullets_axes = [
            "<b>Horizontal Members (Spans):</b> Local x-axis aligns with Global X. Thus, <b>N</b> represents horizontal axial force, and <b>V</b> represents vertical shear.",
            "<b>Vertical Members (Walls/Piers):</b> Local x-axis aligns with the member axis (Vertical). Thus, <b>N</b> represents vertical axial load, and <b>V</b> represents horizontal shear force."
        ]
        for b in bullets_axes: self.elements.append(Paragraph(f"• {b}", styles['SwecoBody']))

    def _build_uls_equation_text(self):
        def get_eq(p, raw):
            kfi = p.get('KFI', 1.0)
            gg = p.get('gamma_g', 1.0)
            gj = p.get('gamma_j', 1.0)
            phi_txt = self._phi_display_text(p, raw)
            g_veh = p.get('gamma_veh', 1.0)
            g_vehB = p.get('gamma_vehB', 1.0)
            has_A = bool(p.get('vehicle', {}).get('loads'))
            has_B = bool(p.get('vehicleB', {}).get('loads'))

            perm = f"{kfi}·{gg}·SW"
            if p.get('soil'): perm += f" + {kfi}·{gj}·Soil"

            var = ""
            if has_A: var += f" + {kfi}·{g_veh}·{phi_txt}·VehA"
            if has_B: var += f" + {kfi}·{g_vehB}·{phi_txt}·VehB"
            if data_mod.udl_line_load(p) > 0.0:
                var += f" + {kfi}·{p.get('gamma_udl', 0.56)}·UDL (no Phi)"
            if p.get('surcharge'): var += f" + {kfi}·{g_veh}·Surch"
            if (has_A or has_B) and "[" in phi_txt:
                var += " (Phi per member, see Dynamic Factor table)"
            return perm + var
            
        eqA = get_eq(self.params_A, self.raw_A)
        if not self.valid_B:
            return eqA
        
        eqB = get_eq(self.params_B, self.raw_B)
        if eqA == eqB: return f"Design = {eqA}"
        return f"SysA: {eqA} <br/> SysB: {eqB}"

    def _add_header_section(self):
        try:
            logo = Image(data_mod.resource_path("logo.png"), width=4*cm, height=1.5*cm)
            logo.hAlign = 'RIGHT'
            self.elements.append(logo)
        except:
            self.elements.append(Paragraph("[Sweco Logo Missing]", self.styles['Normal']))
            
        self.elements.append(Spacer(1, 1*cm))
        self.elements.append(Paragraph(f"BriCoS Analysis Report (v{self.version})", self.styles['Title']))
        self.elements.append(Spacer(1, 1*cm))
        
        data = [
            ["Project No:", self.meta.get('proj_no', ''), "Date:", datetime.date.today().strftime("%Y-%m-%d")],
            ["Project Name:", self.meta.get('proj_name', ''), "Revision:", self.meta.get('rev', '')],
            ["Author:", self.meta.get('author', ''), "Checker:", self.meta.get('checker', '')],
            ["Approver:", self.meta.get('approver', ''), "Analysis Ver:", f"v{self.version}"]
        ]
        
        t = Table(data, colWidths=[2.5*cm, 5.5*cm, 2.5*cm, 5.5*cm], hAlign='LEFT')
        t.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
            ('FONTNAME', (2,0), (2,-1), 'Helvetica-Bold'),
            ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.lightgrey),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        self.elements.append(t)
        self.elements.append(Spacer(1, 1*cm))
        
        if self.meta.get('comments', '').strip():
            self.elements.append(Paragraph("Comments:", self.styles['Heading3']))
            self.elements.append(Paragraph(self.meta['comments'], self.styles['SwecoBody']))
            self.elements.append(Spacer(1, 1*cm))

    def _add_global_settings_summary(self):
        p = self.params_A
        
        # Mapped "Both" to "Forwards & Backwards"
        v_dir = p.get('vehicle_direction', 'Forward')
        if v_dir == "Both": v_dir = "Forwards & Backwards"

        data = [
            ["Parameter", "Value", "Description"],
            ["Mesh Size", f"{p.get('mesh_size', 0.5)} m", "FE discretization length; governs deflection interpolation accuracy (see 1.1)"],
            ["Step Size", f"{p.get('step_size', 0.5)} m", "Vehicle moving load increment"],
            ["Vehicle Direction", f"{v_dir}", "Traffic flow direction"]
        ]
        
        use_shear = p.get('use_shear_def', False)
        shear_status = "Enabled for prismatic members (Timoshenko)" if use_shear else "Disabled (Euler-Bernoulli)"
        shear_desc = "Non-prismatic members use Euler-Bernoulli EI(x) integration" if use_shear else "Stiffness matrix formulation"
        data.append(["Shear Deformation", shear_status, shear_desc])
        
        # Use Paragraph to render HTML tags (subscript)
        lbl_beff = Paragraph("Effective Width (<i>b<sub>eff</sub></i>)", self.styles['SwecoBody'])
        data.append([lbl_beff, f"{p.get('b_eff', 1.0)} m", "Used for shear area & axial area estimation"])

        if use_shear:
            data.append(["Poisson's Ratio (ν)", f"{p.get('nu', 0.2)}", "Used for shear modulus G"])

        surch_txt = self._surcharge_interaction_text(p)
        data.append(["Surcharge Interaction", surch_txt, "Vehicle load & surcharge combination"])

        if not self._has_any_vehicle():
            data.append(["Vehicle Results", "No vehicle load models were defined. Moving-load envelope sections are omitted.", "Report content"])

        t = self._make_std_table(data, [4*cm, 5.5*cm, 6.5*cm])
        self.elements.append(KeepTogether([t]))

    def _draw_vehicle_stick_model(self, loads, spacing, width=400, height=72, udl_gap=None):
        # Fixed vertical bands from the bottom: caption (baseline 2), axle
        # spacing labels (baseline 12), dimension line (22), axle line (34),
        # arrows up to 50, load labels (baseline 55). Sized so no band
        # overlaps its neighbour (proportional placement used to draw the
        # caption across the dimension labels at small heights).
        height = max(height, 72)
        d = Drawing(width, height)
        if not loads or len(loads) == 0:
            d.add(String(width/2, height/2, "No Load Data", textAnchor='middle', fontSize=10, fillColor=colors.gray))
            return d

        cum_dist = np.cumsum(spacing)
        total_len = cum_dist[-1]
        draw_w = width * 0.8
        margin_x = width * 0.1

        if total_len < 0.1: scale_x = 0; offset_x = width / 2
        else: scale_x = draw_w / total_len; offset_x = margin_x

        y_axle_line = 34.0
        arrow_len = 16.0

        d.add(Line(margin_x - 10, y_axle_line, width - margin_x + 10, y_axle_line, strokeColor=colors.black, strokeWidth=1))

        if udl_gap is not None:
            # Schematic of the accompanying UDL fore and aft of the vehicle
            # with the clear-distance dimension (not to scale).
            band_col = colors.Color(0.18, 0.55, 0.34, alpha=0.5)
            band_h = 7
            for x0, x1 in ((2, margin_x - 28), (width - margin_x + 28, width - 2)):
                if x1 > x0:
                    d.add(Polygon(points=[x0, y_axle_line, x1, y_axle_line,
                                          x1, y_axle_line + band_h, x0, y_axle_line + band_h],
                                  fillColor=band_col, strokeWidth=0))
            d.add(String(margin_x - 19, y_axle_line + 10, f"{udl_gap:g} m",
                         textAnchor='middle', fontSize=7, fillColor=colors.Color(0.13, 0.4, 0.25)))
            d.add(String(width - margin_x + 19, y_axle_line + 10, f"{udl_gap:g} m",
                         textAnchor='middle', fontSize=7, fillColor=colors.Color(0.13, 0.4, 0.25)))
            d.add(String(width / 2, 2, "UDL accompanies the vehicle with the shown clear distance (not to scale)",
                         textAnchor='middle', fontSize=6, fillColor=colors.gray))
        
        for i, load_val in enumerate(loads):
            x_pos = offset_x + (cum_dist[i] if total_len > 0.1 else 0) * scale_x
            p = Polygon(points=[x_pos, y_axle_line, x_pos-3, y_axle_line+6, x_pos+3, y_axle_line+6], fillColor=colors.red, strokeWidth=0)
            d.add(p)
            d.add(Line(x_pos, y_axle_line, x_pos, y_axle_line + arrow_len, strokeColor=colors.red, strokeWidth=2))
            label = f"{load_val}t"
            d.add(String(x_pos, y_axle_line + arrow_len + 5, label, textAnchor='middle', fontSize=8, fillColor=colors.red))
            d.add(Group(Polygon(points=[x_pos-2, y_axle_line-2, x_pos+2, y_axle_line-2, x_pos+2, y_axle_line+2, x_pos-2, y_axle_line+2], 
                                fillColor=colors.black, strokeWidth=0)))

        dim_y = y_axle_line - 12
        for i in range(len(spacing)):
            if i == 0: continue
            dist = spacing[i]
            x_prev = offset_x + cum_dist[i-1] * scale_x
            x_curr = offset_x + cum_dist[i] * scale_x
            d.add(Line(x_prev, dim_y, x_curr, dim_y, strokeColor=colors.blue, strokeWidth=0.5))
            d.add(Line(x_prev, dim_y-2, x_prev, dim_y+2, strokeColor=colors.blue, strokeWidth=0.5))
            d.add(Line(x_curr, dim_y-2, x_curr, dim_y+2, strokeColor=colors.blue, strokeWidth=0.5))
            mid_x = (x_prev + x_curr) / 2
            d.add(String(mid_x, dim_y - 10, f"{dist}m", textAnchor='middle', fontSize=7, fillColor=colors.blue))

        return d

    def _get_geometry_description(self, p, prefix, idx, simple_list_key):
        geom_key = f"{prefix}_geom_{idx}"
        val_default = p[simple_list_key][idx]
        
        desc = f"H = {val_default:.3f}m"
        
        if geom_key in p:
            g = p[geom_key]
            is_height = (g.get('type', 1) == 1)
            lbl = "H" if is_height else "I"
            fmt = "{:.3f}m" if is_height else "{:.4e}"
            shape = g.get('shape', 0)
            vals = g.get('vals', [0.0, 0.0, 0.0])
            safe_vals = [v if (v is not None and not np.isnan(v)) else 0.0 for v in vals]
            
            v1 = fmt.format(safe_vals[0])
            v2 = fmt.format(safe_vals[1])
            v3 = fmt.format(safe_vals[2])
            
            if shape == 0: desc = f"{lbl} = {v1}"
            elif shape == 1: desc = f"{lbl}: {v1} -> {v3} (Taper)"
            elif shape == 2: desc = f"{lbl}: {v1} -> {v2} -> {v3} (3-Pt)"
                
            if prefix == 'span' and g.get('align_type') == 1:
                mode = g.get('incline_mode', 0)
                val = g.get('incline_val', 0.0)
                if val is None or np.isnan(val): val = 0.0
                inc_txt = f"{val:.2f}%" if mode == 0 else f"{val:.2f}m"
                desc += f"<br/>Slope: {inc_txt}"
        return desc

    @staticmethod
    def _udl_application_text(p, sys_has_vehicle):
        """Step-result application wording for the Traffic UDL line."""
        if p.get('udl_mode') == 'Static':
            return "static full deck at every step"
        if not sys_has_vehicle:
            return ("no vehicle load model in this system, so no step results exist; "
                    "the UDL enters the Total Envelope as the full adverse envelope")
        if p.get('udl_footprint', False):
            return "moving with the vehicle, also applied within the vehicle window"
        return (f"moving with the vehicle, clear distance "
                f"{p.get('udl_gap', 10.0):.2f} m in front of and behind the axles "
                "(the excluded window snaps inward to whole mesh segments, "
                "erring on the loaded side - conservative)")

    def _add_system_input_summary(self, sys_label, p, raw_res, props, sys_key_id):
        self.elements.append(Paragraph(f"<b>{sys_label} ({p.get('name', '')})</b> - {p['mode']}", self.styles['Heading3']))

        sys_has_vehicle = self._has_vehicle_loads(p)
        if not sys_has_vehicle:
            # Phi only multiplies vehicle effects; a value would be noise here.
            phi_txt = "n/a (no vehicle load model in this system)"
        elif p.get('phi_mode') == 'Calculate' and raw_res:
            method = "per span" if p.get('phi_linf_mode') == 'Span' else "combined system"
            phi_txt = f"{self._phi_display_text(p, raw_res)} (Calc, {method})"
        else:
            scope = "per span" if p.get('phi_manual_scope') == 'Per span' else "global"
            phi_txt = f"{self._phi_display_text(p, raw_res)} (Manual, {scope})"
        sls_mode = p.get('phi_sls_mode', 'Same')
        if sys_has_vehicle and sls_mode == 'Reduced':
            phi_txt += " | SLS reduced per Vejl. 5.4.2"
        elif sys_has_vehicle and sls_mode == 'Manual':
            phi_txt += f" | SLS manual = {p.get('phi_sls', 1.0):.3f}"

        analyze_uls = bool(p.get('analyze_uls', True))
        analyze_sls = bool(p.get('analyze_sls', True))
        ls_txt = ("ULS and SLS" if analyze_uls and analyze_sls else
                  "ULS only" if analyze_uls else
                  "SLS only" if analyze_sls else
                  "none (unfactored combination only)")
        self.elements.append(Paragraph(
            f"<b>Limit states analyzed:</b> {ls_txt} | <b>Phi:</b> {phi_txt}",
            self.styles['SwecoBody']))

        # Combination factor table: one row per load component, the ULS
        # partial factor (multiplied by KFI) and the SLS combination factor.
        kfi = p.get('KFI', 1.0)
        uls_col = f"ULS factor (x KFI = {kfi})" if analyze_uls else "ULS factor (not analyzed)"
        sls_col = "SLS factor" if analyze_sls else "SLS factor (not analyzed)"
        fact_rows = [["Load component", uls_col, sls_col, "Dynamic factor"]]
        fact_rows.append(["Selfweight", f"{p.get('gamma_g', 1.0)}", f"{p.get('sls_g', 1.0)}", "-"])
        if p.get('soil'):
            fact_rows.append(["Soil", f"{p.get('gamma_j', 1.0)}", f"{p.get('sls_j', 1.0)}", "-"])
        if bool(p.get('vehicle', {}).get('loads')):
            fact_rows.append(["Vehicle A", f"{p.get('gamma_veh', 1.0)}", f"{p.get('sls_veh', 1.0)}", "Phi applied"])
        if bool(p.get('vehicleB', {}).get('loads')):
            fact_rows.append(["Vehicle B", f"{p.get('gamma_vehB', 1.0)}", f"{p.get('sls_vehB', 1.0)}", "Phi applied"])
        udl_line = data_mod.udl_line_load(p)
        if udl_line > 0.0:
            fact_rows.append(["Traffic UDL", f"{p.get('gamma_udl', 0.56)}", f"{p.get('sls_udl', 0.40)}",
                              "not applied (intensity includes the dynamic increment, DK NA A.2.3.2)"])
        if p.get('surcharge'):
            fact_rows.append(["Surcharge", f"{p.get('gamma_veh', 1.0)} (= Vehicle A)", f"{p.get('sls_veh', 1.0)} (= Vehicle A)", "not applied (static)"])
        t = self._make_std_table(fact_rows, [3.2*cm, 4.4*cm, 4.0*cm, 5.6*cm], font_size=8)
        self.elements.append(KeepTogether([t]))

        if udl_line > 0.0:
            app_txt = self._udl_application_text(p, sys_has_vehicle)
            self.elements.append(Paragraph(
                f"<b>Traffic UDL:</b> q = {udl_line:.2f} kN/m "
                "(line load on the analysis strip; loaded width considered in the input) | "
                "adverse-only application (EN 1991-2, 4.3.2(1)(b)) | "
                f"step results: {app_txt}. The Total Envelope combines the vehicle envelope "
                "with the full adverse UDL envelope by independent superposition, which "
                "bounds every window arrangement including the vehicle-absent situation.",
                self.styles['SwecoBody']))
        self.elements.append(Spacer(1, 0.2*cm))
        
        # Widened Material Column
        w_id, w_dim, w_load, w_mat = 1.5*cm, 2.0*cm, 2.0*cm, 4.5*cm 
        page_width = 18.0*cm
        w_geom = page_width - (w_id + w_dim + w_load + w_mat)
        col_widths = [w_id, w_dim, w_geom, w_load, w_mat]

        # 1. SPANS
        span_data = [["Span", "L [m]", "Section Geometry", "SW [kN/m]", "Material / E"]]
        for i in range(p['num_spans']):
            if p['L_list'][i] > 0.001:
                pid = f"S{i+1}"
                e_real = props['Spans'].get(pid, {}).get('E', 0.0) / 1e6 
                
                if p['e_mode'] == 'Eurocode':
                    fck = p['fck_span_list'][i]
                else:
                    fck = 0 
                    
                if p['e_mode'] == 'Eurocode':
                    mat_str = f"fck = {fck:.0f} MPa / E = {e_real:.0f} GPa"
                else:
                    mat_str = f"Custom (E = {e_real:.0f} GPa)"
                
                geom_desc = self._get_geometry_description(p, 'span', i, 'Is_list')
                geom_flowable = Paragraph(geom_desc, self.styles['SwecoCell'])
                span_data.append([
                    f"S{i+1}", f"{p['L_list'][i]:.2f}", geom_flowable, 
                    f"{p['sw_list'][i]:.1f}", mat_str
                ])
        if len(span_data) > 1:
            t = self._make_std_table(span_data, col_widths)
            self.elements.append(KeepTogether([t]))
        else:
            self.elements.append(Paragraph("No spans defined.", self.styles['SwecoBody']))

        # 2. WALLS
        if p['mode'] == 'Frame':
            self.elements.append(Spacer(1, 0.2*cm))
            wall_data = [["Wall", "H [m]", "Section Geometry", "Surch [kN/m]", "Material / E"]]
            has_wall = False
            for i in range(p['num_spans'] + 1):
                if p['h_list'][i] > 0.001:
                    has_wall = True
                    pid = f"W{i+1}"
                    e_real = props['Walls'].get(pid, {}).get('E', 0.0) / 1e6
                    
                    if p['e_mode'] == 'Eurocode':
                        fck = p['fck_wall_list'][i]
                        mat_str = f"fck = {fck:.0f} MPa / E = {e_real:.0f} GPa"
                    else:
                        mat_str = f"Custom (E = {e_real:.0f} GPa)"

                    sur = next((x['q'] for x in p.get('surcharge', []) if x['wall_idx']==i), 0.0)
                    geom_desc = self._get_geometry_description(p, 'wall', i, 'Iw_list')
                    geom_flowable = Paragraph(geom_desc, self.styles['SwecoCell'])
                    wall_data.append([
                        f"W{i+1}", f"{p['h_list'][i]:.2f}", geom_flowable, 
                        f"{sur:.1f}", mat_str
                    ])
            if has_wall: 
                t = self._make_std_table(wall_data, col_widths)
                self.elements.append(KeepTogether([t]))

        # 3. SUPPORTS
        self.elements.append(Spacer(1, 0.2*cm))
        supp_data = [["Support Node", "Type", "Stiffness (Kx, Ky, Km) [kN/m, kN/m, kNm/rad]"]]
        has_supp = False
        supp_list = p.get('supports', [])
        num_expected = p['num_spans'] + 1
        for i in range(num_expected):
            lbl = f"Wall {i+1} Base" if p['mode'] == 'Frame' else f"Support {i+1}"
            s_type = "Fixed"
            k_vals = [1e14, 1e14, 1e14]
            if i < len(supp_list):
                s_type = supp_list[i].get('type', 'Fixed')
                k_vals = supp_list[i].get('k', [1e14, 1e14, 1e14])
            else:
                 if p['mode'] != 'Frame':
                      if i == 0: s_type, k_vals = "Pinned", [1e14, 1e14, 0.0]
                      else: s_type, k_vals = "Roller X", [0.0, 1e14, 0.0]
            k_str = f"[{k_vals[0]:.1e}, {k_vals[1]:.1e}, {k_vals[2]:.1e}]"
            supp_data.append([lbl, s_type, k_str])
            has_supp = True
        if has_supp:
            t = self._make_std_table(supp_data, [4*cm, 4*cm, 7*cm])
            self.elements.append(KeepTogether([
                Paragraph("Boundary Conditions:", self.styles['SwecoSmall']), t]))

        # 4. SOIL
        soil_list = p.get('soil', [])
        if soil_list:
            self.elements.append(Spacer(1, 0.2*cm))
            soil_table = [["Wall", "Face", "Height [m]", "q_top [kN/m]", "q_bot [kN/m]"]]
            for s in soil_list:
                soil_table.append([
                    f"W{s['wall_idx']+1}", s['face'], f"{s['h']:.2f}", f"{s['q_top']:.1f}", f"{s['q_bot']:.1f}"
                ])
            t = self._make_std_table(soil_table, [2*cm, 2*cm, 3*cm, 3*cm, 3*cm])
            self.elements.append(KeepTogether([
                Paragraph("Soil Loads (Earth Pressure):", self.styles['SwecoSmall']), t]))

        # 5. VEHICLES
        self.elements.append(Spacer(1, 0.2*cm))
        def add_veh_table(key, title_suffix, prefix):
            veh = p.get(key, {})
            v_loads = veh.get('loads', [])
            v_spac = veh.get('spacing', [])
            
            if v_loads:
                # Match current configuration against standard vehicles.csv
                class_name = self._match_vehicle_class(v_loads, v_spac)
                
                header_text = f"Vehicle {title_suffix}: {class_name}"
                if class_name != "Custom":
                    header_text += " - In accordance with DS/EN 1991-2, DK:NA (bridges):2017"
                
                self.elements.append(Paragraph(header_text, self.styles['SwecoSmall']))
                udl_gap_draw = None
                if (data_mod.udl_line_load(p) > 0.0 and p.get('udl_mode') != 'Static'
                        and not p.get('udl_footprint', False)):
                    udl_gap_draw = float(p.get('udl_gap', 10.0))
                drawing = self._draw_vehicle_stick_model(v_loads, v_spac, width=400, height=72, udl_gap=udl_gap_draw)
                self.elements.append(drawing)
                self.elements.append(Spacer(1, 0.3*cm))
        
        add_veh_table('vehicle', "A", "vehA")
        add_veh_table('vehicleB', "B", "vehB")

        # Vehicles are defined per system; a one-sided definition is easy to
        # miss in the sidebar (it edits the active system only), so call it
        # out where the vehicle tables would have been.
        if not sys_has_vehicle and self.valid_B:
            other_p = self.params_B if sys_key_id == "sysA" else self.params_A
            other_lbl = "System B" if sys_key_id == "sysA" else "System A"
            if self._has_vehicle_loads(other_p):
                self.elements.append(Paragraph(
                    f"<b>Note:</b> no vehicle load model is defined for {sys_label}, "
                    f"while {other_lbl} includes one. Moving-load envelopes and "
                    "critical vehicle steps are reported only for systems with a "
                    "vehicle definition.",
                    self.styles['SwecoBody']))
                self.elements.append(Spacer(1, 0.2*cm))

        # 6. DYNAMIC FACTOR (PHI)
        if raw_res:
            if sys_has_vehicle:
                self._add_dynamic_factor_section(p, raw_res)
            else:
                self.elements.append(Spacer(1, 0.2*cm))
                self.elements.append(Paragraph(
                    "Dynamic Factor (<i>Φ</i>): not applicable - no vehicle "
                    "load model in this system.",
                    self.styles['SwecoSmall']))

        # 7. GLOBAL EQUILIBRIUM CHECK
        if raw_res and raw_res.get('Equilibrium'):
            self._add_equilibrium_section(raw_res['Equilibrium'])

    @staticmethod
    def _equilibrium_rows(equilibrium):
        """Table rows for the global equilibrium check.

        Cases without any load definition show "(no loads)". Cases whose
        loads cancel in the global sums (e.g. mirrored earth pressure on
        both walls) still get the residual check: the reactions must cancel
        too, so PASS/FAIL remains meaningful at zero sums.
        """
        rows = [["Load case", "Sum applied Fx / Fy [kN]", "Sum reactions Rx / Ry [kN]", "Residual Fx / Fy [kN]", "Status"]]
        for case in ('Selfweight', 'Soil', 'Surcharge'):
            eq = equilibrium.get(case)
            if not eq:
                continue
            a_x, a_y = eq['applied_x'], eq['applied_y']
            r_x, r_y = eq['reactions_x'], eq['reactions_y']
            res_x, res_y = eq['residual_x'], eq['residual_y']
            has_loads = eq.get('has_loads')
            if has_loads is None:
                # Legacy raw results without the flag: infer from magnitudes.
                has_loads = max(abs(a_x), abs(a_y), abs(r_x), abs(r_y)) >= 1e-9
            if not has_loads:
                rows.append([case, "0.00 / 0.00", "0.00 / 0.00", "-", "(no loads)"])
                continue
            scale = max(abs(a_x), abs(a_y), abs(r_x), abs(r_y), 1.0)
            tol = max(1e-3, 1e-6 * scale)
            ok = abs(res_x) <= tol and abs(res_y) <= tol
            rows.append([
                case,
                f"{a_x:.2f} / {a_y:.2f}",
                f"{r_x:.2f} / {r_y:.2f}",
                f"{res_x:.2e} / {res_y:.2e}",
                "PASS" if ok else "CHECK FAILED",
            ])
        return rows

    def _add_equilibrium_section(self, equilibrium):
        """Applied loads vs support reactions per unfactored static case."""
        rows = self._equilibrium_rows(equilibrium)
        heading = Paragraph("Global Equilibrium Check:", self.styles['SwecoSmall'])
        explanation = Paragraph(
            "Sum of applied loads (computed from the load definitions by simple statics) compared "
            "with the sum of support reactions (boundary-spring forces) for each unfactored static "
            "load case, in global axes. A vanishing residual verifies load assembly, load splitting "
            "across the mesh and the solution itself. Opposing load definitions (e.g. symmetric "
            "earth pressure on both walls) may cancel to zero sums; the residual comparison "
            "remains valid.",
            self.styles['SwecoCell'])
        t = self._make_std_table(rows, [2.8*cm, 4.3*cm, 4.3*cm, 4.0*cm, 2.6*cm], font_size=8)
        self.elements.append(Spacer(1, 0.2*cm))
        # Keep the heading and explanation with the table so a page break
        # cannot strand them at the bottom of the previous page.
        self.elements.append(KeepTogether([heading, explanation, t]))

    def _add_dynamic_factor_section(self, p, raw_res):
        """Methodology statement, per-member phi table (ULS and SLS values),
        and the calculation log for the dynamic factor."""
        self.elements.append(Spacer(1, 0.2*cm))
        self.elements.append(Paragraph("Dynamic Factor (<i>Φ</i>):", self.styles['SwecoSmall']))

        # Methodology statement with clause references.
        if p.get('phi_mode') != 'Calculate':
            if p.get('phi_manual_scope') == 'Per span':
                method_txt = ("Manual input by the user, per span; walls take the max of "
                              "the adjacent spans' <i>Φ</i>.")
            else:
                method_txt = "Manual input by the user (single value for all members)."
        elif p.get('phi_linf_mode') == 'Span':
            app = ("applied per member; walls use the max of the adjacent spans"
                   if p.get('phi_application') != 'Governing'
                   else "governing (max) value applied to all members")
            method_txt = (
                "<i>L<sub>inf</sub></i> = actual span per DS/EN 1991-2 DK NA:2017, "
                f"Annex A, A.2.3.5(2); {app}."
            )
        else:
            method_txt = (
                "<i>L<sub>inf</sub></i> = determinant length of the combined system per "
                "DS/EN 1991-2:2003, Table 6.2, Case 5.1/5.2/5.3 (renumbered Table 8.2 in the "
                "2023 edition); <i>Φ</i> per DS/EN 1991-2 DK NA:2017, Annex A, A.2.3.5(2)."
            )
        self.elements.append(Paragraph(method_txt, self.styles['SwecoFormula']))

        sls_mode = p.get('phi_sls_mode', 'Same')
        if sls_mode == 'Reduced':
            self.elements.append(Paragraph(
                "SLS reduction enabled: <i>Φ<sub>SLS</sub></i> = 1 + (<i>Φ<sub>ULS</sub></i> - 1)/2 "
                "per Vejledning til belastnings- og beregningsgrundlag for broer, 5.4.2.",
                self.styles['SwecoFormula']))
        elif sls_mode == 'Manual':
            self.elements.append(Paragraph(
                f"SLS: user-defined uniform <i>Φ<sub>SLS</sub></i> = {p.get('phi_sls', 1.0):.3f} "
                "replaces all member values in the Characteristic (SLS) result mode.",
                self.styles['SwecoFormula']))
        self.elements.append(Spacer(1, 0.15*cm))

        # Phi value table: per member when available, otherwise one row.
        # Per-member values exist for the calculated span-based methodology
        # and for manual per-span input alike.
        phi_uniform = raw_res.get('phi_calc', 1.0) if p.get('phi_mode') == 'Calculate' else p.get('phi', 1.0)
        members = raw_res.get('Phi Members') or {}

        def fmt_row(label, val):
            if sls_mode == 'Reduced':
                sls_note = f"{1.0 + (val - 1.0) / 2.0:.3f}"
            elif sls_mode == 'Manual':
                sls_note = f"{p.get('phi_sls', 1.0):.3f} (manual)"
            else:
                sls_note = f"{val:.3f} (no reduction)"
            return [label, f"{val:.3f}", sls_note]

        phi_table = [["Member", "Phi (ULS)", "Phi (SLS)"]]
        if members:
            for eid in sorted(members.keys(), key=lambda x: (x[0], int(x[1:]))):
                phi_table.append(fmt_row(eid, members[eid]))
        else:
            phi_table.append(fmt_row("All members", phi_uniform))
        t = self._make_std_table(phi_table, [4*cm, 3.5*cm, 4.5*cm], font_size=8)
        self.elements.append(KeepTogether([t]))

        # Calculation log.
        if raw_res.get('phi_log'):
            formatted_lines = []
            for line in raw_res['phi_log']:
                # Use Unicode directly instead of entities to avoid & display issues
                txt = line.replace("L_phi", "<i>L<sub>Φ</sub></i>")\
                          .replace("L_mean", "<i>L<sub>mean</sub></i>")\
                          .replace("L_inf", "<i>L<sub>inf</sub></i>")\
                          .replace("Phi", "<i>Φ</i>")
                formatted_lines.append(txt)

            for line in formatted_lines:
                # Use SwecoLog (leading=14) to prevent overlap
                self.elements.append(Paragraph(f"• {line}", self.styles['SwecoLog']))

    # -----------------------------------------------
    # PARALLEL RENDERING HELPER (WITH PROGRESS)
    # -----------------------------------------------
    def _render_plot_task(self, fig_kwargs):
        """Executed in ThreadPool to offload Plotly I/O."""
        # Export resolution: grid images are placed at 8 cm width in the
        # PDF, where the default 700 px figure already gives >200 dpi at
        # scale 1.0; the full-width step plots (16 cm) keep scale 1.5.
        # Export time scales with the pixel count.
        export_scale = fig_kwargs.pop('export_scale', 1.5)
        try:
            fig = viz.create_plotly_fig(**fig_kwargs)
            b = io.BytesIO()
            # Figure construction runs in parallel across the pool, but the
            # PNG export is serialized: kaleido 0.2.x shares a single global
            # scope that is not thread-safe (the source of intermittently
            # corrupted images), and the kaleido 1.x sync server processes
            # one request at a time anyway.
            with self._image_export_lock:
                fig.write_image(b, format='png', scale=export_scale)
            b.seek(0)
            if not b.getvalue().startswith(b'\x89PNG'):
                _logger.error("Plot export produced a corrupt image for %r",
                              fig_kwargs.get('title', '?'))
                return None
            return b
        except Exception:
            # The placeholder "[Image Failed]" appears in the PDF; without a
            # log line the cause is undiagnosable.
            _logger.exception("Plot rendering failed for %r",
                              fig_kwargs.get('title', '?'))
            return None

    def _submit_parallel_plots(self, task_list, prog_range=(0.0, 0.0)):
        if not task_list:
            return []
            
        futures = {}
        for i, kwargs in enumerate(task_list):
            f = self.executor.submit(self._render_plot_task, kwargs)
            futures[f] = i
        
        results = [None] * len(task_list)
        total_tasks = len(task_list)
        completed_count = 0
        start_p, end_p = prog_range
        prog_span = end_p - start_p
        
        for f in as_completed(futures):
            idx = futures[f]
            try:
                results[idx] = f.result()
            except Exception:
                results[idx] = None
            
            completed_count += 1
            if total_tasks > 0:
                current_fraction = completed_count / total_tasks
                new_val = start_p + (prog_span * current_fraction)
                self._update_progress(new_val)

        return results

    def _add_results_section(self, res_mode, prog_range=(0.0, 0.0)):
        res_A = self._combined('A', res_mode)
        res_B = {}
        if self.valid_B:
            res_B = self._combined('B', res_mode)
        
        self.elements.append(Paragraph(f"Visualizations - {res_mode}", self.styles['Heading4']))
        
        tasks = []
        types = [('M', 'Bending Moment [kNm]'), ('V', 'Shear Force [kN]'), 
                 ('N', 'Normal Force [kN]'), ('Def', 'Deformation [mm]')]
        
        for t_code, t_title in types:
            tasks.append({
                'nodes': self.nodes_A, 'sysA_data': res_A.get("Total Envelope", {}), 'sysB_data': {},
                'type_base': t_code, 'title': f"{t_title} - {self.params_A['name']}", 
                'load_case_name': "Total Envelope",
                'name_A': self.params_A['name'], 'name_B': self.params_B['name'],
                'geom_A': self.raw_A.get('Selfweight'), 'geom_B': None,
                'params_A': self.params_A, 'params_B': self.params_B,
                'show_A': True, 'show_B': False, 'show_supports': True, 'font_scale': 1.5,
                'export_scale': 1.0
            })
            
        if self.valid_B:
            for t_code, t_title in types:
                tasks.append({
                    'nodes': self.nodes_B, 'sysA_data': {}, 'sysB_data': res_B.get("Total Envelope", {}),
                    'type_base': t_code, 'title': f"{t_title} - {self.params_B['name']}", 
                    'load_case_name': "Total Envelope",
                    'name_A': self.params_A['name'], 'name_B': self.params_B['name'],
                    'geom_A': None, 'geom_B': self.raw_B.get('Selfweight'),
                    'params_A': self.params_A, 'params_B': self.params_B,
                    'show_A': False, 'show_B': True, 'show_supports': True, 'font_scale': 1.5,
                    'export_scale': 1.0
                })

        images = self._submit_parallel_plots(tasks, prog_range)

        self._append_image_grid(images[0:4], heading="System A")

        if self.valid_B:
            self.elements.append(Spacer(1, 0.3*cm))
            self._append_image_grid(images[4:8], heading="System B")

        self.elements.append(Spacer(1, 0.5*cm))

        title_str = f"Tabular Summary - {res_mode.upper()}"
        self.elements.append(Paragraph(title_str, self.styles['Heading3']))
        
        self._add_force_summary_table(res_A['Total Envelope'], res_B.get('Total Envelope', {}))
        self.elements.append(Spacer(1, 0.3*cm))
        self._add_reaction_table(res_A, self.params_A, res_B, self.params_B)

    def _add_component_section(self, load_key, prog_range=(0.0, 0.0)):
        res_A = self._combined('A', "Unfactored")
        res_B = {}
        if self.valid_B:
             res_B = self._combined('B', "Unfactored")
        
        tasks = []
        types = [('M', 'Bending Moment [kNm]'), ('V', 'Shear Force [kN]'), 
                 ('N', 'Normal Force [kN]'), ('Def', 'Deformation [mm]')]
        
        for t_code, t_title in types:
            tasks.append({
                'nodes': self.nodes_A, 'sysA_data': res_A.get(load_key, {}), 'sysB_data': {},
                'type_base': t_code, 'title': f"{t_title} - {self.params_A['name']}", 
                'load_case_name': load_key,
                'name_A': self.params_A['name'], 'name_B': self.params_B['name'],
                'geom_A': self.raw_A.get('Selfweight'), 'geom_B': None,
                'params_A': self.params_A, 'params_B': self.params_B,
                'show_A': True, 'show_B': False, 'show_supports': True, 'font_scale': 1.5,
                'export_scale': 1.0
            })
        
        if self.valid_B:
            for t_code, t_title in types:
                tasks.append({
                    'nodes': self.nodes_B, 'sysA_data': {}, 'sysB_data': res_B.get(load_key, {}),
                    'type_base': t_code, 'title': f"{t_title} - {self.params_B['name']}", 
                    'load_case_name': load_key,
                    'name_A': self.params_A['name'], 'name_B': self.params_B['name'],
                    'geom_A': None, 'geom_B': self.raw_B.get('Selfweight'),
                    'params_A': self.params_A, 'params_B': self.params_B,
                    'show_A': False, 'show_B': True, 'show_supports': True, 'font_scale': 1.5,
                    'export_scale': 1.0
                })

        images = self._submit_parallel_plots(tasks, prog_range)

        self._append_image_grid(images[0:4], heading="System A")

        if self.valid_B:
            self.elements.append(Spacer(1, 0.3*cm))
            self._append_image_grid(images[4:8], heading="System B")

        self.elements.append(Spacer(1, 0.5*cm))
        self._add_force_summary_table(res_A.get(load_key, {}), res_B.get(load_key, {}))
        self.elements.append(Spacer(1, 0.3*cm))
        
        wrap_A = {'Total Envelope': res_A.get(load_key, {})}
        wrap_B = {'Total Envelope': res_B.get(load_key, {})} if self.valid_B else {}
        self._add_reaction_table(wrap_A, self.params_A, wrap_B, self.params_B)

    def _append_image_grid(self, img_bytes_list, heading=None):
        img_flowables = []
        for b in img_bytes_list:
            if b:
                try:
                    img = Image(b, width=8*cm, height=5*cm)
                    img_flowables.append(img)
                except Exception:
                    img_flowables.append(Paragraph("[Image Corrupt]", self.styles['SwecoSmall']))
            else:
                img_flowables.append(Paragraph("[Image Failed]", self.styles['SwecoSmall']))

        if img_flowables:
            rows = []
            for i in range(0, len(img_flowables), 2):
                rows.append(img_flowables[i:i+2])
            if len(rows[-1]) == 1:
                rows[-1].append(Paragraph("", self.styles['Normal']))

            t = Table(rows, colWidths=[8.5*cm, 8.5*cm], hAlign='LEFT')
            t.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('ALIGN', (0,0), (-1,-1), 'CENTER')]))
            # Keep the heading with the grid: appended separately, a page
            # break can strand it alone at the bottom of the previous page.
            parts = [t]
            if heading:
                parts.insert(0, Paragraph(heading, self.styles['SwecoSmall']))
            self.elements.append(KeepTogether(parts))

    def _add_unfactored_vehicle_table(self):
        """Adds table showing raw vehicle effects (Unfactored) for validation."""
        data = [["Elem", "Direction", "M_max", "M_min", "V_max", "V_min", "System"]]

        def process_steps(steps, sys_name, direction_label):
            if not steps:
                return
            all_ids = set()
            for step in steps:
                all_ids.update(step.get('res', {}).keys())
            all_ids = sorted(all_ids, key=lambda x: (x[0], int(x[1:])))

            for eid in all_ids:
                max_M, min_M = -1e15, 1e15
                max_V, min_V = -1e15, 1e15
                found = False
                for step in steps:
                    elem_res = step.get('res', {}).get(eid)
                    if not elem_res:
                        continue
                    found = True
                    max_M = max(max_M, float(np.max(elem_res['M'])))
                    min_M = min(min_M, float(np.min(elem_res['M'])))
                    max_V = max(max_V, float(np.max(elem_res['V'])))
                    min_V = min(min_V, float(np.min(elem_res['V'])))
                if found:
                    data.append([eid, direction_label, f"{max_M:.1f}", f"{min_M:.1f}", f"{max_V:.1f}", f"{min_V:.1f}", sys_name])

        for _, raw, sys_label, _, step_key, veh_label, direction_label in self._iter_vehicle_step_sources():
            process_steps(raw.get(step_key, []), f"{sys_label} ({veh_label})", direction_label)

        if len(data) > 1:
            t = self._make_std_table(data, [1.8*cm, 2.2*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 3.0*cm])
            self.elements.append(KeepTogether([t]))
        else:
            self.elements.append(Paragraph("No vehicle step results found for the active vehicle load models.", self.styles['SwecoSmall']))

    def _add_smart_vehicle_steps(self, prog_range=(0.0, 0.0)):
        all_task_groups = []
        
        for p, r, s_lbl, n, step_key, veh_lbl, direction_label in self._iter_vehicle_step_sources():
            g = self._identify_critical_steps(p, r, s_lbl, n, step_key, veh_lbl, direction_label)
            if g:
                all_task_groups.append({
                    'main_header': f"{s_lbl} - {veh_lbl} ({direction_label})",
                    'groups': g
                })

        all_render_configs = []
        
        for section in all_task_groups:
            for group in section['groups']:
                for plot_req in group['plots']:
                    all_render_configs.append(plot_req['config'])

        if all_render_configs:
            rendered_images = self._submit_parallel_plots(all_render_configs, prog_range)
        else:
            rendered_images = []
        
        img_cursor = 0
        
        for section in all_task_groups:
            self.elements.append(Paragraph(f"<b>{section['main_header']}</b>", self.styles['Heading4']))
            
            for group in section['groups']:
                self.elements.append(Paragraph(f"<b>{group['header']}</b>", self.styles['SwecoBody']))
                for plot_req in group['plots']:
                    if img_cursor < len(rendered_images):
                        img_data = rendered_images[img_cursor]
                        img_cursor += 1
                        
                        if img_data:
                            img = Image(img_data, width=16*cm, height=8*cm)
                            self.elements.append(KeepTogether([img]))
                        else:
                            self.elements.append(Paragraph("[Plot Generation Failed]", self.styles['SwecoCell']))
                    self.elements.append(Spacer(1, 0.2*cm))
            
            self.elements.append(Spacer(1, 0.5*cm))

    def _identify_critical_steps(self, params, raw_data, sys_label, sys_nodes, step_key, veh_label, direction_label="Forward"):
        steps = raw_data.get(step_key, [])
        if not steps: return []

        output_groups = []
        num_spans = params['num_spans']
        
        for i in range(num_spans):
            eid = f"S{i+1}"
            
            max_M, idx_max_M = -1e15, -1
            min_M, idx_min_M = 1e15, -1
            max_V, idx_max_V = -1e15, -1
            min_V, idx_min_V = 1e15, -1
            
            found_data = False
            for idx, step in enumerate(steps):
                res = step['res']
                if eid in res:
                    found_data = True
                    m_arr = res[eid]['M']; v_arr = res[eid]['V']
                    mx_m = np.max(m_arr); mn_m = np.min(m_arr)
                    mx_v = np.max(v_arr); mn_v = np.min(v_arr)
                    
                    if mx_m > max_M: max_M = mx_m; idx_max_M = idx
                    if mn_m < min_M: min_M = mn_m; idx_min_M = idx
                    if mx_v > max_V: max_V = mx_v; idx_max_V = idx
                    if mn_v < min_V: min_V = mn_v; idx_min_V = idx
            
            if not found_data: continue
            
            group = {'header': f"Element {eid} ({direction_label} critical steps)", 'plots': []}
            
            critical_cases = [
                (idx_min_M, "Min M", 'M'),
                (idx_max_M, "Max M", 'M'),
                (idx_min_V, "Min V", 'V'),
                (idx_max_V, "Max V", 'V')
            ]
            
            processed = set() 
            
            for idx, label, type_code in critical_cases:
                if idx == -1: continue
                if (idx, type_code) in processed: continue
                processed.add((idx, type_code))
                
                step = steps[idx]
                x_loc = step['x']
                
                # UPDATED: Construct unified plot title with System and Vehicle Info
                s_short = "Sys A" if "A" in sys_label else "Sys B"
                v_short = "Veh A" if "A" in veh_label else "Veh B"
                title = (f"{s_short} - {v_short} - {direction_label} - Step {idx}: "
                         f"{label} | lead axle at x = {x_loc:.2f} m")
                
                is_A = (sys_label == "System A")
                
                config = {
                    'nodes': sys_nodes,
                    'sysA_data': step['res'] if is_A else {},
                    'sysB_data': step['res'] if not is_A else {},
                    'type_base': type_code,
                    'title': title,
                    'load_case_name': "Vehicle Steps",
                    'name_A': self.params_A['name'], 'name_B': self.params_B['name'],
                    'geom_A': self.raw_A.get('Selfweight'), 'geom_B': self.raw_B.get('Selfweight'),
                    'show_A': is_A, 'show_B': (not is_A),
                    'params_A': self.params_A, 'params_B': self.params_B,
                    'show_supports': True, 'font_scale': 1.5
                }
                
                group['plots'].append({'title': title, 'config': config})
            
            output_groups.append(group)
            
        return output_groups

    def _calculate_reaction_envelope(self, res_dict):
        reacts = {}
        target_data = res_dict.get('Total Envelope', {})
        if not target_data: return reacts
        
        for eid, dat in target_data.items():
            if 'ni_id' not in dat or 'nj_id' not in dat: continue
            def add_to_node(nid, fx_mx, fx_mn, fy_mx, fy_mn, mz_mx, mz_mn):
                if nid not in reacts: 
                    reacts[nid] = {'Rx_max':0.0, 'Rx_min':0.0, 'Ry_max':0.0, 'Ry_min':0.0, 'Mz_max':0.0, 'Mz_min':0.0}
                reacts[nid]['Rx_max'] += fx_mx; reacts[nid]['Rx_min'] += fx_mn
                reacts[nid]['Ry_max'] += fy_mx; reacts[nid]['Ry_min'] += fy_mn
                reacts[nid]['Mz_max'] += mz_mx; reacts[nid]['Mz_min'] += mz_mn

            c, s = dat['cx'], dat['cy']
            def get_val(key, idx):
                arr = dat.get(key)
                if arr is None: return 0.0
                if np.isscalar(arr): return arr
                if len(arr) == 0: return 0.0
                return arr[idx]

            n_mx = get_val('N_max', 0); n_mn = get_val('N_min', 0)
            v_mx = get_val('V_max', 0); v_mn = get_val('V_min', 0)
            m_mx = get_val('M_max', 0); m_mn = get_val('M_min', 0)
            
            def get_bounds(c_fac, s_fac, n_max, n_min, v_max, v_min):
                vals = []
                for n_v in [n_max, n_min]:
                    for v_v in [v_max, v_min]: vals.append(c_fac*n_v - s_fac*v_v)
                return max(vals), min(vals)
            
            fx_mx, fx_mn = get_bounds(c, s, n_mx, n_mn, v_mx, v_mn)
            fy_mx, fy_mn = get_bounds(s, -c, n_mx, n_mn, v_mx, v_mn) 
            add_to_node(dat['ni_id'], fx_mx, fx_mn, fy_mx, fy_mn, m_mx, m_mn)

            n_mx = get_val('N_max', -1); n_mn = get_val('N_min', -1)
            v_mx = get_val('V_max', -1); v_mn = get_val('V_min', -1)
            m_mx = get_val('M_max', -1); m_mn = get_val('M_min', -1)
            n_mx, n_mn = -n_mn, -n_mx; v_mx, v_mn = -v_mn, -v_mx; m_mx, m_mn = -m_mn, -m_mx
            
            fx_mx, fx_mn = get_bounds(c, s, n_mx, n_mn, v_mx, v_mn)
            fy_mx, fy_mn = get_bounds(s, -c, n_mx, n_mn, v_mx, v_mn)
            add_to_node(dat['nj_id'], fx_mx, fx_mn, fy_mx, fy_mn, m_mx, m_mn)
        return reacts

    def _add_force_summary_table(self, resA_dict, resB_dict):
        all_ids = sorted(list(set(resA_dict.keys()) | set(resB_dict.keys())), 
                         key=lambda x: (x[0], int(x[1:])))
        
        headers = [
            ["Elem", "M_Max", "M_Min", "V_Max", "V_Min", "N_Max", "N_Min", "Def_Max", "Def_Min"],
            ["[-]", "[kNm]", "[kNm]", "[kN]", "[kN]", "[kN]", "[kN]", "[mm]", "[mm]"]
        ]
        table_data = headers + [] 
        
        col_widths = [1.5*cm] + [2.0*cm]*8
        for eid in all_ids:
            row = [eid]
            dA = resA_dict.get(eid, {})
            dB = resB_dict.get(eid, {})
            def fmt_val(val_dict, k):
                if not val_dict: return 0.0
                arr = val_dict.get(k, [0.0])
                if np.isscalar(arr): return float(arr)
                if len(arr) == 0: return 0.0
                if 'min' in k: return np.min(arr)
                return np.max(arr)
            def cell_txt(k):
                vA = fmt_val(dA, k)
                
                is_def = "def" in k
                if is_def: vA *= 1000.0
                
                if self.valid_B:
                    vB = fmt_val(dB, k)
                    if is_def: vB *= 1000.0
                    return f"{vA:.1f} / {vB:.1f}"
                else:
                    return f"{vA:.1f}"
            
            row.extend([cell_txt('M_max'), cell_txt('M_min'), cell_txt('V_max'), cell_txt('V_min'), cell_txt('N_max'), cell_txt('N_min')])
            k_def_max = 'def_x_max' if eid.startswith('W') else 'def_y_max'
            k_def_min = 'def_x_min' if eid.startswith('W') else 'def_y_min'
            row.extend([cell_txt(k_def_max), cell_txt(k_def_min)])
            table_data.append(row)
        
        t = self._make_std_table(table_data, col_widths, font_size=7, header_rows=2)
        self.elements.append(KeepTogether([t]))
        if self.valid_B:
            self.elements.append(Paragraph("Values shown as: Sys A / Sys B", self.styles['Italic']))

    @staticmethod
    def _valid_support_nodes(params, raw_res):
        """Node ids to show in reaction tables.

        Prefers the solver-reported 'Restrained Nodes' (authoritative,
        includes Frame-mode supports placed at top nodes for zero-height
        walls). Falls back to the legacy id-range heuristic for results
        produced before the key existed.
        """
        restrained = (raw_res or {}).get('Restrained Nodes')
        if restrained:
            return set(restrained)
        params = params or {}
        mode = params.get('mode', 'Frame')
        num = params.get('num_spans', 1)
        base = 100 if mode == 'Frame' else 200
        return {base + i for i in range(num + 1)}

    def _add_reaction_table(self, resA_full, paramsA, resB_full, paramsB):
        if self.valid_B:
            self.elements.append(Paragraph("Support Reactions (Sys A / Sys B)", self.styles['SwecoTableHead']))
        else:
            self.elements.append(Paragraph("Support Reactions (System A)", self.styles['SwecoTableHead']))
            
        reactA = self._calculate_reaction_envelope(resA_full)
        reactB = {}
        if self.valid_B:
            reactB = self._calculate_reaction_envelope(resB_full)
        
        valid_A = self._valid_support_nodes(paramsA, self.raw_A)
        valid_B = set()
        if self.valid_B:
            valid_B = self._valid_support_nodes(paramsB, self.raw_B)
            
        all_nodes = sorted(list(set(reactA.keys()) | set(reactB.keys())))
        filtered_nodes = [n for n in all_nodes if (n in valid_A) or (n in valid_B)]
        
        headers = [
            ["Node", "Rx Max", "Rx Min", "Ry Max", "Ry Min", "Mz Max", "Mz Min"],
            ["[-]", "[kN]", "[kN]", "[kN]", "[kN]", "[kNm]", "[kNm]"]
        ]
        table_data = headers + []
        
        col_widths = [1.5*cm] + [2.6*cm]*6
        for nid in filtered_nodes:
            lbl = f"{nid}"
            if nid >= 200: lbl = f"Supp {nid-200+1}"
            elif nid >= 100: lbl = f"W{nid-100+1} Base"
            row = [lbl]
            dA = reactA.get(nid, {}); dB = reactB.get(nid, {})
            for comp in ['Rx', 'Ry', 'Mz']:
                for bound in ['max', 'min']:
                    k = f"{comp}_{bound}"
                    vA = dA.get(k, 0.0)
                    if self.valid_B:
                         vB = dB.get(k, 0.0)
                         row.append(f"{vA:.1f} / {vB:.1f}")
                    else:
                         row.append(f"{vA:.1f}")
            table_data.append(row)
        
        t = self._make_std_table(table_data, col_widths, font_size=7, header_rows=2)
        self.elements.append(KeepTogether([t]))

    def _make_std_table(self, data, col_widths, font_size=9, header_rows=1):
        # ReportLab does not wrap plain-string cells: any text wider than its
        # column runs over the table edge or into the neighbour cell. Wrap
        # cells that would overflow into Paragraphs (which do wrap); short
        # strings stay plain so the table's CENTER alignment applies.
        body_style = ParagraphStyle(
            f'_cell{font_size}', parent=self.styles['Normal'],
            fontSize=font_size, leading=font_size + 2)
        head_style = ParagraphStyle(
            f'_cellh{font_size}', parent=body_style, fontName='Helvetica-Bold')
        pad = 12.0  # default LEFTPADDING + RIGHTPADDING
        rows = []
        for r, row in enumerate(data):
            cells = []
            for ci, cell in enumerate(row):
                if isinstance(cell, str) and ci < len(col_widths) and col_widths[ci]:
                    is_head = r < header_rows
                    font = 'Helvetica-Bold' if is_head else 'Helvetica'
                    if stringWidth(cell, font, font_size) > col_widths[ci] - pad:
                        cell = Paragraph(cell, head_style if is_head else body_style)
                cells.append(cell)
            rows.append(cells)

        t = Table(rows, colWidths=col_widths, hAlign='LEFT')

        t.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), font_size),
            ('FONTNAME', (0,0), (-1, header_rows-1), 'Helvetica-Bold'),
            ('BACKGROUND', (0,0), (-1, header_rows-1), colors.lightgrey),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ]))
        return t
