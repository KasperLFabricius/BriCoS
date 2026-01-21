import io
import datetime
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfgen import canvas

# Graphics Imports for Vehicle Diagram
from reportlab.graphics.shapes import Drawing, Line, String, Polygon, Group
from reportlab.graphics import renderPDF

# Internal Modules
import bricos_solver as solver
import bricos_viz as viz
import bricos_data as data_mod

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
    def __init__(self, buffer, meta_data, session_state, raw_res_A, raw_res_B, nodes_A, nodes_B, version="0.31", progress_callback=None):
        self.buffer = buffer
        self.meta = meta_data
        self.state = session_state
        self.version = version
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

    def _update_progress(self, val):
        if self.progress_callback:
            self.progress_callback(val)

    def _match_vehicle_class(self, current_loads, current_spacing):
        """
        Checks if the current load/spacing configuration matches a standard vehicle 
        defined in vehicles.csv. Returns the name if found, else 'Custom'.
        """
        try:
            csv_path = data_mod.resource_path("vehicles.csv")
            # Fail silently if CSV is missing
            try:
                df = pd.read_csv(csv_path)
            except FileNotFoundError:
                return "Custom"
            
            # Convert current config to arrays for comparison
            curr_L = np.array(current_loads, dtype=float)
            curr_S = np.array(current_spacing, dtype=float)
            
            for index, row in df.iterrows():
                try:
                    # Parse CSV strings "1.0, 2.0" -> numpy array
                    std_L = np.fromstring(row['Loads'], sep=',')
                    std_S = np.fromstring(row['Spacing'], sep=',')
                    
                    # 1. Compare Lengths
                    if len(std_L) != len(curr_L) or len(std_S) != len(curr_S):
                        continue
                        
                    # 2. Compare Values with tolerance
                    if np.allclose(std_L, curr_L, atol=1e-3) and np.allclose(std_S, curr_S, atol=1e-3):
                        return row['Name']
                except Exception:
                    continue
                    
            return "Custom"
        except Exception:
            return "Custom"

    def generate(self):
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
        
        # 5. Total Envelope (ULS)
        self.elements.append(Paragraph(f"{self.chapter_count}. Design Results (ULS) - Total Envelope", self.styles['SwecoSubHeader']))
        eq_txt = self._build_uls_equation_text()
        self.elements.append(Paragraph(f"<b>Formula:</b> {eq_txt}", self.styles['SwecoSmall']))
        self.elements.append(Spacer(1, 0.2*cm))
        self._add_results_section("Design (ULS)", prog_range=(0.15, 0.35)) 
        self.elements.append(PageBreak())
        self.chapter_count += 1
        
        # 6. Total Envelope (SLS)
        self.elements.append(Paragraph(f"{self.chapter_count}. Characteristic Results (SLS) - Total Envelope", self.styles['SwecoSubHeader']))
        self.elements.append(Paragraph(r"<b>Formula:</b> 1.0 · Permanent + 1.0 · Variable (No KFI, No Gamma, No Phi)", self.styles['SwecoSmall']))
        self.elements.append(Spacer(1, 0.2*cm))
        self._add_results_section("Characteristic (No Dynamic Factor)", prog_range=(0.35, 0.50))
        self.elements.append(PageBreak())
        self.chapter_count += 1
        
        # 7. Load Components (Unfactored)
        # Check active components across BOTH systems if valid_B, else just A
        has_sw = any(v > 0 for v in self.params_A['sw_list'])
        has_soil = len(self.params_A.get('soil', [])) > 0
        has_surch = len(self.params_A.get('surcharge', [])) > 0
        
        if self.valid_B:
            if any(v > 0 for v in self.params_B['sw_list']): has_sw = True
            if len(self.params_B.get('soil', [])) > 0: has_soil = True
            if len(self.params_B.get('surcharge', [])) > 0: has_surch = True
        
        active_comps = sum([has_sw, has_soil, has_surch])
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
        
        # 8. Critical Vehicle Steps
        self.elements.append(Paragraph(f"{self.chapter_count}. Critical Vehicle Steps (Unfactored)", self.styles['SwecoSubHeader']))
        
        self.elements.append(Paragraph("<b>Table 8.1: Critical Vehicle Effects (Raw Envelope Values)</b>", self.styles['SwecoSmall']))
        self.elements.append(Paragraph("Values represent the raw Min/Max envelope before application of Partial Factors (Gamma, KFI) and Dynamic Factor (Phi).", self.styles['SwecoCell']))
        self._add_unfactored_vehicle_table()
        self.elements.append(Spacer(1, 0.4*cm))

        self.elements.append(Paragraph("<b>Vehicle Step Plots</b>", self.styles['SwecoBody']))
        self.elements.append(Paragraph("Vehicle positions causing peak effects per span.", self.styles['SwecoSmall']))
        self.elements.append(Spacer(1, 0.2*cm))
        self._add_smart_vehicle_steps(prog_range=(0.75, 0.95))

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
            "The structure is discretized using beam elements with two available formulations: "
            "<b>Euler-Bernoulli</b> (slender members, neglecting shear deformation) or <b>Timoshenko</b> "
            "(deep members, accounting for shear deformation via the parameter <i>&Phi;<sub>s</sub></i> = 12<i>EI</i> / (<i>GA<sub>s</sub>L</i><sup>2</sup>)). "
            "Non-prismatic (tapered) elements are handled via numerical integration of the flexibility matrix. "
            "Material behavior is assumed Linear Elastic.")

        # Condensed 1.3
        add_sub("1.2 Boundary Conditions", 
            "Supports are modeled using the Penalty Method with high-stiffness springs for Fixed/Pinned supports (<i>k</i> &approx; 10<sup>14</sup>) "
            "and discrete springs for elastic foundations based on user-specified stiffness (<i>K<sub>x</sub>, K<sub>y</sub>, K<sub>&theta;</sub></i>).")

        # Condensed 1.4
        add_sub("1.3 Moving Load Analysis",
            "Traffic actions are evaluated using a Quasi-Static algorithm, stepping the vehicle model across the structure to compute absolute maximum and minimum envelopes for forces and displacements. "
            "The Dynamic Amplification Factor (<i>&Phi;</i>) is calculated automatically based on the influence length (compliant with <b>DS/EN 1991-2 DK NA</b>) or defined manually.")

        # Condensed 1.5
        add_sub("1.4 Load Combinations", 
            "Design values (<i>E<sub>d</sub></i>) are computed by superposition of factored envelopes: "
            "<i>E<sub>d</sub></i> = <i>K<sub>FI</sub></i> &middot; (<i>&gamma;<sub>G</sub>E<sub>SW</sub></i> + <i>&gamma;<sub>Soil</sub>E<sub>Soil</sub></i> + <i>&gamma;<sub>Q</sub>&Phi;E<sub>Veh</sub></i> + <i>&gamma;<sub>Q</sub>E<sub>Surch</sub></i>). "
            "Partial factors (<i>&gamma;</i>) and Consequence Class factor (<i>K<sub>FI</sub></i>) are applied as defined in settings. "
            "Traffic surcharge interaction is applied according to user selection (Exclusive or Simultaneous with vehicle load).")

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
            phi_val = p.get('phi', 1.0)
            if p.get('phi_mode') == 'Calculate' and raw: phi_val = raw.get('phi_calc', 1.0)
            g_veh = p.get('gamma_veh', 1.0)
            g_vehB = p.get('gamma_vehB', 1.0)
            has_A = bool(p.get('vehicle', {}).get('loads'))
            has_B = bool(p.get('vehicleB', {}).get('loads'))
            
            perm = f"{kfi}·{gg}·SW"
            if p.get('soil'): perm += f" + {kfi}·{gj}·Soil"
            
            var = ""
            if has_A: var += f" + {kfi}·{g_veh}·{phi_val:.2f}·VehA"
            if has_B: var += f" + {kfi}·{g_vehB}·{phi_val:.2f}·VehB"
            if p.get('surcharge'): var += f" + {kfi}·{g_veh}·Surch"
            return perm + var
            
        eqA = get_eq(self.params_A, self.raw_A)
        if not self.valid_B:
            return eqA
        
        eqB = get_eq(self.params_B, self.raw_B)
        if eqA == eqB: return f"Design = {eqA}"
        return f"SysA: {eqA} <br/> SysB: {eqB}"

    def _add_header_section(self):
        try:
            logo = Image("logo.png", width=4*cm, height=1.5*cm)
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
            ["Mesh Size", f"{p.get('mesh_size', 0.5)} m", "Finite Element discretization length"],
            ["Step Size", f"{p.get('step_size', 0.2)} m", "Vehicle moving load increment"],
            ["Vehicle Direction", f"{v_dir}", "Traffic flow direction"]
        ]
        
        use_shear = p.get('use_shear_def', False)
        shear_status = "Enabled (Timoshenko)" if use_shear else "Disabled (Euler-Bernoulli)"
        data.append(["Shear Deformation", shear_status, "Stiffness matrix formulation"])
        
        # Use Paragraph to render HTML tags (subscript)
        lbl_beff = Paragraph("Effective Width (<i>b<sub>eff</sub></i>)", self.styles['SwecoBody'])
        data.append([lbl_beff, f"{p.get('b_eff', 1.0)} m", "Used for shear area & axial area estimation"])

        if use_shear:
            data.append(["Poisson's Ratio (ν)", f"{p.get('nu', 0.2)}", "Used for shear modulus G"])
            
        # --- NEW: Surcharge Interaction Mode ---
        surch_mode = p.get('surch_mode', 'Exclusive')
        surch_txt = "Exclusive from Traffic" if surch_mode == 'Exclusive' else "Simultaneous with Traffic"
        data.append(["Surcharge Interaction", surch_txt, "Vehicle load & surcharge combination"])

        t = self._make_std_table(data, [4*cm, 5.5*cm, 6.5*cm])
        self.elements.append(KeepTogether([t]))

    def _draw_vehicle_stick_model(self, loads, spacing, width=400, height=80):
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
            
        y_axle_line = height * 0.4
        arrow_len = height * 0.25
        
        d.add(Line(margin_x - 10, y_axle_line, width - margin_x + 10, y_axle_line, strokeColor=colors.black, strokeWidth=1))
        
        for i, load_val in enumerate(loads):
            x_pos = offset_x + (cum_dist[i] if total_len > 0.1 else 0) * scale_x
            p = Polygon(points=[x_pos, y_axle_line, x_pos-3, y_axle_line+6, x_pos+3, y_axle_line+6], fillColor=colors.red, strokeWidth=0)
            d.add(p)
            d.add(Line(x_pos, y_axle_line, x_pos, y_axle_line + arrow_len, strokeColor=colors.red, strokeWidth=2))
            label = f"{load_val}t"
            d.add(String(x_pos, y_axle_line + arrow_len + 5, label, textAnchor='middle', fontSize=8, fillColor=colors.red))
            d.add(Group(Polygon(points=[x_pos-2, y_axle_line-2, x_pos+2, y_axle_line-2, x_pos+2, y_axle_line+2, x_pos-2, y_axle_line+2], 
                                fillColor=colors.black, strokeWidth=0)))

        dim_y = y_axle_line - 15
        for i in range(len(spacing)):
            if i == 0: continue
            dist = spacing[i]
            x_prev = offset_x + cum_dist[i-1] * scale_x
            x_curr = offset_x + cum_dist[i] * scale_x
            d.add(Line(x_prev, dim_y, x_curr, dim_y, strokeColor=colors.blue, strokeWidth=0.5))
            d.add(Line(x_prev, dim_y-2, x_prev, dim_y+2, strokeColor=colors.blue, strokeWidth=0.5))
            d.add(Line(x_curr, dim_y-2, x_curr, dim_y+2, strokeColor=colors.blue, strokeWidth=0.5))
            mid_x = (x_prev + x_curr) / 2
            d.add(String(mid_x, dim_y - 8, f"{dist}m", textAnchor='middle', fontSize=7, fillColor=colors.blue))
            
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

    def _add_system_input_summary(self, sys_label, p, raw_res, props, sys_key_id):
        self.elements.append(Paragraph(f"<b>{sys_label} ({p.get('name', '')})</b> - {p['mode']}", self.styles['Heading3']))
        
        phi_val = p.get('phi', 1.0)
        phi_txt = f"{phi_val:.3f} (Manual)"
        if p.get('phi_mode') == 'Calculate' and raw_res:
            calc_val = raw_res.get('phi_calc', 1.0)
            phi_txt = f"{calc_val:.3f} (Calc)"

        gamma_str = f"G={p.get('gamma_g', 1.0)} | Soil={p.get('gamma_j', 1.0)}"
        has_A = bool(p.get('vehicle', {}).get('loads'))
        has_B = bool(p.get('vehicleB', {}).get('loads'))
        
        if has_A and not has_B: gamma_str += f" | Q_A={p.get('gamma_veh', 1.0)}"
        elif has_B and not has_A: gamma_str += f" | Q_B={p.get('gamma_vehB', 1.0)}"
        elif has_A and has_B: gamma_str += f" | Q_A={p.get('gamma_veh', 1.0)} | Q_B={p.get('gamma_vehB', 1.0)}"
        
        txt_settings = (f"<b>KFI:</b> {p.get('KFI', 1.0)} | <b>Gammas:</b> {gamma_str} | <b>Phi:</b> {phi_txt}")
        self.elements.append(Paragraph(txt_settings, self.styles['SwecoBody']))
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
            self.elements.append(Paragraph("Boundary Conditions:", self.styles['SwecoSmall']))
            t = self._make_std_table(supp_data, [4*cm, 4*cm, 7*cm])
            self.elements.append(KeepTogether([t]))

        # 4. SOIL
        soil_list = p.get('soil', [])
        if soil_list:
            self.elements.append(Spacer(1, 0.2*cm))
            self.elements.append(Paragraph("Soil Loads (Earth Pressure):", self.styles['SwecoSmall']))
            soil_table = [["Wall", "Face", "Height [m]", "q_top [kN/m]", "q_bot [kN/m]"]]
            for s in soil_list:
                soil_table.append([
                    f"W{s['wall_idx']+1}", s['face'], f"{s['h']:.2f}", f"{s['q_top']:.1f}", f"{s['q_bot']:.1f}"
                ])
            t = self._make_std_table(soil_table, [2*cm, 2*cm, 3*cm, 3*cm, 3*cm])
            self.elements.append(KeepTogether([t]))

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
                drawing = self._draw_vehicle_stick_model(v_loads, v_spac, width=400, height=60)
                self.elements.append(drawing)
                self.elements.append(Spacer(1, 0.3*cm))
        
        add_veh_table('vehicle', "A", "vehA")
        add_veh_table('vehicleB', "B", "vehB")

        # 6. PHI CALCULATION LOG
        if p.get('phi_mode', 'Calculate') == 'Calculate' and raw_res and raw_res.get('phi_log'):
            self.elements.append(Spacer(1, 0.2*cm))
            self.elements.append(Paragraph("Dynamic Factor Calculation (<i>Φ</i>):", self.styles['SwecoSmall']))
            
            log_lines = raw_res['phi_log']
            formatted_lines = []
            for line in log_lines:
                # Use Unicode directly instead of entities to avoid & display issues
                txt = line.replace("L_phi", "<i>L<sub>Φ</sub></i>")\
                          .replace("L_mean", "<i>L<sub>mean</sub></i>")\
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
        try:
            fig = viz.create_plotly_fig(**fig_kwargs)
            b = io.BytesIO()
            fig.write_image(b, format='png', scale=1.5)
            b.seek(0)
            if not b.getvalue().startswith(b'\x89PNG'):
                return None
            return b
        except Exception:
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
        res_A = solver.combine_results(self.raw_A, self.params_A, res_mode)
        res_B = {}
        if self.valid_B:
            res_B = solver.combine_results(self.raw_B, self.params_B, res_mode)
        
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
                'show_A': True, 'show_B': False, 'show_supports': True, 'font_scale': 1.5
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
                    'show_A': False, 'show_B': True, 'show_supports': True, 'font_scale': 1.5
                })

        images = self._submit_parallel_plots(tasks, prog_range)
        
        self.elements.append(Paragraph("System A", self.styles['SwecoSmall']))
        self._append_image_grid(images[0:4]) 
        
        if self.valid_B:
            self.elements.append(Spacer(1, 0.3*cm))
            self.elements.append(Paragraph("System B", self.styles['SwecoSmall']))
            self._append_image_grid(images[4:8]) 

        self.elements.append(Spacer(1, 0.5*cm))
        
        title_str = f"Tabular Summary - {res_mode.upper()}"
        self.elements.append(Paragraph(title_str, self.styles['Heading3']))
        
        self._add_force_summary_table(res_A['Total Envelope'], res_B.get('Total Envelope', {}))
        self.elements.append(Spacer(1, 0.3*cm))
        self._add_reaction_table(res_A, self.params_A, res_B, self.params_B)

    def _add_component_section(self, load_key, prog_range=(0.0, 0.0)):
        res_A = solver.combine_results(self.raw_A, self.params_A, "Characteristic (No Dynamic Factor)")
        res_B = {}
        if self.valid_B:
             res_B = solver.combine_results(self.raw_B, self.params_B, "Characteristic (No Dynamic Factor)")
        
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
                'show_A': True, 'show_B': False, 'show_supports': True, 'font_scale': 1.5
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
                    'show_A': False, 'show_B': True, 'show_supports': True, 'font_scale': 1.5
                })

        images = self._submit_parallel_plots(tasks, prog_range)
        
        self.elements.append(Paragraph("System A", self.styles['SwecoSmall']))
        self._append_image_grid(images[0:4])
        
        if self.valid_B:
            self.elements.append(Spacer(1, 0.3*cm))
            self.elements.append(Paragraph("System B", self.styles['SwecoSmall']))
            self._append_image_grid(images[4:8])
        
        self.elements.append(Spacer(1, 0.5*cm))
        self._add_force_summary_table(res_A.get(load_key, {}), res_B.get(load_key, {}))
        self.elements.append(Spacer(1, 0.3*cm))
        
        wrap_A = {'Total Envelope': res_A.get(load_key, {})}
        wrap_B = {'Total Envelope': res_B.get(load_key, {})} if self.valid_B else {}
        self._add_reaction_table(wrap_A, self.params_A, wrap_B, self.params_B)

    def _append_image_grid(self, img_bytes_list):
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
            self.elements.append(KeepTogether([t]))

    def _add_unfactored_vehicle_table(self):
        """Adds table showing raw vehicle effects (Unfactored) for validation."""
        data = [["Elem", "M_max", "M_min", "V_max", "V_min", "System"]]
        
        def process_sys(raw_env, sys_name):
            if not raw_env: return
            all_ids = sorted(raw_env.keys(), key=lambda x: (x[0], int(x[1:])))
            for eid in all_ids:
                dat = raw_env[eid]
                def get_v(k):
                    arr = dat.get(k)
                    if arr is None or len(arr) == 0: return 0.0
                    return np.max(arr) if 'max' in k else np.min(arr)
                
                mmx = get_v('M_max'); mmn = get_v('M_min')
                vmx = get_v('V_max'); vmn = get_v('V_min')
                data.append([eid, f"{mmx:.1f}", f"{mmn:.1f}", f"{vmx:.1f}", f"{vmn:.1f}", sys_name])

        # Helper to safely check if a vehicle is defined (has loads) in the params
        def has_veh(p, key):
            # Safe access: params -> vehicle_key -> loads list
            return bool(p.get(key, {}).get('loads'))

        # Only process systems if the vehicle is actually defined in the input parameters
        if has_veh(self.params_A, 'vehicle'):
            process_sys(self.raw_A.get('Vehicle Envelope A', {}), "A (Veh A)")
        
        if has_veh(self.params_A, 'vehicleB'):
            process_sys(self.raw_A.get('Vehicle Envelope B', {}), "A (Veh B)")

        if self.valid_B:
            if has_veh(self.params_B, 'vehicle'):
                process_sys(self.raw_B.get('Vehicle Envelope A', {}), "B (Veh A)")
                
            if has_veh(self.params_B, 'vehicleB'):
                process_sys(self.raw_B.get('Vehicle Envelope B', {}), "B (Veh B)")
        
        if len(data) > 1:
            t = self._make_std_table(data, [2*cm, 3*cm, 3*cm, 3*cm, 3*cm, 2*cm])
            self.elements.append(KeepTogether([t]))
        else:
            self.elements.append(Paragraph("No vehicle results found.", self.styles['SwecoSmall']))

    def _add_smart_vehicle_steps(self, prog_range=(0.0, 0.0)):
        # Define 4 specific combos: SysA-VehA, SysA-VehB, SysB-VehA, SysB-VehB
        combos = [
            (self.params_A, self.raw_A, "System A", self.nodes_A, 'Vehicle Steps A', "Vehicle A"),
            (self.params_A, self.raw_A, "System A", self.nodes_A, 'Vehicle Steps B', "Vehicle B")
        ]
        
        if self.valid_B:
            combos.extend([
                (self.params_B, self.raw_B, "System B", self.nodes_B, 'Vehicle Steps A', "Vehicle A"),
                (self.params_B, self.raw_B, "System B", self.nodes_B, 'Vehicle Steps B', "Vehicle B")
            ])
        
        all_task_groups = []
        
        for p, r, s_lbl, n, step_key, veh_lbl in combos:
            # Only process if results exist for this vehicle
            if r.get(step_key):
                g = self._identify_critical_steps(p, r, s_lbl, n, step_key, veh_lbl)
                if g:
                    all_task_groups.append({
                        'main_header': f"{s_lbl} - {veh_lbl}", 
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

    def _identify_critical_steps(self, params, raw_data, sys_label, sys_nodes, step_key, veh_label):
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
            
            group = {'header': f"Element {eid} (Critical Steps)", 'plots': []}
            
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
                title = f"{s_short} - {v_short} - Step {idx}: {label} @ X={x_loc:.2f}m"
                
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

    def _calculate_reaction_envelope(self, res_dict, nodes_dict):
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

    def _add_reaction_table(self, resA_full, paramsA, resB_full, paramsB):
        if self.valid_B:
            self.elements.append(Paragraph("Support Reactions (Sys A / Sys B)", self.styles['SwecoTableHead']))
        else:
            self.elements.append(Paragraph("Support Reactions (System A)", self.styles['SwecoTableHead']))
            
        reactA = self._calculate_reaction_envelope(resA_full, self.nodes_A)
        reactB = {}
        if self.valid_B:
            reactB = self._calculate_reaction_envelope(resB_full, self.nodes_B)
        
        def get_valid_supports(p):
            mode = p.get('mode', 'Frame'); num = p.get('num_spans', 1)
            valid_ids = []
            base = 100 if mode == 'Frame' else 200
            for i in range(num + 1): valid_ids.append(base + i)
            return set(valid_ids)
            
        valid_A = get_valid_supports(paramsA)
        valid_B = set()
        if self.valid_B:
            valid_B = get_valid_supports(paramsB)
            
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
        t = Table(data, colWidths=col_widths, hAlign='LEFT')
        
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