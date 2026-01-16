import io
import datetime
import numpy as np
import pandas as pd
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

import bricos_solver as solver
import bricos_viz as viz

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
        self.drawRightString(200*mm, 10*mm, # A4 is ~210mm wide. 200mm is right margin.
            "Page %d of %d" % (self._pageNumber, page_count))

# ==========================================
# REPORT GENERATION MODULE
# ==========================================

class BricosReportGenerator:
    def __init__(self, buffer, meta_data, session_state):
        self.buffer = buffer
        self.meta = meta_data
        self.state = session_state
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

        # Run Solver fresh to ensure data consistency
        self.params_A = self.state['sysA']
        self.params_B = self.state['sysB']
        self.raw_A, self.nodes_A, _ = solver.run_raw_analysis(self.params_A)
        self.raw_B, self.nodes_B, _ = solver.run_raw_analysis(self.params_B)
        
    def generate(self):
        # 1. Cover / Metadata
        self._add_header_section()
        self.elements.append(PageBreak())
        
        # 2. Background Theory (New Chapter 1)
        self.elements.append(Paragraph(f"{self.chapter_count}. Basis of Analysis & Methodology", self.styles['SwecoSubHeader']))
        self._add_theory_section()
        self.elements.append(PageBreak())
        self.chapter_count += 1
        
        # 3. Global Analysis Settings
        self.elements.append(Paragraph(f"{self.chapter_count}. Global Analysis Settings", self.styles['SwecoSubHeader']))
        self._add_global_settings_summary()
        self.elements.append(PageBreak())
        self.chapter_count += 1
        
        # 4. Input Summary
        self.elements.append(Paragraph(f"{self.chapter_count}. System Configuration & Geometry", self.styles['SwecoSubHeader']))
        self._add_system_input_summary("System A", self.params_A, self.raw_A)
        self.elements.append(PageBreak())
        self._add_system_input_summary("System B", self.params_B, self.raw_B)
        self.elements.append(PageBreak())
        self.chapter_count += 1
        
        # 5. Total Envelope (ULS)
        self.elements.append(Paragraph(f"{self.chapter_count}. Design Results (ULS) - Total Envelope", self.styles['SwecoSubHeader']))
        eq_txt = self._build_uls_equation_text()
        self.elements.append(Paragraph(f"<b>Formula:</b> {eq_txt}", self.styles['SwecoSmall']))
        self.elements.append(Spacer(1, 0.2*cm))
        self._add_results_section("Design (ULS)") 
        self.elements.append(PageBreak())
        self.chapter_count += 1
        
        # 6. Total Envelope (SLS)
        self.elements.append(Paragraph(f"{self.chapter_count}. Characteristic Results (SLS) - Total Envelope", self.styles['SwecoSubHeader']))
        self.elements.append(Paragraph(r"<b>Formula:</b> 1.0 · Permanent + 1.0 · Variable (No KFI, No Gamma, No Phi)", self.styles['SwecoSmall']))
        self.elements.append(Spacer(1, 0.2*cm))
        self._add_results_section("Characteristic (No Dynamic Factor)")
        self.elements.append(PageBreak())
        self.chapter_count += 1
        
        # 7. Load Components (Unfactored)
        has_sw = any(v > 0 for v in self.params_A['sw_list']) or any(v > 0 for v in self.params_B['sw_list'])
        has_soil = len(self.params_A.get('soil', [])) > 0 or len(self.params_B.get('soil', [])) > 0
        has_surch = len(self.params_A.get('surcharge', [])) > 0 or len(self.params_B.get('surcharge', [])) > 0
        
        if has_sw:
            self.elements.append(Paragraph(f"{self.chapter_count}. Load Case: Selfweight (Unfactored)", self.styles['SwecoSubHeader']))
            self._add_component_section("Selfweight")
            self.elements.append(PageBreak())
            self.chapter_count += 1
            
        if has_soil:
            self.elements.append(Paragraph(f"{self.chapter_count}. Load Case: Soil (Unfactored)", self.styles['SwecoSubHeader']))
            self._add_component_section("Soil")
            self.elements.append(PageBreak())
            self.chapter_count += 1
            
        if has_surch:
            self.elements.append(Paragraph(f"{self.chapter_count}. Load Case: Surcharge (Unfactored)", self.styles['SwecoSubHeader']))
            self._add_component_section("Surcharge")
            self.elements.append(PageBreak())
            self.chapter_count += 1
        
        # 8. Critical Vehicle Steps
        self.elements.append(Paragraph(f"{self.chapter_count}. Critical Vehicle Steps (Unfactored)", self.styles['SwecoSubHeader']))
        self.elements.append(Paragraph("Vehicle positions causing peak effects per span, ordered by: Min M, Max M, Min V, Max V.", self.styles['SwecoSmall']))
        self.elements.append(Spacer(1, 0.2*cm))
        self._add_smart_vehicle_steps()

        # Build PDF using Custom Canvas
        doc = SimpleDocTemplate(
            self.buffer, pagesize=A4,
            rightMargin=1.5*cm, leftMargin=1.5*cm,
            topMargin=2*cm, bottomMargin=2*cm
        )
        doc.build(self.elements, canvasmaker=NumberedCanvas)

    def _add_theory_section(self):
        """Adds standard background theory text."""
        styles = self.styles
        
        # Helper for subsections
        def add_sub(title, text):
            self.elements.append(Paragraph(f"<b>{title}</b>", styles['SwecoBody']))
            self.elements.append(Paragraph(text, styles['SwecoBody']))
            self.elements.append(Spacer(1, 0.2*cm))

        # 1. Calculation Method
        add_sub("1.1 Calculation Method", 
            "The structural analysis is performed using <b>BriCoS v0.30</b>, a proprietary Finite Element Analysis (FEM) tool based on the Matrix Stiffness Method for 2D planar frames.")

        # 2. Element Formulation
        add_sub("1.2 Element Formulation", 
            "The structure is discretized using advanced beam elements that account for both constant and variable cross-sections. Two formulations are available:")
        
        bullets = [
            "<b>Euler-Bernoulli Theory:</b> Classical beam theory assuming plane sections remain plane and perpendicular to the neutral axis. Shear deformations are neglected (<i>&Phi;<sub>s</sub></i> = 0). Suitable for slender members.",
            "<b>Timoshenko Theory:</b> Explicitly accounts for shear deformation, ensuring accuracy for deep members (e.g., piers, thick slabs). The stiffness matrix is modified by the shear parameter:"
        ]
        
        # Math Equation for Phi_s
        eq_phi = "<i>&Phi;<sub>s</sub></i> = 12<i>EI</i> / (<i>GA<sub>s</sub>L</i><sup>2</sup>)"
        
        bullets2 = [
            "Where <i>A<sub>s</sub></i> is the effective shear area. Poisson's ratio (<i>&nu;</i>) is user-defined (default 0.2).",
            "<b>Variable Stiffness (Non-Prismatic):</b> For tapered or haunched elements, the stiffness matrix is computed via numerical integration of the flexibility matrix, ensuring accurate distribution of forces in varying geometries.",
            "<b>Material Behavior:</b> The analysis assumes Linear Elastic material behavior. Geometric non-linearity (P-Delta) is not currently included."
        ]

        for b in bullets:
            self.elements.append(Paragraph(f"• {b}", styles['SwecoBody']))
        
        # Insert Equation nicely
        self.elements.append(Paragraph(eq_phi, styles['SwecoMath']))

        for b in bullets2:
            self.elements.append(Paragraph(f"• {b}", styles['SwecoBody']))
            
        self.elements.append(Spacer(1, 0.2*cm))

        # 3. Boundary Conditions
        add_sub("1.3 Boundary Conditions", "Supports are modeled using the Penalty Method.")
        bullets_bc = [
            "<b>Fixed/Pinned Supports:</b> Represented by high-stiffness springs (<i>k</i> &approx; 10<sup>14</sup> kN/m or kNm/rad).",
            "<b>Elastic Foundations:</b> Modeled as discrete springs based on user-specified stiffness (<i>K<sub>x</sub>, K<sub>y</sub>, K<sub>&theta;</sub></i>)."
        ]
        for b in bullets_bc:
            self.elements.append(Paragraph(f"• {b}", styles['SwecoBody']))
        self.elements.append(Spacer(1, 0.2*cm))

        # 4. Moving Load Analysis
        add_sub("1.4 Moving Load Analysis", "Traffic actions are evaluated using a Quasi-Static Moving Load algorithm.")
        bullets_mv = [
            "<b>Stepping:</b> The vehicle model is stepped across the structure in user-defined increments.",
            "<b>Envelopes:</b> The software computes the absolute maximum and minimum envelopes for Internal Forces (<i>M, V, N</i>), Displacements, and Reactions.",
            "<b>Dynamic Factor:</b> The Dynamic Amplification Factor (<i>&Phi;</i>) is calculated automatically based on the influence length (compliant with <b>DS/EN 1991-2 DK NA</b>), or defined manually."
        ]
        for b in bullets_mv:
            self.elements.append(Paragraph(f"• {b}", styles['SwecoBody']))
        self.elements.append(Spacer(1, 0.2*cm))

        # 5. Load Combinations
        add_sub("1.5 Load Combinations", 
            "Results are combined using the Principle of Superposition. The final design values (<i>E<sub>d</sub></i>) are computed by summing the factored envelopes.")
        
        # Proper Math Equation
        eq_comb = "<i>E<sub>d</sub></i> = <i>&gamma;<sub>G</sub>E<sub>SW</sub></i> + <i>&gamma;<sub>Soil</sub>E<sub>Soil</sub></i> + <i>&gamma;<sub>Q</sub>&Phi;E<sub>Veh</sub></i> + <i>&gamma;<sub>Q</sub>E<sub>Surch</sub></i>"
        self.elements.append(Paragraph(eq_comb, styles['SwecoMath']))
        
        bullets_lc = [
            "<b>Surcharge Interaction:</b> Traffic surcharge is combined with the main vehicle load based on the user selection (either <i>Exclusive</i> or <i>Simultaneous</i>).",
            "<b>Partial Factors:</b> Factors for self-weight (<i>&gamma;<sub>G</sub></i>), soil/earth pressure (<i>&gamma;<sub>Soil</sub></i>), variable loads (<i>&gamma;<sub>Q</sub></i>), and Consequence Class (<i>K<sub>FI</sub></i>) are applied as defined in the project settings."
        ]
        for b in bullets_lc:
            self.elements.append(Paragraph(f"• {b}", styles['SwecoBody']))

    def _build_uls_equation_text(self):
        def get_eq(p, raw):
            kfi = p.get('KFI', 1.0)
            gg = p.get('gamma_g', 1.0)
            gj = p.get('gamma_j', 1.0)
            
            phi_val = p.get('phi', 1.0)
            if p.get('phi_mode') == 'Calculate' and raw:
                phi_val = raw.get('phi_calc', 1.0)
            
            g_veh = p.get('gamma_veh', 1.0)
            g_vehB = p.get('gamma_vehB', 1.0)
            
            has_A = bool(p.get('vehicle', {}).get('loads'))
            has_B = bool(p.get('vehicleB', {}).get('loads'))
            
            # Use proper symbols for output formula too
            perm = f"{kfi}·{gg}·SW"
            if p.get('soil'): perm += f" + {kfi}·{gj}·Soil"
            
            var = ""
            if has_A: var += f" + {kfi}·{g_veh}·{phi_val:.2f}·VehA"
            if has_B: var += f" + {kfi}·{g_vehB}·{phi_val:.2f}·VehB"
            if p.get('surcharge'): var += f" + {kfi}·{g_veh}·Surch"
            
            return perm + var
            
        eqA = get_eq(self.params_A, self.raw_A)
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
        self.elements.append(Paragraph("BriCoS Analysis Report", self.styles['Title']))
        self.elements.append(Spacer(1, 1*cm))
        
        data = [
            ["Project No:", self.meta.get('proj_no', ''), "Date:", datetime.date.today().strftime("%Y-%m-%d")],
            ["Project Name:", self.meta.get('proj_name', ''), "Revision:", self.meta.get('rev', '')],
            ["Author:", self.meta.get('author', ''), "Checker:", self.meta.get('checker', '')],
            ["Approver:", self.meta.get('approver', ''), "Analysis Ver:", "v0.30"]
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
        data = [
            ["Parameter", "Value", "Description"],
            ["Mesh Size", f"{p.get('mesh_size', 0.5)} m", "Finite Element discretization length"],
            ["Step Size", f"{p.get('step_size', 0.2)} m", "Vehicle moving load increment"],
            ["Vehicle Direction", f"{p.get('vehicle_direction', 'Forward')}", "Traffic flow direction"]
        ]
        
        use_shear = p.get('use_shear_def', False)
        shear_status = "Enabled (Timoshenko)" if use_shear else "Disabled (Euler-Bernoulli)"
        data.append(["Shear Deformation", shear_status, "stiffness matrix formulation"])
        
        # Always report Effective Width (needed for Axial Area estimation)
        data.append(["Effective Width (b_eff)", f"{p.get('b_eff', 1.0)} m", "Used for Shear Area & Axial Area estimation"])

        # Only report Nu if shear is active
        if use_shear:
            data.append(["Poisson's Ratio (nu)", f"{p.get('nu', 0.2)}", "Used for Shear Modulus G"])
        
        t = self._make_std_table(data, [4*cm, 5*cm, 7*cm])
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
        
        if total_len < 0.1:
            scale_x = 0
            offset_x = width / 2
        else:
            scale_x = draw_w / total_len
            offset_x = margin_x
            
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
        """
        Helper to robustly describe section geometry (ISSUE F FIX).
        Handles: I vs H, Shapes (Const/Lin/3Pt), and Inclination.
        """
        geom_key = f"{prefix}_geom_{idx}"
        
        # Default Fallback
        val_default = p[simple_list_key][idx]
        desc = f"I = {val_default:.4e}"
        
        if geom_key in p:
            g = p[geom_key]
            
            # 1. Determine Type (Inertia or Height)
            is_height = (g.get('type', 0) == 1)
            lbl = "H" if is_height else "I"
            fmt = "{:.3f}m" if is_height else "{:.4e}"
            
            # 2. Determine Shape and Values
            shape = g.get('shape', 0)
            vals = g.get('vals', [0.0, 0.0, 0.0])
            
            # Ensure vals are floats
            v1 = fmt.format(vals[0])
            v2 = fmt.format(vals[1])
            v3 = fmt.format(vals[2])
            
            if shape == 0: # Constant
                desc = f"{lbl} = {v1}"
            elif shape == 1: # Linear (Start -> End)
                # Note: bricos_solver logic uses vals[0] (Start) and vals[2] (End) for linear
                desc = f"{lbl}: {v1} -> {v3} (Taper)"
            elif shape == 2: # 3-Point
                desc = f"{lbl}: {v1} -> {v2} -> {v3} (3-Pt)"
                
            # 3. Span Inclination
            if prefix == 'span' and g.get('align_type') == 1:
                mode = g.get('incline_mode', 0)
                val = g.get('incline_val', 0.0)
                inc_txt = f"{val:.2f}%" if mode == 0 else f"{val:.2f}m"
                desc += f"\nSlope: {inc_txt}"
                
        return desc

    def _add_system_input_summary(self, sys_label, p, raw_res):
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
        
        # DYNAMIC COLUMN WIDTH LOGIC FOR GEOMETRY
        # Fixed Widths: ID (1.5) + Dim (2.0) + Load (2.0) + Mat (3.5) = 9.0 cm
        # Page: A4 (21cm) - Margins (1.5+1.5) = 18.0 cm
        # Available for Geom = 18.0 - 9.0 = 9.0 cm (Increased from 5.0)
        
        w_id, w_dim, w_load, w_mat = 1.5*cm, 2.0*cm, 2.0*cm, 3.5*cm
        page_width = 18.0*cm
        w_geom = page_width - (w_id + w_dim + w_load + w_mat)
        
        col_widths = [w_id, w_dim, w_geom, w_load, w_mat]

        # 1. SPANS
        span_data = [["Span", "L [m]", "Section Geometry", "SW [kN/m]", "Material"]]
        for i in range(p['num_spans']):
            if p['L_list'][i] > 0.001:
                if p['e_mode'] == 'Eurocode':
                    mat_str = f"fck = {p['fck_span_list'][i]} MPa"
                else:
                    mat_str = f"E = {p['E_custom_span'][i]} GPa"

                geom_desc = self._get_geometry_description(p, 'span', i, 'Is_list')

                span_data.append([
                    f"S{i+1}", f"{p['L_list'][i]:.2f}", geom_desc, 
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
            wall_data = [["Wall", "H [m]", "Section Geometry", "Surch [kN/m]", "Material"]]
            has_wall = False
            for i in range(p['num_spans'] + 1):
                if p['h_list'][i] > 0.001:
                    has_wall = True
                    if p['e_mode'] == 'Eurocode':
                        mat_str = f"fck = {p['fck_wall_list'][i]} MPa"
                    else:
                        mat_str = f"E = {p['E_custom_wall'][i]} GPa"
                    
                    sur = next((x['q'] for x in p.get('surcharge', []) if x['wall_idx']==i), 0.0)
                    geom_desc = self._get_geometry_description(p, 'wall', i, 'Iw_list')

                    wall_data.append([
                        f"W{i+1}", f"{p['h_list'][i]:.2f}", geom_desc, 
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
            soil_table = [["Wall", "Face", "Height [m]", "q_top [kN/m2]", "q_bot [kN/m2]"]]
            for s in soil_list:
                soil_table.append([
                    f"W{s['wall_idx']+1}", s['face'], f"{s['h']:.2f}", f"{s['q_top']:.1f}", f"{s['q_bot']:.1f}"
                ])
            t = self._make_std_table(soil_table, [2*cm, 2*cm, 3*cm, 3*cm, 3*cm])
            self.elements.append(KeepTogether([t]))

        # 5. VEHICLES
        self.elements.append(Spacer(1, 0.2*cm))
        def add_veh_table(key, title_suffix):
            veh = p.get(key, {})
            v_loads = veh.get('loads', [])
            v_spac = veh.get('spacing', [])
            if v_loads:
                self.elements.append(Paragraph(f"Vehicle {title_suffix}:", self.styles['SwecoSmall']))
                drawing = self._draw_vehicle_stick_model(v_loads, v_spac, width=400, height=60)
                self.elements.append(drawing)
                self.elements.append(Spacer(1, 0.3*cm))

        add_veh_table('vehicle', "A")
        add_veh_table('vehicleB', "B")

    def _add_results_section(self, res_mode):
        res_A = solver.combine_results(self.raw_A, self.params_A, res_mode)
        res_B = solver.combine_results(self.raw_B, self.params_B, res_mode)
        
        # Split Plots: System A then System B
        self.elements.append(Paragraph(f"Visualizations - {res_mode}", self.styles['Heading4']))
        
        self.elements.append(Paragraph("System A", self.styles['SwecoSmall']))
        self._add_plot_set_single_sys(res_A, "Total Envelope", "A")
        self.elements.append(Spacer(1, 0.3*cm))
        
        self.elements.append(Paragraph("System B", self.styles['SwecoSmall']))
        self._add_plot_set_single_sys(res_B, "Total Envelope", "B")
        self.elements.append(Spacer(1, 0.5*cm))
        
        title_str = f"Tabular Summary - {res_mode.upper()}"
        self.elements.append(Paragraph(title_str, self.styles['Heading3']))
        
        self._add_force_summary_table(res_A['Total Envelope'], res_B['Total Envelope'])
        self.elements.append(Spacer(1, 0.3*cm))
        
        # Reaction Table
        self._add_reaction_table(res_A, self.params_A, res_B, self.params_B)

    def _add_component_section(self, load_key):
        res_A = solver.combine_results(self.raw_A, self.params_A, "Characteristic (No Dynamic Factor)")
        res_B = solver.combine_results(self.raw_B, self.params_B, "Characteristic (No Dynamic Factor)")
        
        # Split Plots
        self.elements.append(Paragraph("System A", self.styles['SwecoSmall']))
        self._add_plot_set_single_sys(res_A, load_key, "A")
        self.elements.append(Spacer(1, 0.3*cm))
        
        self.elements.append(Paragraph("System B", self.styles['SwecoSmall']))
        self._add_plot_set_single_sys(res_B, load_key, "B")
        self.elements.append(Spacer(1, 0.5*cm))
        
        # Force Summary
        self._add_force_summary_table(res_A.get(load_key, {}), res_B.get(load_key, {}))
        self.elements.append(Spacer(1, 0.3*cm))
        
        # Reaction Table
        wrap_A = {'Total Envelope': res_A.get(load_key, {})} # Hack to use same calculator
        wrap_B = {'Total Envelope': res_B.get(load_key, {})}
        self._add_reaction_table(wrap_A, self.params_A, wrap_B, self.params_B)

    def _calculate_reaction_envelope(self, res_dict, nodes_dict):
        """
        Re-calculates reactions from element end forces in the envelope.
        Matches the logic in bricos_main.py to ensure Min/Max separation.
        """
        reacts = {}
        target_data = res_dict.get('Total Envelope', {})
        if not target_data: return reacts
        
        for eid, dat in target_data.items():
            if 'ni_id' not in dat or 'nj_id' not in dat: continue
            
            def add_to_node(nid, fx_mx, fx_mn, fy_mx, fy_mn, mz_mx, mz_mn):
                if nid not in reacts: 
                    reacts[nid] = {
                        'Rx_max': 0.0, 'Rx_min': 0.0, 
                        'Ry_max': 0.0, 'Ry_min': 0.0, 
                        'Mz_max': 0.0, 'Mz_min': 0.0
                    }
                reacts[nid]['Rx_max'] += fx_mx
                reacts[nid]['Rx_min'] += fx_mn
                reacts[nid]['Ry_max'] += fy_mx
                reacts[nid]['Ry_min'] += fy_mn
                reacts[nid]['Mz_max'] += mz_mx
                reacts[nid]['Mz_min'] += mz_mn

            c, s = dat['cx'], dat['cy']
            
            # Helper to safely get value from data which might be float or array
            def get_val(key, idx):
                arr = dat.get(key)
                if arr is None: return 0.0
                if np.isscalar(arr): return arr
                if len(arr) == 0: return 0.0
                return arr[idx]

            # Start Node
            n_mx = get_val('N_max', 0); n_mn = get_val('N_min', 0)
            v_mx = get_val('V_max', 0); v_mn = get_val('V_min', 0)
            m_mx = get_val('M_max', 0); m_mn = get_val('M_min', 0)
            
            def get_bounds(c_fac, s_fac, n_max, n_min, v_max, v_min):
                vals = []
                for n_v in [n_max, n_min]:
                    for v_v in [v_max, v_min]:
                        vals.append(c_fac*n_v - s_fac*v_v)
                return max(vals), min(vals)
            
            fx_mx, fx_mn = get_bounds(c, s, n_mx, n_mn, v_mx, v_mn)
            fy_mx, fy_mn = get_bounds(s, -c, n_mx, n_mn, v_mx, v_mn) 
            
            add_to_node(dat['ni_id'], fx_mx, fx_mn, fy_mx, fy_mn, m_mx, m_mn)

            # End Node
            n_mx = get_val('N_max', -1); n_mn = get_val('N_min', -1)
            v_mx = get_val('V_max', -1); v_mn = get_val('V_min', -1)
            m_mx = get_val('M_max', -1); m_mn = get_val('M_min', -1)
            
            # Inverse forces for end node
            n_mx, n_mn = -n_mn, -n_mx 
            v_mx, v_mn = -v_mn, -v_mx
            m_mx, m_mn = -m_mn, -m_mx
            
            fx_mx, fx_mn = get_bounds(c, s, n_mx, n_mn, v_mx, v_mn)
            fy_mx, fy_mn = get_bounds(s, -c, n_mx, n_mn, v_mx, v_mn)
            
            add_to_node(dat['nj_id'], fx_mx, fx_mn, fy_mx, fy_mn, m_mx, m_mn)
                 
        return reacts

    def _add_force_summary_table(self, resA_dict, resB_dict):
        all_ids = sorted(list(set(resA_dict.keys()) | set(resB_dict.keys())), 
                         key=lambda x: (x[0], int(x[1:])))
        
        table_data = [["Elem", "M_Max", "M_Min", "V_Max", "V_Min", "N_Max", "N_Min", "Def_Max", "Def_Min"]]
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
                vB = fmt_val(dB, k)
                if "def" in k: vA *= 1000.0; vB *= 1000.0
                return f"{vA:.1f} / {vB:.1f}"

            row.append(cell_txt('M_max'))
            row.append(cell_txt('M_min'))
            row.append(cell_txt('V_max'))
            row.append(cell_txt('V_min'))
            row.append(cell_txt('N_max'))
            row.append(cell_txt('N_min'))
            k_def_max = 'def_x_max' if eid.startswith('W') else 'def_y_max'
            k_def_min = 'def_x_min' if eid.startswith('W') else 'def_y_min'
            row.append(cell_txt(k_def_max))
            row.append(cell_txt(k_def_min))
            table_data.append(row)
            
        t = self._make_std_table(table_data, col_widths, font_size=7)
        self.elements.append(KeepTogether([t]))
        self.elements.append(Paragraph("Values shown as: Sys A / Sys B", self.styles['Italic']))

    def _add_reaction_table(self, resA_full, paramsA, resB_full, paramsB):
        self.elements.append(Paragraph("Support Reactions (Sys A / Sys B)", self.styles['SwecoTableHead']))
        
        # 1. Calculate Envelopes explicitly to get Min != Max
        reactA = self._calculate_reaction_envelope(resA_full, self.nodes_A)
        reactB = self._calculate_reaction_envelope(resB_full, self.nodes_B)
        
        # 2. Identify Valid Support Nodes (Filtering "Phantom Supports")
        def get_valid_supports(p):
            mode = p.get('mode', 'Frame')
            num = p.get('num_spans', 1)
            valid_ids = []
            if mode == 'Frame':
                # Supports are at Base (100, 101...)
                for i in range(num + 1): valid_ids.append(100 + i)
            else:
                # Supports are at Deck Level (200, 201...)
                for i in range(num + 1): valid_ids.append(200 + i)
            return set(valid_ids)
            
        valid_A = get_valid_supports(paramsA)
        valid_B = get_valid_supports(paramsB)
        
        # Only include nodes that are valid supports in EITHER system
        all_nodes = sorted(list(set(reactA.keys()) | set(reactB.keys())))
        filtered_nodes = [n for n in all_nodes if (n in valid_A) or (n in valid_B)]
        
        table_data = [["Node", "Rx Max", "Rx Min", "Ry Max", "Ry Min", "Mz Max", "Mz Min"]]
        col_widths = [1.5*cm] + [2.6*cm]*6
        
        for nid in filtered_nodes:
            # Format Node Label
            lbl = f"{nid}"
            if nid >= 200: lbl = f"Supp {nid-200+1}"
            elif nid >= 100: lbl = f"W{nid-100+1} Base"

            row = [lbl]
            
            dA = reactA.get(nid, {})
            dB = reactB.get(nid, {})
            
            # We assume dA/dB are dicts with keys 'Rx_max', 'Rx_min' etc from _calculate_reaction_envelope
            
            for comp in ['Rx', 'Ry', 'Mz']:
                for bound in ['max', 'min']:
                    k = f"{comp}_{bound}"
                    vA = dA.get(k, 0.0)
                    vB = dB.get(k, 0.0)
                    row.append(f"{vA:.1f} / {vB:.1f}")
            table_data.append(row)
            
        t = self._make_std_table(table_data, col_widths, font_size=7)
        self.elements.append(KeepTogether([t]))

    def _add_plot_set_single_sys(self, res, case_name, sys_key):
        # Generates plots for ONE system to allow A/B splitting
        types = [('M', 'Bending Moment [kNm]'), ('V', 'Shear Force [kN]'), 
                 ('N', 'Normal Force [kN]'), ('Def', 'Deformation [mm]')]
        img_flowables = []
        
        # Prepare params based on sys_key
        if sys_key == "A":
            p = self.params_A
            nm = self.params_A['name']
            geom = self.raw_A.get('Selfweight')
            rA, rB = res.get(case_name, {}), {} # Only pass A
            nodes = self.nodes_A
            showA, showB = True, False
        else:
            p = self.params_B
            nm = self.params_B['name']
            geom = self.raw_B.get('Selfweight')
            rA, rB = {}, res.get(case_name, {}) # Only pass B
            nodes = self.nodes_B
            showA, showB = False, True
            
        for t_code, t_title in types:
            try:
                fig = viz.create_plotly_fig(
                    nodes, rA, rB, 
                    type_base=t_code, title=f"{t_title} - {nm}", load_case_name=case_name,
                    name_A=self.params_A['name'], name_B=self.params_B['name'],
                    geom_A=geom if sys_key=="A" else None, 
                    geom_B=geom if sys_key=="B" else None,
                    params_A=self.params_A, params_B=self.params_B,
                    show_A=showA, show_B=showB,
                    show_supports=True,
                    font_scale=1.5
                )
                img_bytes = self._fig_to_bytes(fig, scale=3.0) 
                if img_bytes:
                    img = Image(img_bytes, width=8*cm, height=5*cm)
                    img_flowables.append(img)
            except: pass

        if img_flowables:
            rows = []
            for i in range(0, len(img_flowables), 2):
                rows.append(img_flowables[i:i+2])
            t = Table(rows, colWidths=[8.5*cm, 8.5*cm], hAlign='LEFT')
            t.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('ALIGN', (0,0), (-1,-1), 'CENTER')]))
            self.elements.append(KeepTogether([t]))

    def _add_smart_vehicle_steps(self):
        self._process_system_span_by_span(self.params_A, self.raw_A, "System A", self.nodes_A)
        self.elements.append(Spacer(1, 0.5*cm))
        self._process_system_span_by_span(self.params_B, self.raw_B, "System B", self.nodes_B)

    def _process_system_span_by_span(self, params, raw_data, sys_label, sys_nodes):
        steps = raw_data.get('Vehicle Steps A', [])
        if not steps: return

        self.elements.append(Paragraph(f"<b>{sys_label} ({params.get('name')})</b>", self.styles['Heading4']))
        
        num_spans = params['num_spans']
        
        for i in range(num_spans):
            eid = f"S{i+1}"
            
            # 1. FIND CRITICAL INDICES
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
            
            self.elements.append(Paragraph(f"<b>Element {eid}</b>", self.styles['SwecoBody']))
            
            tasks = [
                (idx_min_M, 'M', f"Min Moment ({eid})"),
                (idx_max_M, 'M', f"Max Moment ({eid})"),
                (idx_min_V, 'V', f"Min Shear ({eid})"),
                (idx_max_V, 'V', f"Max Shear ({eid})")
            ]
            
            for idx, type_code, title in tasks:
                if idx == -1: continue
                
                step = steps[idx]
                x_loc = step['x']
                
                full_title = f"{title} @ X={x_loc:.2f}m"
                self.elements.append(Paragraph(f"Step {idx}: {full_title}", self.styles['SwecoCell']))
                
                try:
                    is_A = (sys_label == "System A")
                    
                    fig = viz.create_plotly_fig(
                        sys_nodes, 
                        step['res'] if is_A else {}, # A Data
                        step['res'] if not is_A else {}, # B Data
                        type_base=type_code, 
                        title=full_title,
                        load_case_name="Vehicle Steps",
                        name_A=self.params_A['name'], name_B=self.params_B['name'],
                        geom_A=self.raw_A.get('Selfweight'), geom_B=self.raw_B.get('Selfweight'),
                        show_A=is_A, show_B=(not is_A),
                        params_A=self.params_A, params_B=self.params_B,
                        show_supports=True,
                        font_scale=1.5
                    )
                    self._add_fig_image(fig)
                except Exception as e:
                    pass
                self.elements.append(Spacer(1, 0.2*cm))

    def _fig_to_bytes(self, fig, scale=2.0):
        try:
            b = io.BytesIO()
            fig.write_image(b, format='png', scale=scale)
            b.seek(0)
            return b
        except: return None

    def _make_std_table(self, data, col_widths, font_size=9):
        t = Table(data, colWidths=col_widths, hAlign='LEFT')
        t.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), font_size),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ]))
        return t

    def _add_fig_image(self, fig):
        b = self._fig_to_bytes(fig, scale=3.0)
        if b:
            img = Image(b, width=16*cm, height=8*cm)
            self.elements.append(KeepTogether([img]))