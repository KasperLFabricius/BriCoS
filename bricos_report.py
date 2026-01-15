import io
import datetime
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT

import bricos_solver as solver
import bricos_viz as viz

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
        
        # Define Custom Styles
        self.styles.add(ParagraphStyle(name='SwecoHeader', parent=self.styles['Heading1'], fontSize=16, spaceAfter=12))
        self.styles.add(ParagraphStyle(name='SwecoSubHeader', parent=self.styles['Heading2'], fontSize=14, spaceAfter=10, textColor=colors.darkblue))
        self.styles.add(ParagraphStyle(name='SwecoTableHead', parent=self.styles['Normal'], fontSize=9, fontName='Helvetica-Bold'))
        self.styles.add(ParagraphStyle(name='SwecoBody', parent=self.styles['Normal'], fontSize=9, leading=11))
        
        # Run Solver fresh to ensure data consistency
        self.params_A = self.state['sysA']
        self.params_B = self.state['sysB']
        self.raw_A, self.nodes_A, _ = solver.run_raw_analysis(self.params_A)
        self.raw_B, self.nodes_B, _ = solver.run_raw_analysis(self.params_B)
        
    def generate(self):
        # 1. Cover / Metadata
        self._add_header_section()
        
        # 2. Input Summary
        self.elements.append(Paragraph("1. System Configuration & Geometry", self.styles['SwecoSubHeader']))
        self._add_system_input_summary("System A", self.params_A)
        self.elements.append(Spacer(1, 0.5*cm))
        self._add_system_input_summary("System B", self.params_B)
        self.elements.append(PageBreak())
        
        # 3. Total Envelope (ULS)
        self.elements.append(Paragraph("2. Design Results (ULS) - Total Envelope", self.styles['SwecoSubHeader']))
        self._add_results_section("Design (ULS)")
        self.elements.append(PageBreak())
        
        # 4. Total Envelope (SLS)
        self.elements.append(Paragraph("3. Characteristic Results (SLS) - Total Envelope", self.styles['SwecoSubHeader']))
        self._add_results_section("Characteristic (SLS)", tables_only=True)
        self.elements.append(PageBreak())
        
        # 5. Load Components (Selfweight, Soil, Surcharge)
        self.elements.append(Paragraph("4. Load Component Details", self.styles['SwecoSubHeader']))
        self._add_component_section("Selfweight")
        self._add_component_section("Soil")
        self._add_component_section("Surcharge")
        
        # 6. Vehicle Steps
        self.elements.append(Paragraph("5. Vehicle Analysis Steps (Sample)", self.styles['SwecoSubHeader']))
        self._add_vehicle_steps_sample()

        # Build PDF
        doc = SimpleDocTemplate(
            self.buffer, pagesize=A4,
            rightMargin=2*cm, leftMargin=2*cm,
            topMargin=2*cm, bottomMargin=2*cm
        )
        doc.build(self.elements)

    def _add_header_section(self):
        # Logo
        try:
            logo = Image("logo.png", width=4*cm, height=1.5*cm)
            logo.hAlign = 'RIGHT'
            self.elements.append(logo)
        except:
            self.elements.append(Paragraph("[Sweco Logo Missing]", self.styles['Normal']))
            
        self.elements.append(Spacer(1, 1*cm))
        self.elements.append(Paragraph("BriCoS Analysis Report", self.styles['Title']))
        self.elements.append(Spacer(1, 1*cm))
        
        # Metadata Table
        data = [
            ["Project No:", self.meta.get('proj_no', ''), "Date:", datetime.date.today().strftime("%Y-%m-%d")],
            ["Project Name:", self.meta.get('proj_name', ''), "Revision:", self.meta.get('rev', '')],
            ["Author:", self.meta.get('author', ''), "Checker:", self.meta.get('checker', '')],
            ["Approver:", self.meta.get('approver', ''), "Analysis Ver:", "v0.30"]
        ]
        
        t = Table(data, colWidths=[2.5*cm, 5.5*cm, 2.5*cm, 5.5*cm])
        t.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'), # Col 1 labels
            ('FONTNAME', (2,0), (2,-1), 'Helvetica-Bold'), # Col 3 labels
            ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.lightgrey),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        self.elements.append(t)
        self.elements.append(Spacer(1, 1*cm))
        
        # Comments
        if self.meta.get('comments', '').strip():
            self.elements.append(Paragraph("Comments:", self.styles['Heading3']))
            self.elements.append(Paragraph(self.meta['comments'], self.styles['SwecoBody']))
            self.elements.append(Spacer(1, 1*cm))

    def _add_system_input_summary(self, sys_label, p):
        self.elements.append(Paragraph(f"<b>{sys_label} ({p.get('name', '')})</b> - {p['mode']}", self.styles['Heading3']))
        
        # Config Grid
        txt = f"<b>KFI:</b> {p.get('KFI', 1.0)} | <b>Gamma_G:</b> {p.get('gamma_g', 1.0)} | <b>Gamma_Q:</b> {p.get('gamma_veh', 1.0)}"
        self.elements.append(Paragraph(txt, self.styles['SwecoBody']))
        self.elements.append(Spacer(1, 0.2*cm))
        
        # Spans Table
        span_data = [["Span", "Length [m]", "Inertia [m4]", "Load [kN/m]", "Material"]]
        for i in range(p['num_spans']):
            if p['L_list'][i] > 0.001:
                mat = f"C30 (fck={p['fck_span_list'][i]})" if p['e_mode']=='Eurocode' else f"E={p['E_custom_span'][i]} GPa"
                span_data.append([
                    f"S{i+1}", f"{p['L_list'][i]:.2f}", f"{p['Is_list'][i]:.4e}", 
                    f"{p['sw_list'][i]:.1f}", mat
                ])
        
        if len(span_data) > 1:
            self._add_std_table(span_data, [2*cm, 3*cm, 3*cm, 3*cm, 4*cm])
        else:
            self.elements.append(Paragraph("No spans defined.", self.styles['SwecoBody']))

        # Walls Table (If Frame)
        if p['mode'] == 'Frame':
            self.elements.append(Spacer(1, 0.2*cm))
            wall_data = [["Wall", "Height [m]", "Inertia [m4]", "Surch [kN/m]", "Material"]]
            has_wall = False
            for i in range(p['num_spans'] + 1):
                if p['h_list'][i] > 0.001:
                    has_wall = True
                    mat = f"C30 (fck={p['fck_wall_list'][i]})" if p['e_mode']=='Eurocode' else f"E={p['E_custom_wall'][i]} GPa"
                    sur = next((x['q'] for x in p.get('surcharge', []) if x['wall_idx']==i), 0.0)
                    wall_data.append([
                        f"W{i+1}", f"{p['h_list'][i]:.2f}", f"{p['Iw_list'][i]:.4e}", 
                        f"{sur:.1f}", mat
                    ])
            if has_wall: self._add_std_table(wall_data, [2*cm, 3*cm, 3*cm, 3*cm, 4*cm])

    def _add_results_section(self, res_mode, tables_only=False):
        # Calculate
        res_A = solver.combine_results(self.raw_A, self.params_A, res_mode)
        res_B = solver.combine_results(self.raw_B, self.params_B, res_mode)
        
        # Plots
        if not tables_only:
            self._add_plot_set(res_A, res_B, "Total Envelope", 
                             geom_A=res_A['Selfweight'], geom_B=res_B['Selfweight'])
        
        # Summary Table
        self.elements.append(Spacer(1, 0.5*cm))
        self.elements.append(Paragraph(f"Tabular Summary ({res_mode})", self.styles['Heading3']))
        
        # We aggregate all elements present in A or B
        all_ids = sorted(list(set(res_A['Total Envelope'].keys()) | set(res_B['Total Envelope'].keys())), 
                         key=lambda x: (x[0], int(x[1:])))
        
        table_data = [["Elem", "M_max", "M_min", "V_max", "V_min", "N_max", "N_min", "Def_max"]]
        
        for eid in all_ids:
            row = [eid]
            dA = res_A['Total Envelope'].get(eid, {})
            
            # Helper to format cell: "A / B"
            def fmt(key_max, key_min=None):
                vA = np.max(dA.get(key_max, [0])) if dA else 0
                vB = np.max(res_B['Total Envelope'].get(eid, {}).get(key_max, [0])) if eid in res_B['Total Envelope'] else 0
                
                if key_min: # Finding absolute min
                    vA_n = np.min(dA.get(key_min, [0])) if dA else 0
                    vB_n = np.min(res_B['Total Envelope'].get(eid, {}).get(key_min, [0])) if eid in res_B['Total Envelope'] else 0
                    return f"{vA:.1f} / {vB:.1f}\n{vA_n:.1f} / {vB_n:.1f}"
                else:
                    return f"{vA:.1f} / {vB:.1f}"

            # M
            row.append(fmt('M_max', 'M_min'))
            # V
            row.append(fmt('V_max', 'V_min'))
            # N
            row.append(fmt('N_max', 'N_min'))
            # Def
            k_def = 'def_x_max' if eid.startswith('W') else 'def_y_max'
            k_def_min = 'def_x_min' if eid.startswith('W') else 'def_y_min'
            
            # Scale def by 1000 for mm
            def fmt_def(k_mx, k_mn):
                vA = np.max(dA.get(k_mx, [0]))*1000 if dA else 0
                vB = np.max(res_B['Total Envelope'].get(eid, {}).get(k_mx, [0]))*1000 if eid in res_B['Total Envelope'] else 0
                vA_n = np.min(dA.get(k_mn, [0]))*1000 if dA else 0
                vB_n = np.min(res_B['Total Envelope'].get(eid, {}).get(k_mn, [0]))*1000 if eid in res_B['Total Envelope'] else 0
                return f"{vA:.1f} / {vB:.1f}\n{vA_n:.1f} / {vB_n:.1f}"
            
            row.append(fmt_def(k_def, k_def_min))
            table_data.append(row)
            
        self._add_std_table(table_data, [1.5*cm] + [2.6*cm]*7, font_size=8)
        self.elements.append(Paragraph("Values shown as: Sys A / Sys B", self.styles['Italic']))

    def _add_component_section(self, load_key):
        # Check if active
        res_A = solver.combine_results(self.raw_A, self.params_A, "Design (ULS)")
        res_B = solver.combine_results(self.raw_B, self.params_B, "Design (ULS)")
        
        has_A = bool(res_A.get(load_key))
        has_B = bool(res_B.get(load_key))
        
        if not has_A and not has_B: return
        
        self.elements.append(Paragraph(f"Load Case: {load_key}", self.styles['Heading3']))
        self._add_plot_set(res_A, res_B, load_key, geom_A=res_A['Selfweight'], geom_B=res_B['Selfweight'])
        self.elements.append(Spacer(1, 0.5*cm))

    def _add_vehicle_steps_sample(self):
        # We'll pick 3 steps: Start, Middle, End
        # We need to access the raw step lists
        steps_A = self.raw_A.get('Vehicle Steps A', [])
        steps_B = self.raw_B.get('Vehicle Steps B', [])
        
        # Just grab from System A for simplicity if available, else B
        steps_src = steps_A if steps_A else steps_B
        sys_src_name = "System A" if steps_A else "System B"
        
        if not steps_src:
            self.elements.append(Paragraph("No vehicle steps analysis found.", self.styles['SwecoBody']))
            return

        indices = [0, len(steps_src)//2, len(steps_src)-1]
        indices = sorted(list(set(indices))) # Unique
        
        for idx in indices:
            if idx >= len(steps_src): continue
            step = steps_src[idx]
            x_loc = step['x']
            
            self.elements.append(Paragraph(f"Step {idx+1}: Vehicle at X = {x_loc:.2f} m", self.styles['Heading4']))
            
            # Create a fig just for this step
            # We construct a dummy result dict that looks like a full result but only has this step
            # Note: bricos_viz expects a structure like {'ElementID': {'M': ..., 'x': ...}}
            # step['res'] has exactly that.
            
            # We only plot M and V for brevity
            try:
                # We reuse the plot function but pass Single Step Data as if it was the main data
                # We set load_case_name to "Vehicle Steps" to trigger arrow drawing in Viz
                fig_M = viz.create_plotly_fig(
                    self.nodes_A if steps_A else self.nodes_B, 
                    step['res'] if steps_A else {}, 
                    steps_B[idx]['res'] if (steps_B and idx < len(steps_B)) else {},
                    type_base='M', 
                    title=f"Bending Moment [kNm] - Step {idx}",
                    load_case_name="Vehicle Steps",
                    name_A=self.params_A['name'], name_B=self.params_B['name'],
                    geom_A=self.raw_A.get('Selfweight'), geom_B=self.raw_B.get('Selfweight'),
                    show_A=bool(steps_A), show_B=bool(steps_B),
                    params_A=self.params_A, params_B=self.params_B,
                    show_supports=True
                )
                self._add_fig_image(fig_M)
            except Exception as e:
                self.elements.append(Paragraph(f"Could not render plot: {str(e)}", self.styles['Italic']))
            
            self.elements.append(Spacer(1, 0.5*cm))

    def _add_plot_set(self, res_A, res_B, case_name, geom_A, geom_B):
        # M, V, N, Def
        types = [('M', 'Bending Moment [kNm]'), ('V', 'Shear Force [kN]'), 
                 ('N', 'Normal Force [kN]'), ('Def', 'Deformation [mm]')]
        
        # We create a 2x2 grid of images
        img_flowables = []
        
        for t_code, t_title in types:
            try:
                fig = viz.create_plotly_fig(
                    self.nodes_A, 
                    res_A.get(case_name, {}), 
                    res_B.get(case_name, {}), 
                    type_base=t_code, 
                    title=t_title,
                    load_case_name=case_name,
                    name_A=self.params_A['name'], name_B=self.params_B['name'],
                    geom_A=geom_A, geom_B=geom_B,
                    params_A=self.params_A, params_B=self.params_B,
                    show_supports=True
                )
                img_bytes = self._fig_to_bytes(fig)
                if img_bytes:
                    img = Image(img_bytes, width=8*cm, height=5*cm)
                    img_flowables.append(img)
            except:
                pass

        if img_flowables:
            # Arrange in Table for Grid Layout
            # Rows of 2
            rows = []
            for i in range(0, len(img_flowables), 2):
                rows.append(img_flowables[i:i+2])
            
            t = Table(rows, colWidths=[8.5*cm, 8.5*cm])
            t.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('ALIGN', (0,0), (-1,-1), 'CENTER')]))
            self.elements.append(t)

    def _fig_to_bytes(self, fig):
        try:
            b = io.BytesIO()
            # Requires kaleido
            fig.write_image(b, format='png', scale=2)
            b.seek(0)
            return b
        except:
            return None

    def _add_std_table(self, data, col_widths, font_size=9):
        t = Table(data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), font_size),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ]))
        self.elements.append(t)

    def _add_fig_image(self, fig):
        b = self._fig_to_bytes(fig)
        if b:
            img = Image(b, width=16*cm, height=8*cm)
            self.elements.append(img)