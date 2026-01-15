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
        self.styles.add(ParagraphStyle(name='SwecoSmall', parent=self.styles['Normal'], fontSize=8, leading=10))
        self.styles.add(ParagraphStyle(name='SwecoCell', parent=self.styles['Normal'], fontSize=8, leading=9))
        
        # Run Solver fresh to ensure data consistency
        self.params_A = self.state['sysA']
        self.params_B = self.state['sysB']
        self.raw_A, self.nodes_A, _ = solver.run_raw_analysis(self.params_A)
        self.raw_B, self.nodes_B, _ = solver.run_raw_analysis(self.params_B)
        
    def generate(self):
        # 1. Cover / Metadata
        self._add_header_section()
        
        # Page Break after comments/header
        self.elements.append(PageBreak())
        
        # 2. Input Summary
        self.elements.append(Paragraph("1. System Configuration & Geometry", self.styles['SwecoSubHeader']))
        self._add_system_input_summary("System A", self.params_A, self.raw_A)
        
        # Forced Page Break after System A summary as requested
        self.elements.append(PageBreak())
        
        self._add_system_input_summary("System B", self.params_B, self.raw_B)
        self.elements.append(PageBreak())
        
        # 3. Total Envelope (ULS)
        self.elements.append(Paragraph("2. Design Results (ULS) - Total Envelope", self.styles['SwecoSubHeader']))
        eq_txt = self._build_uls_equation_text()
        self.elements.append(Paragraph(f"<b>Formula:</b> {eq_txt}", self.styles['SwecoSmall']))
        self.elements.append(Spacer(1, 0.2*cm))
        self._add_results_section("Design (ULS)")
        self.elements.append(PageBreak())
        
        # 4. Total Envelope (SLS)
        self.elements.append(Paragraph("3. Characteristic Results (SLS) - Total Envelope", self.styles['SwecoSubHeader']))
        self.elements.append(Paragraph(r"<b>Formula:</b> 1.0 · Permanent + 1.0 · Variable (No KFI, No Gamma, No Phi)", self.styles['SwecoSmall']))
        self.elements.append(Spacer(1, 0.2*cm))
        self._add_results_section("Characteristic (No Dynamic Factor)", tables_only=True)
        self.elements.append(PageBreak())
        
        # 5. Load Components (Unfactored)
        self.elements.append(Paragraph("4. Load Component Details (Unfactored)", self.styles['SwecoSubHeader']))
        self.elements.append(Paragraph("Visualization of individual load cases without partial coefficients or dynamic factors.", self.styles['SwecoSmall']))
        self.elements.append(Spacer(1, 0.3*cm))
        
        self._add_component_section("Selfweight")
        self._add_component_section("Soil")
        self._add_component_section("Surcharge")
        
        # 6. Critical Vehicle Steps
        self.elements.append(PageBreak())
        self.elements.append(Paragraph("5. Critical Vehicle Steps (Unfactored)", self.styles['SwecoSubHeader']))
        self.elements.append(Paragraph("Vehicle positions causing peak effects in each span element, separated by system.", self.styles['SwecoSmall']))
        self.elements.append(Spacer(1, 0.2*cm))
        self._add_smart_vehicle_steps()

        # Build PDF
        doc = SimpleDocTemplate(
            self.buffer, pagesize=A4,
            rightMargin=1.5*cm, leftMargin=1.5*cm,
            topMargin=2*cm, bottomMargin=2*cm
        )
        doc.build(self.elements)

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
        
        txt_settings = (
            f"<b>KFI:</b> {p.get('KFI', 1.0)} | <b>Gammas:</b> {gamma_str} | <b>Phi:</b> {phi_txt}<br/>"
            f"<b>Mesh:</b> {p.get('mesh_size', 0.5)}m | <b>Step:</b> {p.get('step_size', 0.2)}m | <b>Direction:</b> {p.get('vehicle_direction', 'Forward')}"
        )
        self.elements.append(Paragraph(txt_settings, self.styles['SwecoBody']))
        self.elements.append(Spacer(1, 0.2*cm))
        
        # 1. SPANS
        span_data = [["Span", "L [m]", "I [m4]", "SW [kN/m]", "Material"]]
        for i in range(p['num_spans']):
            if p['L_list'][i] > 0.001:
                if p['e_mode'] == 'Eurocode':
                    mat_str = f"fck = {p['fck_span_list'][i]} MPa"
                else:
                    mat_str = f"E = {p['E_custom_span'][i]} GPa"

                span_data.append([
                    f"S{i+1}", f"{p['L_list'][i]:.2f}", f"{p['Is_list'][i]:.4e}", 
                    f"{p['sw_list'][i]:.1f}", mat_str
                ])
        
        if len(span_data) > 1:
            t = self._make_std_table(span_data, [1.5*cm, 2.0*cm, 2.5*cm, 2.0*cm, 3.5*cm])
            self.elements.append(KeepTogether([t]))
        else:
            self.elements.append(Paragraph("No spans defined.", self.styles['SwecoBody']))

        # 2. WALLS
        if p['mode'] == 'Frame':
            self.elements.append(Spacer(1, 0.2*cm))
            wall_data = [["Wall", "H [m]", "I [m4]", "Surch [kN/m]", "Material"]]
            has_wall = False
            for i in range(p['num_spans'] + 1):
                if p['h_list'][i] > 0.001:
                    has_wall = True
                    if p['e_mode'] == 'Eurocode':
                        mat_str = f"fck = {p['fck_wall_list'][i]} MPa"
                    else:
                        mat_str = f"E = {p['E_custom_wall'][i]} GPa"
                    
                    sur = next((x['q'] for x in p.get('surcharge', []) if x['wall_idx']==i), 0.0)
                    wall_data.append([
                        f"W{i+1}", f"{p['h_list'][i]:.2f}", f"{p['Iw_list'][i]:.4e}", 
                        f"{sur:.1f}", mat_str
                    ])
            if has_wall: 
                t = self._make_std_table(wall_data, [1.5*cm, 2.0*cm, 2.5*cm, 2.0*cm, 3.5*cm])
                self.elements.append(KeepTogether([t]))

        # 3. SUPPORTS
        self.elements.append(Spacer(1, 0.2*cm))
        supp_data = [["Support Node", "Type", "Stiffness (Kx, Ky, Km)"]]
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
                p_loads = Paragraph(str(v_loads), self.styles['SwecoCell'])
                p_spac = Paragraph(str(v_spac), self.styles['SwecoCell'])
                v_data = [["Parameter", "Values"], ["Axle Loads [t]", p_loads], ["Spacing [m]", p_spac]]
                t = self._make_std_table(v_data, [3*cm, 12*cm])
                self.elements.append(KeepTogether([t]))
                self.elements.append(Spacer(1, 0.1*cm))

        add_veh_table('vehicle', "A")
        add_veh_table('vehicleB', "B")

    def _add_results_section(self, res_mode, tables_only=False):
        res_A = solver.combine_results(self.raw_A, self.params_A, res_mode)
        res_B = solver.combine_results(self.raw_B, self.params_B, res_mode)
        
        if not tables_only:
            self._add_plot_set(res_A, res_B, "Total Envelope", 
                             geom_A=res_A['Selfweight'], geom_B=res_B['Selfweight'])
        
        self.elements.append(Spacer(1, 0.5*cm))
        self.elements.append(Paragraph(f"Tabular Summary ({res_mode})", self.styles['Heading3']))
        
        all_ids = sorted(list(set(res_A['Total Envelope'].keys()) | set(res_B['Total Envelope'].keys())), 
                         key=lambda x: (x[0], int(x[1:])))
        
        table_data = [["Elem", "M_Max", "M_Min", "V_Max", "V_Min", "N_Max", "N_Min", "Def_Max", "Def_Min"]]
        col_widths = [1.5*cm] + [2.0*cm]*8
        
        for eid in all_ids:
            row = [eid]
            dA = res_A['Total Envelope'].get(eid, {})
            dB = res_B['Total Envelope'].get(eid, {})
            
            def fmt_val(val_dict, k):
                if not val_dict: return 0.0
                arr = val_dict.get(k, [0.0])
                if isinstance(arr, (float, int)): return float(arr)
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

    def _add_component_section(self, load_key):
        def is_active(p, key):
            if key == "Selfweight": return any(v > 0 for v in p['sw_list'])
            if key == "Soil": return len(p.get('soil', [])) > 0
            if key == "Surcharge": return len(p.get('surcharge', [])) > 0
            return False

        act_A = is_active(self.params_A, load_key)
        act_B = is_active(self.params_B, load_key)
        
        if not act_A and not act_B: return
        
        res_A = solver.combine_results(self.raw_A, self.params_A, "Characteristic (No Dynamic Factor)")
        res_B = solver.combine_results(self.raw_B, self.params_B, "Characteristic (No Dynamic Factor)")
        
        self.elements.append(Paragraph(f"Load Case: {load_key} (Unfactored, Phi=1.0)", self.styles['Heading3']))
        self._add_plot_set(res_A, res_B, load_key, geom_A=res_A['Selfweight'], geom_B=res_B['Selfweight'])
        self.elements.append(Spacer(1, 0.5*cm))

    def _add_smart_vehicle_steps(self):
        # This function must scan each system independently and plot ONLY that system's critical steps.
        
        # --- Helper for Scanning & Plotting ---
        def process_system(params, raw_data, sys_label, sys_nodes):
            steps = raw_data.get('Vehicle Steps A', [])
            if not steps: return
            
            num_spans = params['num_spans']
            step_map = {} # idx -> {desc:[], types: set}
            
            # Scan Spans
            for i in range(num_spans):
                eid = f"S{i+1}"
                max_M, idx_max_M = -1e9, -1
                min_M, idx_min_M = 1e9, -1
                max_V, idx_max_V = -1e9, -1
                min_V, idx_min_V = 1e9, -1
                
                found_span = False
                for idx, step in enumerate(steps):
                    res = step['res']
                    if eid in res:
                        found_span = True
                        m_arr = res[eid]['M']; v_arr = res[eid]['V']
                        mx_m = np.max(m_arr); mn_m = np.min(m_arr)
                        mx_v = np.max(v_arr); mn_v = np.min(v_arr)
                        
                        if mx_m > max_M: max_M = mx_m; idx_max_M = idx
                        if mn_m < min_M: min_M = mn_m; idx_min_M = idx
                        if mx_v > max_V: max_V = mx_v; idx_max_V = idx
                        if mn_v < min_V: min_V = mn_v; idx_min_V = idx
                
                if not found_span: continue
                
                def reg(idx, desc, type_code):
                    if idx == -1: return
                    if idx not in step_map: step_map[idx] = {'desc': [], 'types': set()}
                    step_map[idx]['desc'].append(desc)
                    step_map[idx]['types'].add(type_code)

                reg(idx_max_M, f"Max M ({eid})", 'M')
                reg(idx_min_M, f"Min M ({eid})", 'M')
                reg(idx_max_V, f"Max V ({eid})", 'V')
                reg(idx_min_V, f"Min V ({eid})", 'V')
            
            if not step_map: return
            
            self.elements.append(Paragraph(f"<b>{sys_label} ({params.get('name')})</b>", self.styles['Heading4']))
            
            sorted_steps = sorted(step_map.keys())
            for idx in sorted_steps:
                info = step_map[idx]
                desc_str = ", ".join(info['desc'])
                step = steps[idx]
                x_loc = step['x']
                
                self.elements.append(Paragraph(f"Position X = {x_loc:.2f} m (Step {idx}) - Critical for: {desc_str}", self.styles['SwecoCell']))
                
                for type_code in sorted(list(info['types'])):
                    title_map = {'M': "Bending Moment [kNm]", 'V': "Shear Force [kN]"}
                    try:
                        # KEY: Show A=True, Show B=False (if system A)
                        is_A = (sys_label == "System A")
                        
                        fig = viz.create_plotly_fig(
                            sys_nodes, 
                            step['res'] if is_A else {}, # A Data
                            step['res'] if not is_A else {}, # B Data
                            type_base=type_code, 
                            title=f"{title_map.get(type_code,'')} @ X={x_loc:.2f}m",
                            load_case_name="Vehicle Steps",
                            name_A=self.params_A['name'], name_B=self.params_B['name'],
                            geom_A=self.raw_A.get('Selfweight'), geom_B=self.raw_B.get('Selfweight'),
                            show_A=is_A, show_B=(not is_A),
                            params_A=self.params_A, params_B=self.params_B,
                            show_supports=True,
                            font_scale=1.5 
                        )
                        self._add_fig_image(fig)
                    except: pass
                self.elements.append(Spacer(1, 0.3*cm))
        
        # EXECUTE SCAN FOR BOTH
        process_system(self.params_A, self.raw_A, "System A", self.nodes_A)
        self.elements.append(Spacer(1, 0.5*cm))
        process_system(self.params_B, self.raw_B, "System B", self.nodes_B)

    def _add_plot_set(self, res_A, res_B, case_name, geom_A, geom_B):
        types = [('M', 'Bending Moment [kNm]'), ('V', 'Shear Force [kN]'), 
                 ('N', 'Normal Force [kN]'), ('Def', 'Deformation [mm]')]
        img_flowables = []
        
        for t_code, t_title in types:
            try:
                fig = viz.create_plotly_fig(
                    self.nodes_A, res_A.get(case_name, {}), res_B.get(case_name, {}), 
                    type_base=t_code, title=t_title, load_case_name=case_name,
                    name_A=self.params_A['name'], name_B=self.params_B['name'],
                    geom_A=geom_A, geom_B=geom_B,
                    params_A=self.params_A, params_B=self.params_B,
                    show_supports=True,
                    font_scale=1.5 # Adjusted scale
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

    def _fig_to_bytes(self, fig, scale=2.0):
        try:
            b = io.BytesIO()
            fig.write_image(b, format='png', scale=scale)
            b.seek(0)
            return b
        except: return None

    def _make_std_table(self, data, col_widths, font_size=9):
        # Forced Left Align
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