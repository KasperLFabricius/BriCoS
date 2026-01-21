import streamlit as st
import pandas as pd
import numpy as np
import copy
import bricos_solver as solver
import bricos_viz as viz

# ==========================================
# HELPER FUNCTIONS (MATH & FORMATTING)
# ==========================================

def get_peaks(r_dict, key_max, key_min):
    """Finds absolute max/min values in a result dictionary for summary tables."""
    if not r_dict: return None, None
    
    has_env = (key_max in r_dict)
    has_step = ('M' in r_dict) and not has_env
    
    val_max, val_min = -1e9, 1e9
    found = False
    
    if has_env:
        val_max = np.max(r_dict[key_max])
        val_min = np.min(r_dict[key_min])
        found = True
    elif has_step:
        base_k = key_max.replace("_max", "")
        if base_k in r_dict:
            arr = r_dict[base_k]
            val_max = np.max(arr)
            val_min = np.min(arr)
            found = True
    
    if not found: return None, None
    return val_max, val_min

def calc_diff(val_a, val_b, is_max_case=True):
    """Calculates percentage difference for comparison tables."""
    if val_a is None or val_b is None: return np.nan
    
    # Guard for zero division
    denom = abs(val_a)
    if denom < 1e-6:
        if abs(val_b) < 1e-6: return 0.0
        return 9999.0 # Placeholder for Infinity
    
    if is_max_case:
        # For MAX: Algebraic Increase = Red.
        diff = (val_b - val_a)
        return (diff / denom) * 100.0
    else:
        # For MIN: Algebraic Decrease (More Negative) = Red.
        diff = (val_a - val_b)
        return (diff / denom) * 100.0

def color_diff(val):
    """Pandas Styler: Colors cells based on diff value (Red=Worse, Green=Better)."""
    if pd.isna(val): return ""
    if val > 0.05: return 'color: red; font-weight: bold' 
    if val < -0.05: return 'color: green; font-weight: bold' 
    return 'color: gray'

def fmt_pct_cap(val):
    """Pandas Styler: Formats percentage strings."""
    if pd.isna(val): return "--"
    if not isinstance(val, (int, float)): return str(val)
    
    if val > 999.0: return ">999%"
    if val < -999.0: return "<-999%"
    return "{:+.1f}%".format(val)

def get_reaction_envelope(res_dict, nodes_dict, mode):
    """Extracts reaction forces from envelope results."""
    reacts = {}
    if not res_dict or not nodes_dict: return reacts
    
    for eid, dat in res_dict.items():
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
        
        # Helper to safely get values
        def get_val(key, idx):
            if key in dat: return dat[key][idx] 
            elif key.replace("_max","") in dat: 
                 return dat[key.replace("_max","")][idx]
            return 0.0

        # Start Node Processing
        n_mx = get_val('N_max', 0); n_mn = get_val('N_min', 0)
        v_mx = get_val('V_max', 0); v_mn = get_val('V_min', 0)
        m_mx = get_val('M_max', 0); m_mn = get_val('M_min', 0)
        
        def get_bounds(c_fac, s_fac):
            vals = []
            for n_v in [n_mx, n_mn]:
                for v_v in [v_mx, v_mn]:
                    vals.append(c_fac*n_v - s_fac*v_v)
            return max(vals), min(vals)
        
        fx_mx, fx_mn = get_bounds(c, s)
        fy_mx, fy_mn = get_bounds(s, -c) 
        
        y_start = nodes_dict[dat['ni_id']][1]
        is_supp_start = (y_start < -0.01) if mode == 'Frame' else (dat['ni_id'] >= 200) 
        if is_supp_start: add_to_node(dat['ni_id'], fx_mx, fx_mn, fy_mx, fy_mn, m_mx, m_mn)

        # End Node Processing
        n_mx = get_val('N_max', -1); n_mn = get_val('N_min', -1)
        v_mx = get_val('V_max', -1); v_mn = get_val('V_min', -1)
        m_mx = get_val('M_max', -1); m_mn = get_val('M_min', -1)
        
        n_mx, n_mn = -n_mn, -n_mx 
        v_mx, v_mn = -v_mn, -v_mx
        m_mx, m_mn = -m_mn, -m_mx
        
        fx_mx, fx_mn = get_bounds(c, s)
        fy_mx, fy_mn = get_bounds(s, -c)
        
        y_end = nodes_dict[dat['nj_id']][1]
        is_supp_end = (y_end < -0.01) if mode == 'Frame' else (dat['nj_id'] >= 200)
        if is_supp_end: add_to_node(dat['nj_id'], fx_mx, fx_mn, fy_mx, fy_mn, m_mx, m_mn)
             
    return reacts

# ==========================================
# MAIN RENDERER
# ==========================================

def render_results_section(sysA, sysB, raw_res_A, raw_res_B, nodes_A, nodes_B):
    """
    Main controller for the Results UI section.
    Handles Toolbar, Result Combination, and Tab Rendering.
    """
    # --- 0. VALIDITY GATEKEEPING ---
    valid_B = (nodes_B is not None)
    
    # --- 1. VISUAL CONTROL SETTINGS ---
    r1_col1, r1_col2 = st.columns([3, 1])
    ui_locked = st.session_state.get('is_generating_report', False)
    
    with r1_col1:
        man_scale = st.slider("Target Diagram Height [m]", 0.5, 10.0, float(sysA.get('scale_manual', 2.0)), 0.1, disabled=ui_locked)
    with r1_col2:
        show_labels = st.checkbox("Labels", value=True, disabled=ui_locked)

    r2_col1, r2_col2 = st.columns([3, 1])
    with r2_col1:
        support_size = st.slider("Support Size", 0.1, 2.0, 0.5, 0.1, disabled=ui_locked)
    with r2_col2:
        show_supports = st.checkbox("Show Supports", value=True, disabled=ui_locked)

    # Persist Visual Settings
    sysA['scale_manual'] = man_scale
    sysB['scale_manual'] = man_scale

    # --- 2. STICKY TOOLBAR & COMBINATION ---
    view_options = ["Total Envelope", "Selfweight", "Soil", "Surcharge", "Vehicle Envelope", "Vehicle Steps"]
    
    # Ensure session state for selector persistence
    if 'view_case_selector' not in st.session_state: st.session_state.view_case_selector = "Total Envelope"
    
    def set_view_case(): st.session_state.keep_view_case = st.session_state.view_case_selector
    try: v_idx = view_options.index(st.session_state.keep_view_case)
    except ValueError: v_idx = 0

    if 'result_mode' not in st.session_state: st.session_state['result_mode'] = "Design (ULS)"

    with st.container():
        st.markdown('<div id="sticky-results-marker"></div>', unsafe_allow_html=True)
        c_res_tool1, c_res_tool2, c_res_tool3 = st.columns([2, 2, 2])

        view_case = c_res_tool1.selectbox("Load Case", view_options, index=v_idx, key="view_case_selector", on_change=set_view_case, disabled=ui_locked)

        show_sys_mode = "System A"
        if view_case != "Vehicle Steps":
            if valid_B:
                tog_map = {"Both": "Both", "System A": sysA['name'], "System B": sysB['name']}
                show_sys_mode = c_res_tool2.radio("Active Systems View", ["Both", "System A", "System B"], format_func=lambda x: tog_map[x], horizontal=True, key="sys_view_toggle", disabled=ui_locked)
            else:
                c_res_tool2.info("Comparison Disabled (Sys B Empty)")

        curr_res_mode = st.session_state.get('result_mode', "Design (ULS)")
        res_opts = ["Design (ULS)", "Characteristic (SLS)", "Characteristic (No Dynamic Factor)"]
        try: res_idx = res_opts.index(curr_res_mode)
        except: res_idx = 0
        st.session_state['result_mode'] = c_res_tool3.radio("Result Type", res_opts, index=res_idx, horizontal=True, key="result_mode_main_ui", disabled=ui_locked)
        result_mode_val = st.session_state['result_mode']

    # --- 3. COMBINE RESULTS ---
    has_res_A = (raw_res_A is not None) and (nodes_A is not None)
    has_res_B = (raw_res_B is not None) and (nodes_B is not None)

    res_A = solver.combine_results(raw_res_A, sysA, result_mode_val) if has_res_A else {}
    res_B = {}
    if valid_B and has_res_B:
        res_B = solver.combine_results(raw_res_B, sysB, result_mode_val)

    # --- 4. PREPARE VIEW DATA (STEPS VS ENVELOPE) ---
    rA, rB = {}, {}
    step_view_sys = "System A"
    active_veh_step = "Vehicle A"
    show_A_step = True
    show_B_step = False
    list_A = []
    list_B = []

    if view_case == "Vehicle Steps":
        st.markdown("---")
        
        # Direction Logic
        is_both_active = (sysA['vehicle_direction'] == 'Both')
        is_reverse_only = (sysA['vehicle_direction'] == 'Reverse')
        step_dir_suffix = ""
        
        if is_both_active:
            c_veh_tog, c_dir_tog, c_step_slide, c_step_tog = st.columns([1, 1, 2, 1])
            step_dir_sel = c_dir_tog.radio("Step Direction:", ["Forward", "Reverse"], horizontal=True, key="step_dir_radio", disabled=ui_locked)
            if step_dir_sel == "Reverse": step_dir_suffix = "_Rev"
        elif is_reverse_only:
            c_veh_tog, c_step_slide, c_step_tog = st.columns([1, 2, 1])
            step_dir_suffix = "_Rev"
        else:
            c_veh_tog, c_step_slide, c_step_tog = st.columns([1, 2, 1])
        
        def set_anim_veh(): st.session_state.keep_active_veh_step = st.session_state.anim_veh_radio
        try: av_idx = ["Vehicle A", "Vehicle B"].index(st.session_state.keep_active_veh_step)
        except ValueError: av_idx = 0
        active_veh_step = c_veh_tog.radio("Anim Vehicle:", ["Vehicle A", "Vehicle B"], index=av_idx, horizontal=True, key="anim_veh_radio", on_change=set_anim_veh, disabled=ui_locked)
        
        base_key = "Vehicle Steps A" if active_veh_step == "Vehicle A" else "Vehicle Steps B"
        veh_key_res = f"{base_key}{step_dir_suffix}"
        
        list_A = res_A.get(veh_key_res, [])
        list_B = res_B.get(veh_key_res, []) if valid_B else []
        
        if valid_B:
            def set_step_sys(): st.session_state.keep_step_view_sys = st.session_state.step_sys_radio
            try: ss_idx = ["Both", "System A", "System B"].index(st.session_state.keep_step_view_sys)
            except ValueError: ss_idx = 0
            step_tog_map = {"Both": "Both", "System A": sysA['name'], "System B": sysB['name']}
            step_view_sys = c_step_tog.radio("View System:", ["Both", "System A", "System B"], index=ss_idx, format_func=lambda x: step_tog_map[x], horizontal=True, key="step_sys_radio", on_change=set_step_sys, disabled=ui_locked)
            
            show_A_step = (step_view_sys == "Both" or step_view_sys == "System A")
            show_B_step = (step_view_sys == "Both" or step_view_sys == "System B")
        else:
             c_step_tog.caption(f"View: {sysA['name']}")
             show_A_step = True
             show_B_step = False
        
        valid_A_dat = len(list_A) > 0
        valid_B_dat = len(list_B) > 0
        
        if show_A_step and not valid_A_dat:
            st.warning(f"⚠️ {active_veh_step} is not defined for {sysA['name']} (or has no steps).")
        if show_B_step and not valid_B_dat:
            st.warning(f"⚠️ {active_veh_step} is not defined for {sysB['name']} (or has no steps).")
            
        if (show_A_step and valid_A_dat) or (show_B_step and valid_B_dat):
            max_steps = max(1, len(list_A), len(list_B))
            step_idx = c_step_slide.slider("Step Index", 0, max_steps-1, 0, key="veh_step_slider_persistent", disabled=ui_locked)
            
            st.markdown("---")
            def get_step(res, idx, k_res, f_factor):
                s_list = res.get(k_res, [])
                if idx < len(s_list):
                    step_data = s_list[idx]['res']
                    out = {}
                    for k, v in step_data.items():
                        # Scale loads for visualization
                        scaled_loads = []
                        if 'loads' in v:
                            for l in v['loads']:
                                new_l = copy.deepcopy(l)
                                if new_l['params']:
                                    new_l['params'][0] *= f_factor
                                scaled_loads.append(new_l)

                        out[k] = {**v, 
                            'loads': scaled_loads,
                            'M':v['M']*f_factor, 'V':v['V']*f_factor, 'N':v['N']*f_factor, 
                            'M_max':v['M']*f_factor, 'M_min':v['M']*f_factor,
                            'V_max':v['V']*f_factor, 'V_min':v['V']*f_factor,
                            'N_max':v['N']*f_factor, 'N_min':v['N']*f_factor,
                            'def_x':v['def_x']*f_factor, 'def_y':v['def_y']*f_factor,
                            'def_x_max':v['def_x']*f_factor, 'def_x_min':v['def_x']*f_factor,
                            'def_y_max':v['def_y']*f_factor, 'def_y_min':v['def_y']*f_factor
                        }
                    return out
                return {}

            f_A = res_A['f_vehA'] if active_veh_step == "Vehicle A" and has_res_A else 1.0
            f_B = res_B['f_vehA'] if active_veh_step == "Vehicle A" and valid_B and has_res_B else 1.0
            if active_veh_step == "Vehicle B":
                 f_A = res_A['f_vehB'] if has_res_A else 1.0
                 f_B = res_B['f_vehB'] if valid_B and has_res_B else 1.0
                 
            rA = get_step(res_A, step_idx, veh_key_res, f_A)
            rB = get_step(res_B, step_idx, veh_key_res, f_B) if valid_B else {}
    else:
        key_map = {"Total Envelope": "Total Envelope", "Selfweight": "Selfweight", "Soil": "Soil", "Surcharge": "Surcharge", "Vehicle Envelope": "Vehicle Envelope"}
        target_key = key_map.get(view_case, "Total Envelope")
        rA = res_A.get(target_key, {})
        rB = res_B.get(target_key, {}) if valid_B else {}

    # --- 5. RENDER TABS ---
    t1, t2, t3 = st.tabs(["Visualization", "Tabular Data", "Summary"])
    name_A = sysA['name']
    name_B = sysB['name'] if valid_B else "System B"
    
    # --- TAB 1: VISUALIZATION ---
    with t1:
        if view_case == "Vehicle Steps":
            valid_A_dat = len(list_A) > 0
            valid_B_dat = len(list_B) > 0
            has_vis_content = (show_A_step and valid_A_dat) or (show_B_step and valid_B_dat)
            
            if not has_vis_content:
                 st.info("No visualization available for selected system/vehicle combination.")
            else:
                _render_viz_chart("Bending Moment [kNm]", nodes_A, rA, rB, 'M', man_scale, show_A_step, show_B_step, show_labels, view_case, name_A, name_B, res_A, res_B, sysA, sysB, show_supports, support_size)
                _render_viz_chart("Shear Force [kN]", nodes_A, rA, rB, 'V', man_scale, show_A_step, show_B_step, show_labels, view_case, name_A, name_B, res_A, res_B, sysA, sysB, show_supports, support_size)
                _render_viz_chart("Normal Force [kN]", nodes_A, rA, rB, 'N', man_scale, show_A_step, show_B_step, show_labels, view_case, name_A, name_B, res_A, res_B, sysA, sysB, show_supports, support_size)
                _render_viz_chart("Deformation [mm]", nodes_A, rA, rB, 'Def', man_scale, show_A_step, show_B_step, show_labels, view_case, name_A, name_B, res_A, res_B, sysA, sysB, show_supports, support_size)
        else:
            show_A = (show_sys_mode == "Both" or show_sys_mode == "System A")
            show_B = (valid_B and (show_sys_mode == "Both" or show_sys_mode == "System B"))
            
            geom_invalid_A = (nodes_A is None) or (len(nodes_A)==0)
            geom_invalid_B = (valid_B) and ((nodes_B is None) or (len(nodes_B)==0))
            
            if geom_invalid_A and geom_invalid_B: 
                 st.warning("⚠️ No structural geometry defined. Please configure Spans/Walls in the sidebar.")
            else:
                 if (not rA) and (not rB): st.warning(f"⚠️ No results found for **{view_case}**.")

                 _render_viz_chart("Bending Moment [kNm]", nodes_A, rA, rB, 'M', man_scale, show_A, show_B, show_labels, view_case, name_A, name_B, res_A, res_B, sysA, sysB, show_supports, support_size)
                 _render_viz_chart("Shear Force [kN]", nodes_A, rA, rB, 'V', man_scale, show_A, show_B, show_labels, view_case, name_A, name_B, res_A, res_B, sysA, sysB, show_supports, support_size)
                 _render_viz_chart("Normal Force [kN]", nodes_A, rA, rB, 'N', man_scale, show_A, show_B, show_labels, view_case, name_A, name_B, res_A, res_B, sysA, sysB, show_supports, support_size)
                 _render_viz_chart("Deformation [mm]", nodes_A, rA, rB, 'Def', man_scale, show_A, show_B, show_labels, view_case, name_A, name_B, res_A, res_B, sysA, sysB, show_supports, support_size)

    # --- TAB 2: DETAILED DATA ---
    with t2:
        st.markdown(f"### Detailed Data ({view_case})")
        detailed_rows = []
        
        def process_detailed(r_dict, sys_name):
            if not r_dict: return
            for eid, data in r_dict.items():
                x_vals = data.get('x', [])
                n_pts = len(x_vals)
                
                def get_arr(key):
                    arr = data.get(key)
                    if arr is None: return np.zeros(n_pts)
                    return arr
                
                if view_case == "Vehicle Steps":
                    m = get_arr('M'); v = get_arr('V'); n = get_arr('N')
                    dx = get_arr('def_x'); dy = get_arr('def_y')
                    for i in range(n_pts):
                        detailed_rows.append({
                            "System": sys_name, "Element": eid, "Location [m]": x_vals[i],
                            "M [kNm]": m[i], "V [kN]": v[i], "N [kN]": n[i],
                            "Def_X [mm]": dx[i]*1000, "Def_Y [mm]": dy[i]*1000
                        })
                else:
                    m_max = get_arr('M_max'); m_min = get_arr('M_min')
                    v_max = get_arr('V_max'); v_min = get_arr('V_min')
                    n_max = get_arr('N_max'); n_min = get_arr('N_min')
                    dx_max = get_arr('def_x_max'); dx_min = get_arr('def_x_min')
                    dy_max = get_arr('def_y_max'); dy_min = get_arr('def_y_min')
                    
                    if 'M_max' not in data and 'M' in data:
                         m_max = data['M']; m_min = data['M']
                         v_max = data['V']; v_min = data['V']
                         n_max = data['N']; n_min = data['N']
                         dx_max = data['def_x']; dx_min = data['def_x']
                         dy_max = data['def_y']; dy_min = data['def_y']

                    for i in range(n_pts):
                        detailed_rows.append({
                            "System": sys_name, "Element": eid, "Location [m]": x_vals[i],
                            "M_max [kNm]": m_max[i], "M_min [kNm]": m_min[i],
                            "V_max [kN]": v_max[i], "V_min [kN]": v_min[i],
                            "N_max [kN]": n_max[i], "N_min [kN]": n_min[i],
                            "Def_X_max [mm]": dx_max[i]*1000, "Def_X_min [mm]": dx_min[i]*1000,
                            "Def_Y_max [mm]": dy_max[i]*1000, "Def_Y_min [mm]": dy_min[i]*1000
                        })

        process_detailed(rA, name_A)
        if valid_B:
            process_detailed(rB, name_B)
        
        if detailed_rows:
            df_detailed = pd.DataFrame(detailed_rows)
            st.dataframe(df_detailed, width='stretch')
            st.download_button(
                "Download Detailed Data (.csv)", 
                df_detailed.to_csv(index=False).encode('utf-8'), 
                f"bricos_detailed_{view_case.replace(' ', '_')}.csv", 
                "text/csv",
                disabled=ui_locked
            )
        else:
            st.info("No detailed data available for this view.")

    # --- TAB 3: SUMMARY COMPARISON ---
    with t3:
        st.subheader(f"Summary ({view_case})")
        
        all_elems = sorted(list(set(rA.keys()) | set(rB.keys())), key=lambda x: (x[0], int(x[1:])))

        # A. Forces Tables
        _render_summary_table("Bending Moment", [("M_max", "M_min", "M [kNm]")], all_elems, rA, rB, valid_B)
        _render_summary_table("Shear Force", [("V_max", "V_min", "V [kN]")], all_elems, rA, rB, valid_B)
        _render_summary_table("Normal Force", [("N_max", "N_min", "N [kN]")], all_elems, rA, rB, valid_B)

        # B. Deformations Table
        st.markdown("##### Deformations (Spans: Vertical, Walls: Horizontal)")
        def_rows = []
        for eid in all_elems:
            row_dat = {"Element": eid}
            dataA = rA.get(eid, {})
            dataB = rB.get(eid, {}) if valid_B else {}
            
            is_wall = eid.startswith("W")
            k_max = "def_x_max" if is_wall else "def_y_max"
            k_min = "def_x_min" if is_wall else "def_y_min"
            
            a_mx, a_mn = get_peaks(dataA, k_max, k_min)
            if a_mx is not None: a_mx *= 1000; a_mn *= 1000
            
            if valid_B:
                b_mx, b_mn = get_peaks(dataB, k_max, k_min)
                if b_mx is not None: b_mx *= 1000; b_mn *= 1000
                d_mx = calc_diff(a_mx, b_mx, True)
                d_mn = calc_diff(a_mn, b_mn, False)
                row_dat[f"Def (Max) A"] = f"{a_mx:.1f}" if a_mx is not None else "--"
                row_dat[f"Def (Max) B"] = f"{b_mx:.1f}" if b_mx is not None else "--"
                row_dat[f"Def (Max) %"] = d_mx 
                row_dat[f"Def (Min) A"] = f"{a_mn:.1f}" if a_mn is not None else "--"
                row_dat[f"Def (Min) B"] = f"{b_mn:.1f}" if b_mn is not None else "--"
                row_dat[f"Def (Min) %"] = d_mn
            else:
                 row_dat[f"Def (Max) [mm]"] = f"{a_mx:.1f}" if a_mx is not None else "--"
                 row_dat[f"Def (Min) [mm]"] = f"{a_mn:.1f}" if a_mn is not None else "--"
            
            row_dat["Type"] = "Wall (Horiz)" if is_wall else "Span (Vert)"
            def_rows.append(row_dat)
            
        if def_rows:
            df_def = pd.DataFrame(def_rows)
            cols = ['Element', 'Type'] + [c for c in df_def.columns if c not in ['Element', 'Type']]
            df_def = df_def[cols]
            pct_cols_d = [c for c in df_def.columns if "%" in c]
            st.dataframe(
                df_def.style.map(color_diff, subset=pct_cols_d).format(fmt_pct_cap, subset=pct_cols_d, na_rep="--"),
                height=200, width='stretch'
            )
        
        # C. Reactions
        st.markdown("##### Envelope Support Reactions")
        reactsA = get_reaction_envelope(rA, nodes_A, sysA['mode'])
        reactsB = get_reaction_envelope(rB, nodes_B, sysB['mode']) if valid_B else {}
        
        all_react_nodes = sorted(list(set(reactsA.keys()) | set(reactsB.keys())))
        r_rows = []
        
        for nid in all_react_nodes:
            label = f"Node {nid}"
            if nid >= 200: label = f"Support {nid-200+1}"
            elif nid >= 100: label = f"Wall {nid-100+1} Base"
            
            row = {"Location": label}
            dA = reactsA.get(nid, {})
            dB = reactsB.get(nid, {}) if valid_B else {}
            
            for comp in ['Rx', 'Ry', 'Mz']:
                for bnd in ['max', 'min']:
                    key = f"{comp}_{bnd}"
                    valA = dA.get(key)
                    
                    if valid_B:
                        valB = dB.get(key)
                        row[f"{comp} ({bnd}) A"] = f"{valA:.1f}" if valA is not None else "--"
                        row[f"{comp} ({bnd}) B"] = f"{valB:.1f}" if valB is not None else "--"
                        row[f"{comp} ({bnd}) %"] = calc_diff(valA, valB, is_max_case=(bnd=='max'))
                    else:
                        row[f"{comp} ({bnd})"] = f"{valA:.1f}" if valA is not None else "--"
            r_rows.append(row)
        
        if r_rows:
            df_react = pd.DataFrame(r_rows)
            pct_cols_r = [c for c in df_react.columns if "%" in c]
            st.dataframe(
                df_react.style.map(color_diff, subset=pct_cols_r).format(fmt_pct_cap, subset=pct_cols_r, na_rep="--"),
                width='stretch'
            )
        else:
            st.info("No reaction data found (check supports).")

def _render_viz_chart(title, nodes, rA, rB, type_base, scale, show_A, show_B, show_labels, view_case, name_A, name_B, res_A, res_B, sysA, sysB, show_supports, support_size):
    st.subheader(title)
    st.plotly_chart(viz.create_plotly_fig(
        nodes, rA, rB, type_base, scale, "", 
        show_A, show_B, show_labels, view_case, 
        name_A, name_B, 
        geom_A=res_A.get('Selfweight'), geom_B=res_B.get('Selfweight'),
        params_A=sysA, params_B=sysB,
        show_supports=show_supports, support_size=support_size
    ), width='stretch', key=f"chart_{type_base}")

def _render_summary_table(title, metrics_list, all_elems, rA, rB, valid_B):
    st.markdown(f"##### {title}")
    rows = []
    
    for eid in all_elems:
        row_dat = {"Element": eid}
        dataA = rA.get(eid, {})
        dataB = rB.get(eid, {}) if valid_B else {}
        
        for k_max, k_min, label in metrics_list:
            is_def = "def" in k_max
            scale = 1000.0 if is_def else 1.0
            
            a_mx, a_mn = get_peaks(dataA, k_max, k_min)
            if a_mx is not None: a_mx *= scale; a_mn *= scale
            
            if valid_B:
                b_mx, b_mn = get_peaks(dataB, k_max, k_min)
                if b_mx is not None: b_mx *= scale; b_mn *= scale
                
                d_mx = calc_diff(a_mx, b_mx, is_max_case=True)
                d_mn = calc_diff(a_mn, b_mn, is_max_case=False)
                
                # Max Cols
                row_dat[f"{label} (Max) A"] = f"{a_mx:.1f}" if a_mx is not None else "--"
                row_dat[f"{label} (Max) B"] = f"{b_mx:.1f}" if b_mx is not None else "--"
                row_dat[f"{label} (Max) %"] = d_mx 
                
                # Min Cols
                row_dat[f"{label} (Min) A"] = f"{a_mn:.1f}" if a_mn is not None else "--"
                row_dat[f"{label} (Min) B"] = f"{b_mn:.1f}" if b_mn is not None else "--"
                row_dat[f"{label} (Min) %"] = d_mn
            else:
                row_dat[f"{label} (Max)"] = f"{a_mx:.1f}" if a_mx is not None else "--"
                row_dat[f"{label} (Min)"] = f"{a_mn:.1f}" if a_mn is not None else "--"
        
        rows.append(row_dat)
        
    if not rows:
        st.caption("No elements found.")
        return

    df = pd.DataFrame(rows)
    pct_cols = [c for c in df.columns if "%" in c]
    
    st.dataframe(
        df.style.map(color_diff, subset=pct_cols).format(fmt_pct_cap, subset=pct_cols, na_rep="--"),
        height=200, width='stretch'
    )