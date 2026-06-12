import streamlit as st
import pandas as pd
import numpy as np
import bricos_solver as solver
import bricos_viz as viz
import bricos_export as export_mod

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

def get_reaction_envelope(res_dict, nodes_dict, mode, restrained_nodes=None):
    """Extracts reaction forces from envelope results.

    restrained_nodes: collection of node ids carrying boundary springs, as
    reported by the solver ('Restrained Nodes'). When provided it is the
    authoritative support test; the id/coordinate heuristic is only a
    fallback for results produced before the key existed. The heuristic
    misses Frame-mode supports at zero-height walls, where the restraint
    sits at the top node (200+i) at y=0.
    """
    reacts = {}
    if not res_dict or not nodes_dict: return reacts

    restrained_set = set(restrained_nodes) if restrained_nodes else None

    def is_support(nid):
        if restrained_set is not None:
            return nid in restrained_set
        y_node = nodes_dict[nid][1]
        return (y_node < -0.01) if mode == 'Frame' else (nid >= 200)
    
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

        # Start Node Processing. The M field is sagging-positive (v0.58);
        # nodal reaction moments stay in the global CCW-positive
        # convention, so the start node negates the field (M_nodal(0) =
        # -M_field(0)) and the end node uses it directly (the historical
        # end-equilibrium negation and the field flip cancel).
        n_mx = get_val('N_max', 0); n_mn = get_val('N_min', 0)
        v_mx = get_val('V_max', 0); v_mn = get_val('V_min', 0)
        m_mx = get_val('M_max', 0); m_mn = get_val('M_min', 0)
        m_mx, m_mn = -m_mn, -m_mx

        def get_bounds(c_fac, s_fac):
            vals = []
            for n_v in [n_mx, n_mn]:
                for v_v in [v_mx, v_mn]:
                    vals.append(c_fac*n_v - s_fac*v_v)
            return max(vals), min(vals)

        fx_mx, fx_mn = get_bounds(c, s)
        fy_mx, fy_mn = get_bounds(s, -c)

        if is_support(dat['ni_id']): add_to_node(dat['ni_id'], fx_mx, fx_mn, fy_mx, fy_mn, m_mx, m_mn)

        # End Node Processing
        n_mx = get_val('N_max', -1); n_mn = get_val('N_min', -1)
        v_mx = get_val('V_max', -1); v_mn = get_val('V_min', -1)
        m_mx = get_val('M_max', -1); m_mn = get_val('M_min', -1)

        n_mx, n_mn = -n_mn, -n_mx
        v_mx, v_mn = -v_mn, -v_mx

        fx_mx, fx_mn = get_bounds(c, s)
        fy_mx, fy_mn = get_bounds(s, -c)

        if is_support(dat['nj_id']): add_to_node(dat['nj_id'], fx_mx, fx_mn, fy_mx, fy_mn, m_mx, m_mn)
             
    return reacts

# Envelope components selectable for critical-step navigation. Values are
# (base result key, side); 'def' resolves per member (def_x for walls,
# def_y for spans). Sign convention (v0.58): sagging is positive, so
# "Bending (Max)" navigates to the governing sagging step.
STEP_COMPONENT_OPTIONS = {
    "Bending (Max)": ("M", "max"), "Bending (Min)": ("M", "min"),
    "Shear (Max)": ("V", "max"), "Shear (Min)": ("V", "min"),
    "Normal force (Max)": ("N", "max"), "Normal force (Min)": ("N", "min"),
    "Deformation (Max)": ("def", "max"), "Deformation (Min)": ("def", "min"),
}

STEP_EFFECTS_VEHICLE = "Vehicle"
STEP_EFFECTS_COMBINED = "Vehicle + Traffic UDL"
STEP_EFFECTS_UDL = "Traffic UDL"


def resolve_component_key(eid, base_key):
    """Map the generic 'def' component to the member-specific deflection key."""
    if base_key == "def":
        return "def_x" if str(eid).startswith("W") else "def_y"
    return base_key


def step_display_loads(loads, f_factor, effects_mode):
    """Loads to draw on a step chart under the given effects mode.

    Vehicle axle point loads are scaled by the member's vehicle factor; in
    the UDL-only view they are omitted entirely - the axles are not part of
    the displayed effects, so drawing them was misleading.
    """
    out = []
    for load in loads or []:
        if effects_mode == STEP_EFFECTS_UDL and load.get('type') == 'point':
            continue
        new_l = {**load, 'params': list(load['params'])}
        if new_l['params']:
            new_l['params'][0] *= f_factor
        out.append(new_l)
    return out


def step_combined_field(res_el, base_key, side, f_veh, f_udl, effects_mode):
    """Factored per-step field for one component and adverse side.

    The UDL part uses the per-step adverse field for the requested side;
    missing UDL data degrades to the vehicle-only field.
    """
    key = base_key
    veh = res_el[key] * f_veh
    if effects_mode == STEP_EFFECTS_VEHICLE:
        return veh
    udl = res_el.get(f"{key}_udl_{side}")
    if udl is None:
        return veh if effects_mode == STEP_EFFECTS_COMBINED else np.zeros_like(veh)
    if effects_mode == STEP_EFFECTS_UDL:
        return udl * f_udl
    return veh + udl * f_udl


def find_critical_step(steps, eid, base_key, side, f_map, f_default, f_udl, effects_mode):
    """Index of the step producing the extreme factored value of a component.

    Returns (index, value) or (None, None) when the member never appears.
    """
    best_idx, best_val = None, None
    f_v = f_map.get(eid, f_default)
    for i, step in enumerate(steps):
        res_el = step.get("res", {}).get(eid)
        if res_el is None:
            continue
        key = resolve_component_key(eid, base_key)
        field = step_combined_field(res_el, key, side, f_v, f_udl, effects_mode)
        val = float(np.max(field)) if side == "max" else float(np.min(field))
        better = (best_val is None or
                  (side == "max" and val > best_val) or
                  (side == "min" and val < best_val))
        if better:
            best_idx, best_val = i, val
    return best_idx, best_val


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
    view_options = ["Total Envelope", "Selfweight", "Soil", "Surcharge", "Vehicle Envelope", "Traffic UDL", "Vehicle Steps"]
    
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

        # Result modes follow the limit-state toggles; the unfactored
        # combination is always available (and the only one when both
        # toggles are off).
        curr_res_mode = st.session_state.get('result_mode', "Design (ULS)")
        if curr_res_mode == "Characteristic (No Dynamic Factor)":
            curr_res_mode = "Unfactored"  # legacy name from saved sessions
        res_opts = []
        if sysA.get('analyze_uls', True): res_opts.append("Design (ULS)")
        if sysA.get('analyze_sls', True): res_opts.append("Characteristic (SLS)")
        res_opts.append("Unfactored")
        try: res_idx = res_opts.index(curr_res_mode)
        except ValueError: res_idx = 0
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

            if active_veh_step == "Vehicle A":
                f_A = res_A['f_vehA'] if has_res_A else 1.0
                f_B = res_B['f_vehA'] if valid_B and has_res_B else 1.0
                f_map_A = res_A.get('f_vehA_map', {}) if has_res_A else {}
                f_map_B = res_B.get('f_vehA_map', {}) if valid_B and has_res_B else {}
            else:
                f_A = res_A['f_vehB'] if has_res_A else 1.0
                f_B = res_B['f_vehB'] if valid_B and has_res_B else 1.0
                f_map_A = res_A.get('f_vehB_map', {}) if has_res_A else {}
                f_map_B = res_B.get('f_vehB_map', {}) if valid_B and has_res_B else {}
            f_udl_A = res_A.get('f_udl', 1.0) if has_res_A else 1.0
            f_udl_B = res_B.get('f_udl', 1.0) if valid_B and has_res_B else 1.0

            def _steps_have_udl(s_list):
                return bool(s_list) and any('M_udl_max' in v for v in s_list[0].get('res', {}).values())

            udl_in_steps = _steps_have_udl(list_A) or _steps_have_udl(list_B)
            if udl_in_steps:
                help_effects = (
                    "Which load effects the step results show: the vehicle alone, the "
                    "accompanying Traffic UDL alone, or both combined. With the UDL "
                    "included, the charts show a band between the adverse-minimum and "
                    "adverse-maximum UDL application for this step's window; values use "
                    "the factors of the selected Result Type. The combined view is the "
                    "same per-step coupling the Total Envelope is built from."
                )
                effects_mode = st.radio(
                    "Step effects:",
                    [STEP_EFFECTS_VEHICLE, STEP_EFFECTS_COMBINED, STEP_EFFECTS_UDL],
                    horizontal=True, key="step_effects_radio", disabled=ui_locked, help=help_effects)
            else:
                effects_mode = STEP_EFFECTS_VEHICLE

            # --- CRITICAL-STEP NAVIGATION ---
            if show_A_step and valid_A_dat:
                nav_steps, nav_fmap, nav_fdef, nav_fudl = list_A, f_map_A, f_A, f_udl_A
            else:
                nav_steps, nav_fmap, nav_fdef, nav_fudl = list_B, f_map_B, f_B, f_udl_B
            member_ids = sorted(
                {eid for s in nav_steps for eid in s.get('res', {}).keys()},
                key=lambda x: (x[0], int(x[1:])))

            def _jump_to_critical():
                eid = st.session_state.get("crit_member_sel")
                comp_label = st.session_state.get("crit_comp_sel")
                if not eid or comp_label not in STEP_COMPONENT_OPTIONS:
                    return
                base_key, side = STEP_COMPONENT_OPTIONS[comp_label]
                em = st.session_state.get("step_effects_radio", STEP_EFFECTS_VEHICLE)
                idx, _val = find_critical_step(
                    nav_steps, eid, base_key, side, nav_fmap, nav_fdef, nav_fudl, em)
                if idx is not None:
                    st.session_state["veh_step_slider_persistent"] = idx

            help_nav = (
                "Selecting a member and an envelope component moves the step slider to "
                "the vehicle position producing that extreme (with the current step-"
                "effects selection). E.g. S1 + Bending (Max) shows the governing "
                "sagging step for span 1 (sagging is positive)."
            )
            c_nav1, c_nav2, c_nav3 = st.columns([1, 1, 1])
            c_nav1.selectbox("Member:", member_ids, key="crit_member_sel",
                             on_change=_jump_to_critical, disabled=ui_locked, help=help_nav)
            c_nav2.selectbox("Envelope component:", list(STEP_COMPONENT_OPTIONS.keys()),
                             key="crit_comp_sel", on_change=_jump_to_critical, disabled=ui_locked)
            c_nav3.button("Go to critical step", on_click=_jump_to_critical, disabled=ui_locked)

            st.markdown("---")
            def get_step(res, idx, k_res, f_default, f_map, f_udl):
                # f_map holds per-member vehicle factors (eid -> factor incl.
                # phi); f_udl factors the per-step adverse UDL fields. The
                # step-effects selection decides what the charts show.
                s_list = res.get(k_res, [])
                if idx < len(s_list):
                    step_data = s_list[idx]['res']
                    out = {}
                    for k, v in step_data.items():
                        f_factor = f_map.get(k, f_default)
                        # Loads for visualization: factored, and without the
                        # axle arrows in the UDL-only view.
                        scaled_loads = step_display_loads(
                            v.get('loads'), f_factor, effects_mode)

                        out_el = {**v, 'loads': scaled_loads}
                        for base in ('M', 'V', 'N', 'def_x', 'def_y'):
                            veh = v[base] * f_factor
                            u_max = v.get(f'{base}_udl_max')
                            u_min = v.get(f'{base}_udl_min')
                            if effects_mode == STEP_EFFECTS_VEHICLE or u_max is None:
                                out_el[base] = veh
                                out_el[f'{base}_max'] = veh
                                out_el[f'{base}_min'] = veh
                            elif effects_mode == STEP_EFFECTS_COMBINED:
                                out_el[base] = veh
                                out_el[f'{base}_max'] = veh + u_max * f_udl
                                out_el[f'{base}_min'] = veh + u_min * f_udl
                            else:  # Traffic UDL alone
                                z = np.zeros_like(veh)
                                out_el[base] = z
                                out_el[f'{base}_max'] = u_max * f_udl
                                out_el[f'{base}_min'] = u_min * f_udl
                        if effects_mode == STEP_EFFECTS_VEHICLE:
                            out_el.pop('udl_loaded_ranges', None)
                        out[k] = out_el
                    return out
                return {}

            rA = get_step(res_A, step_idx, veh_key_res, f_A, f_map_A, f_udl_A)
            rB = get_step(res_B, step_idx, veh_key_res, f_B, f_map_B, f_udl_B) if valid_B else {}
    else:
        key_map = {"Total Envelope": "Total Envelope", "Selfweight": "Selfweight", "Soil": "Soil", "Surcharge": "Surcharge", "Vehicle Envelope": "Vehicle Envelope", "Traffic UDL": "Traffic UDL"}
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
                _render_viz_chart("Bending Moment [kNm]", nodes_A, nodes_B, rA, rB, 'M', man_scale, show_A_step, show_B_step, show_labels, view_case, name_A, name_B, res_A, res_B, sysA, sysB, show_supports, support_size)
                _render_viz_chart("Shear Force [kN]", nodes_A, nodes_B, rA, rB, 'V', man_scale, show_A_step, show_B_step, show_labels, view_case, name_A, name_B, res_A, res_B, sysA, sysB, show_supports, support_size)
                _render_viz_chart("Normal Force [kN]", nodes_A, nodes_B, rA, rB, 'N', man_scale, show_A_step, show_B_step, show_labels, view_case, name_A, name_B, res_A, res_B, sysA, sysB, show_supports, support_size)
                _render_viz_chart("Deformation [mm]", nodes_A, nodes_B, rA, rB, 'Def', man_scale, show_A_step, show_B_step, show_labels, view_case, name_A, name_B, res_A, res_B, sysA, sysB, show_supports, support_size)
        else:
            show_A = (show_sys_mode == "Both" or show_sys_mode == "System A")
            show_B = (valid_B and (show_sys_mode == "Both" or show_sys_mode == "System B"))
            
            geom_invalid_A = (nodes_A is None) or (len(nodes_A)==0)
            geom_invalid_B = (valid_B) and ((nodes_B is None) or (len(nodes_B)==0))
            
            if geom_invalid_A and geom_invalid_B: 
                 st.warning("⚠️ No structural geometry defined. Please configure Spans/Walls in the sidebar.")
            else:
                 if (not rA) and (not rB): st.warning(f"⚠️ No results found for **{view_case}**.")

                 _render_viz_chart("Bending Moment [kNm]", nodes_A, nodes_B, rA, rB, 'M', man_scale, show_A, show_B, show_labels, view_case, name_A, name_B, res_A, res_B, sysA, sysB, show_supports, support_size)
                 _render_viz_chart("Shear Force [kN]", nodes_A, nodes_B, rA, rB, 'V', man_scale, show_A, show_B, show_labels, view_case, name_A, name_B, res_A, res_B, sysA, sysB, show_supports, support_size)
                 _render_viz_chart("Normal Force [kN]", nodes_A, nodes_B, rA, rB, 'N', man_scale, show_A, show_B, show_labels, view_case, name_A, name_B, res_A, res_B, sysA, sysB, show_supports, support_size)
                 _render_viz_chart("Deformation [mm]", nodes_A, nodes_B, rA, rB, 'Def', man_scale, show_A, show_B, show_labels, view_case, name_A, name_B, res_A, res_B, sysA, sysB, show_supports, support_size)

    # --- TAB 2: DETAILED DATA ---
    with t2:
        st.markdown(f"### Detailed Data ({view_case})")
        # case_dataframe builds one block per element from whole arrays
        # instead of appending Python dicts per point; on fine meshes this
        # is orders of magnitude faster.
        step_mode = (view_case == "Vehicle Steps")
        detailed_frames = [
            df for df in (
                export_mod.case_dataframe(rA, name_A, step_mode=step_mode),
                export_mod.case_dataframe(rB, name_B, step_mode=step_mode) if valid_B else None,
            ) if df is not None
        ]

        if detailed_frames:
            df_detailed = pd.concat(detailed_frames, ignore_index=True)
            st.dataframe(df_detailed, width='stretch')

            c_dl_csv, c_dl_xlsx = st.columns(2)
            if view_case == "Total Envelope":
                # Full QA package: settings + per-load-case unfactored
                # sheets + Total Envelope per analyzed result mode. Built
                # lazily - the combinations only run on actual download.
                def _full_package():
                    return export_mod.total_envelope_export(
                        sysA, raw_res_A, sysB, raw_res_B, valid_B)

                c_dl_csv.download_button(
                    "Download Detailed Data (.csv)",
                    lambda: export_mod.to_csv_bytes(*_full_package()),
                    "bricos_total_envelope.csv",
                    "text/csv",
                    disabled=ui_locked,
                    help="Analysis settings block plus all load cases and "
                         "envelopes as one long-format table."
                )
                c_dl_xlsx.download_button(
                    "Download Detailed Data (.xlsx)",
                    lambda: export_mod.to_xlsx_bytes(*_full_package()),
                    "bricos_total_envelope.xlsx",
                    export_mod.XLSX_MIME,
                    disabled=ui_locked,
                    help="Settings tab, one unfactored tab per applied load "
                         "case, and a Total Envelope tab per analyzed "
                         "result mode."
                )
            else:
                mode_tags = {"Design (ULS)": "ULS", "Characteristic (SLS)": "SLS"}
                sheet_name = f"{view_case} ({mode_tags.get(result_mode_val, 'Unfactored')})"

                def _view_xlsx():
                    settings = export_mod.settings_dataframe(
                        sysA, sysB, raw_res_A, raw_res_B, valid_B)
                    return export_mod.to_xlsx_bytes(settings, {sheet_name: df_detailed})

                c_dl_csv.download_button(
                    "Download Detailed Data (.csv)",
                    lambda: df_detailed.to_csv(index=False).encode('utf-8'),
                    f"bricos_detailed_{view_case.replace(' ', '_')}.csv",
                    "text/csv",
                    disabled=ui_locked
                )
                c_dl_xlsx.download_button(
                    "Download Detailed Data (.xlsx)",
                    _view_xlsx,
                    f"bricos_detailed_{view_case.replace(' ', '_')}.xlsx",
                    export_mod.XLSX_MIME,
                    disabled=ui_locked,
                    help="The displayed table plus a Settings tab."
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
        restrained_A = (raw_res_A or {}).get('Restrained Nodes')
        restrained_B = (raw_res_B or {}).get('Restrained Nodes')
        reactsA = get_reaction_envelope(rA, nodes_A, sysA['mode'], restrained_A)
        reactsB = get_reaction_envelope(rB, nodes_B, sysB['mode'], restrained_B) if valid_B else {}
        
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

def _render_viz_chart(title, nodes_A, nodes_B, rA, rB, type_base, scale, show_A, show_B, show_labels, view_case, name_A, name_B, res_A, res_B, sysA, sysB, show_supports, support_size):
    st.subheader(title)
    st.plotly_chart(viz.create_plotly_fig(
        nodes_A, rA, rB, type_base, scale, "",
        show_A, show_B, show_labels, view_case,
        name_A, name_B,
        geom_A=res_A.get('Selfweight'), geom_B=res_B.get('Selfweight'),
        params_A=sysA, params_B=sysB,
        show_supports=show_supports, support_size=support_size,
        nodes_A=nodes_A, nodes_B=nodes_B
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
