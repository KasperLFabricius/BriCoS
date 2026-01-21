import streamlit as st
import io
import time
import copy
import os
import pandas as pd 

# --- INTERNAL MODULES ---
import bricos_data as data_mod
import bricos_solver as solver
import bricos_results_ui as results_ui
import bricos_report as report_mod

# ==========================================
# GLOBAL CONFIGURATION
# ==========================================

APP_VERSION = "0.31"
AUTOSAVE_FILE = "latest_session.csv"

st.set_page_config(layout="wide", page_title=f"BriCoS v{APP_VERSION}")

# --- CSS FOR STICKY CONTROLS & LAYOUT ---
st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 3rem;}
    div[data-testid="stExpander"] div[role="button"] p {font-size: 1rem; font-weight: bold;}
    .stSelectbox label { font-size: 0.9rem; font-weight: bold; }
    
    /* Sticky Sidebar Container */
    div[data-testid="stVerticalBlock"]:has(div#sticky-sidebar-marker) {
        position: sticky;
        top: 0rem;
        z-index: 1000;
        background-color: inherit; 
        padding-top: 10px;
        padding-bottom: 15px;
        border-bottom: 2px solid rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    
    /* Sticky Results Toolbar (Main Pane) */
    div[data-testid="stVerticalBlock"]:has(div#sticky-results-marker) {
        position: sticky;
        top: 3.75rem; 
        z-index: 999;
        background-color: white;
        padding: 15px;
        margin-top: 10px;
        margin-bottom: 10px;
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        border: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

# --- LOGO DISPLAY ---
logo_path = data_mod.resource_path("logo.png")
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width='stretch')

st.title(f"BriCoS v{APP_VERSION} - Bridge Comparison Software")

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def trigger_lock(geom_data):
    """Callback to lock a geometry element when modified via Profiler."""
    geom_data['locked'] = True

# ==========================================
# INITIALIZATION & AUTOSAVE
# ==========================================

data_mod.initialize_session_state()

# Global Lock (Report Gen)
ui_locked = st.session_state.is_generating_report

# Autosave Logic
current_time = time.time()
interval_sec = st.session_state.autosave_interval * 60

if st.session_state.autosave_interval > 0 and not ui_locked:
    if (current_time - st.session_state.last_autosave_time) > interval_sec:
        try:
            csv_data = data_mod.generate_csv_data()
            with open(data_mod.get_writable_path(AUTOSAVE_FILE), "wb") as f:
                f.write(csv_data)
            st.session_state.last_autosave_time = current_time
            st.toast("Session Autosaved 💾")
        except Exception:
            pass 

# ==========================================
# SIDEBAR CONTROLS
# ==========================================

# --- 1. ABOUT ---
with st.sidebar.expander("About", expanded=False):
    st.markdown(f"**BriCoS v{APP_VERSION}**")
    st.write("Author: Kasper Lindskov Fabricius")
    st.write("Email: Kasper.LindskovFabricius@sweco.dk")
    st.write("A specialized Finite Element Analysis (FEM) tool for rapid bridge analysis and comparison.")

# --- 2. RESET DATA ---
with st.sidebar.expander("Reset Data", expanded=False):
    if 'reset_mode' not in st.session_state: st.session_state.reset_mode = None
    if 'reset_action' not in st.session_state: st.session_state.reset_action = None 
    
    c_res, c_clr = st.columns(2)
    with c_res:
        st.caption("Restore Defaults")
        if st.button("Restore A", disabled=ui_locked):
            st.session_state.reset_mode, st.session_state.reset_action = "A", "restore"
            st.rerun()
        if st.button("Restore B", disabled=ui_locked):
            st.session_state.reset_mode, st.session_state.reset_action = "B", "restore"
            st.rerun()
        if st.button("Restore All", disabled=ui_locked):
            st.session_state.reset_mode, st.session_state.reset_action = "ALL", "restore"
            st.rerun()

    with c_clr:
        st.caption("Clear Data (Zero)")
        if st.button("Clear A", disabled=ui_locked):
            st.session_state.reset_mode, st.session_state.reset_action = "A", "clear"
            st.rerun()
        if st.button("Clear B", disabled=ui_locked):
            st.session_state.reset_mode, st.session_state.reset_action = "B", "clear"
            st.rerun()
        if st.button("Clear All", disabled=ui_locked):
            st.session_state.reset_mode, st.session_state.reset_action = "ALL", "clear"
            st.rerun()
        
    if st.session_state.reset_mode:
        action_text = "Restore Defaults to" if st.session_state.reset_action == "restore" else "Clear All Data from"
        st.warning(f"⚠️ {action_text} {st.session_state.reset_mode}? Unsaved data will be lost.")
        
        c_yes, c_no = st.columns(2)
        if c_yes.button("Confirm Action", disabled=ui_locked):
            mode = st.session_state.reset_mode
            action = st.session_state.reset_action
            
            def reset_system_state(target_key, new_data):
                clean_data = data_mod.sanitize_input_data(new_data)
                st.session_state[target_key] = clean_data
                data_mod.force_ui_update(target_key, clean_data)

            if mode == "A" or mode == "ALL":
                current_mode = st.session_state['sysA']['mode']
                if action == "clear":
                    data = data_mod.get_clear("A", current_mode)
                else:
                    data = {**data_mod.get_def(), 'num_spans':1, 'name': "System A"}
                    data['soil'] = [
                        {'wall_idx': 0, 'face': 'L', 'h': 8.0, 'q_top': 0.0, 'q_bot': 20.0}, 
                        {'wall_idx': 0, 'face': 'R', 'h': 4.0, 'q_top': 0.0, 'q_bot': 10.0}, 
                        {'wall_idx': 1, 'face': 'L', 'h': 4.0, 'q_top': 0.0, 'q_bot': 10.0}, 
                        {'wall_idx': 1, 'face': 'R', 'h': 8.0, 'q_top': 0.0, 'q_bot': 20.0}
                    ]
                reset_system_state("sysA", data)
                
            if mode == "B" or mode == "ALL":
                current_mode = st.session_state['sysB']['mode']
                if action == "clear":
                    data = data_mod.get_clear("B", current_mode)
                else:
                    data = {**data_mod.get_def(), 'num_spans':2, 'name': "System B"}
                    data['soil'] = [
                        {'wall_idx': 0, 'face': 'L', 'h': 8.0, 'q_top': 0.0, 'q_bot': 20.0},
                        {'wall_idx': 0, 'face': 'R', 'h': 4.0, 'q_top': 0.0, 'q_bot': 10.0},
                        {'wall_idx': 2, 'face': 'L', 'h': 4.0, 'q_top': 0.0, 'q_bot': 10.0},
                        {'wall_idx': 2, 'face': 'R', 'h': 8.0, 'q_top': 0.0, 'q_bot': 20.0}
                    ]
                reset_system_state("sysB", data)
                
            st.session_state.reset_mode = None
            st.session_state.reset_action = None
            st.rerun()
            
        if c_no.button("Cancel"):
            st.session_state.reset_mode = None
            st.session_state.reset_action = None
            st.rerun()

# --- 3. FILE OPERATIONS ---
with st.sidebar.expander("File Operations (Save/Load)", expanded=False):
    # Initialize report keys if missing
    rep_keys = ['rep_pno', 'rep_pname', 'rep_rev', 'rep_author', 'rep_check', 'rep_appr', 'rep_comm']
    for rk in rep_keys:
        if rk not in st.session_state: st.session_state[rk] = ""

    st.download_button("Download Configuration (.csv)", data_mod.generate_csv_data(), "brico_config.csv", "text/csv", disabled=ui_locked)

    uploaded_file = st.file_uploader("Upload Configuration (.csv)", type="csv", key=f"uploader_{st.session_state.uploader_key}", disabled=ui_locked)
    if uploaded_file is not None:
        try:
            df_load = pd.read_csv(uploaded_file)
            if data_mod.load_data_from_df(df_load):
                st.session_state.uploader_key += 1
                st.success("Configuration loaded! UI will update.")
                st.rerun()
            else: st.error("Invalid CSV format.")
        except Exception as e: st.error(f"Error loading file: {e}")
    
    st.markdown("---")
    st.caption("Autosave Settings")
    auto_opts = [0, 2, 5, 10, 30]
    
    def on_autosave_change():
        st.session_state.last_autosave_time = time.time()
    
    curr_idx = 0
    if st.session_state.autosave_interval in auto_opts:
        curr_idx = auto_opts.index(st.session_state.autosave_interval)
        
    new_interval = st.select_slider(
        "Autosave Interval [min]", 
        options=auto_opts, 
        value=auto_opts[curr_idx],
        format_func=lambda x: "Never" if x == 0 else f"{x} min",
        on_change=on_autosave_change,
        disabled=ui_locked,
        help="Note: Autosave is triggered by user interaction (clicks, edits). The app does not save while idle."
    )
    st.session_state.autosave_interval = new_interval

# --- 4. COPY SYSTEM ---
with st.sidebar.expander("Copy Data", expanded=False):
    if 'copy_confirm_mode' not in st.session_state: st.session_state.copy_confirm_mode = None
    c_cp1, c_cp2 = st.columns(2)
    if c_cp1.button("Copy A → B", disabled=ui_locked):
        st.session_state.copy_confirm_mode = "A2B"
        st.rerun()
    if c_cp2.button("Copy B → A", disabled=ui_locked):
        st.session_state.copy_confirm_mode = "B2A"
        st.rerun()

    if st.session_state.copy_confirm_mode == "A2B":
        st.warning("⚠️ Overwrite System B?")
        c_yes, c_no = st.columns(2)
        if c_yes.button("Confirm", disabled=ui_locked):
            nm = st.session_state['sysB']['name']
            st.session_state['sysB'] = copy.deepcopy(st.session_state['sysA'])
            st.session_state['sysB']['name'] = nm
            data_mod.force_ui_update('sysB', st.session_state['sysB'])
            st.session_state.copy_confirm_mode = None
            st.rerun()
        if c_no.button("Cancel"):
            st.session_state.copy_confirm_mode = None
            st.rerun()

    elif st.session_state.copy_confirm_mode == "B2A":
        st.warning("⚠️ Overwrite System A?")
        c_yes, c_no = st.columns(2)
        if c_yes.button("Confirm", disabled=ui_locked):
            nm = st.session_state['sysA']['name']
            st.session_state['sysA'] = copy.deepcopy(st.session_state['sysB'])
            st.session_state['sysA']['name'] = nm
            data_mod.force_ui_update('sysA', st.session_state['sysA'])
            st.session_state.copy_confirm_mode = None
            st.rerun()
        if c_no.button("Cancel"):
            st.session_state.copy_confirm_mode = None
            st.rerun()

# --- 5. ANALYSIS SETTINGS ---
with st.sidebar.expander("Analysis & Result Settings", expanded=False):
    help_dir = "Forward: Left to Right. Reverse: Right to Left (axles inverted). Both: Envelope of both directions."
    curr_dir = st.session_state['sysA'].get('vehicle_direction', 'Forward')
    dir_opts = ["Forward", "Reverse", "Both"]
    idx_dir = dir_opts.index(curr_dir) if curr_dir in dir_opts else 0
    
    dir_sel = st.radio("Vehicle Direction", dir_opts, horizontal=True, index=idx_dir, key="veh_dir_radio_sidebar", help=help_dir, disabled=ui_locked)
    st.session_state['sysA']['vehicle_direction'] = dir_sel
    st.session_state['sysB']['vehicle_direction'] = dir_sel
    
    st.markdown("---")
    help_combo = "Define how the Traffic Surcharge (on walls) and the Main Vehicle (on deck) interact.\n- Exclusive: Load is max(Vehicle, Surcharge).\n- Simultaneous: Load is Vehicle + Surcharge."
    is_sim = st.session_state['sysA'].get('combine_surcharge_vehicle', False)
    combo_idx = 1 if is_sim else 0
    
    surch_sel = st.radio("Surcharge Combination", ["Exclusive (Vehicle OR Surcharge)", "Simultaneous (Vehicle + Surcharge)"], index=combo_idx, horizontal=True, key="surcharge_combo_radio_sidebar", help=help_combo, disabled=ui_locked)
    is_simultaneous = (surch_sel == "Simultaneous (Vehicle + Surcharge)")
    st.session_state['sysA']['combine_surcharge_vehicle'] = is_simultaneous
    st.session_state['sysB']['combine_surcharge_vehicle'] = is_simultaneous

    st.markdown("---")
    st.markdown("**Shear Deformations (Timoshenko)**")
    
    help_shear = "Enables shear deformation consideration in the stiffness matrix. Recommended for deep beams and piers."
    use_shear = st.checkbox("Enable Shear Deformations", value=st.session_state['sysA'].get('use_shear_def', False), key="shear_toggle_sidebar", help=help_shear, disabled=ui_locked)
    st.session_state['sysA']['use_shear_def'] = use_shear
    st.session_state['sysB']['use_shear_def'] = use_shear

    col_beff, col_nu = st.columns(2)
    val_beff = st.session_state['sysA'].get('b_eff', 1.0)
    val_nu = st.session_state['sysA'].get('nu', 0.2)
    
    new_beff = col_beff.number_input(r"$b_{eff}$ [m]", value=float(val_beff), min_value=0.01, step=0.1, help="Effective shear width.", key="beff_input_sidebar", disabled=ui_locked)
    new_nu = col_nu.number_input(r"Poisson's Ratio ($\nu$)", value=float(val_nu), min_value=0.0, max_value=0.5, step=0.05, key="nu_input_sidebar", disabled=ui_locked)
    
    st.session_state['sysA']['b_eff'] = new_beff; st.session_state['sysB']['b_eff'] = new_beff
    st.session_state['sysA']['nu'] = new_nu; st.session_state['sysB']['nu'] = new_nu

    st.markdown("---")
    st.markdown("**Calculation Precision**")
    c_mesh, c_step = st.columns(2)
    def_mesh = st.session_state['sysA'].get('mesh_size', 0.5)
    def_step = st.session_state['sysA'].get('step_size', 0.2)
    m_val = c_mesh.slider("Mesh Size [m]", 0.1, 5.0, def_mesh, 0.1, key="common_mesh_slider", disabled=ui_locked)
    s_val = c_step.slider("Vehicle Step [m]", 0.01, 2.0, def_step, 0.01, key="common_step_slider", disabled=ui_locked)

if "common_mesh_slider" in st.session_state:
    st.session_state['sysA']['mesh_size'] = m_val
    st.session_state['sysB']['mesh_size'] = m_val
if "common_step_slider" in st.session_state:
    st.session_state['sysA']['step_size'] = s_val
    st.session_state['sysB']['step_size'] = s_val

# --- 6. REPORT GENERATION ---
with st.sidebar.expander("Report Generation", expanded=False):
    # Ensure keys exist
    if 'rep_pno' not in st.session_state: st.session_state.rep_pno = ""
    # ... (Other keys initialized in data_mod)
    
    st.text_input("Project No.", key="rep_pno", disabled=ui_locked)
    st.text_input("Project Name", key="rep_pname", disabled=ui_locked)
    
    c_r1, c_r2 = st.columns(2)
    c_r1.text_input("Revision", key="rep_rev", disabled=ui_locked)
    c_r2.text_input("Author", key="rep_author", disabled=ui_locked)
    
    c_r3, c_r4 = st.columns(2)
    c_r3.text_input("Checker", key="rep_check", disabled=ui_locked)
    c_r4.text_input("Approver", key="rep_appr", disabled=ui_locked)
    
    st.text_area("Comments", height=100, key="rep_comm", disabled=ui_locked)
    
    prog_bar = st.empty()
    
    if st.button("Generate PDF Report", type="primary", disabled=ui_locked):
        st.session_state.is_generating_report = True
        st.rerun()

    if 'report_buffer' in st.session_state:
        st.download_button("Download Report PDF", st.session_state['report_buffer'], f"BriCoS_Report_{st.session_state.rep_pno}.pdf", "application/pdf")

# ==========================================
# STICKY SIDEBAR: ACTIVE SYSTEM
# ==========================================
st.sidebar.header("Configuration")
with st.sidebar.container():
    st.markdown('<div id="sticky-sidebar-marker"></div>', unsafe_allow_html=True)
    c_nA, c_nB = st.columns(2)
    st.session_state['sysA']['name'] = c_nA.text_input("Name Sys A", st.session_state['sysA']['name'], disabled=ui_locked)
    st.session_state['sysB']['name'] = c_nB.text_input("Name Sys B", st.session_state['sysB']['name'], disabled=ui_locked)

    sys_map = {"sysA": f"{st.session_state['sysA']['name']} (Blue)", "sysB": f"{st.session_state['sysB']['name']} (Red)"}
    # FIXED: Added persistent key to prevent reset during report generation
    active_sys_key = st.radio("Active System:", ["sysA", "sysB"], format_func=lambda x: sys_map[x], horizontal=True, disabled=ui_locked, key="active_system_radio_sidebar")

    if active_sys_key == 'sysA':
        st.markdown("""<style>[data-testid="stSidebar"] { background-color: #F0F8FF; }</style>""", unsafe_allow_html=True)
    else:
        st.markdown("""<style>[data-testid="stSidebar"] { background-color: #FFF5F5; }</style>""", unsafe_allow_html=True)

curr = active_sys_key
p = st.session_state[curr]

# ==========================================
# SYSTEM INPUTS (FACTORS, GEOMETRY, LOADS)
# ==========================================

with st.sidebar.expander("Design Factors & Type", expanded=False):
    help_mode = "Choose 'Frame' for full interaction (Walls + Slab) or 'Superstructure' for a simplified slab-on-supports analysis."
    new_mode_sel = st.selectbox("Model Type", ["Frame", "Superstructure"], index=0 if p['mode']=='Frame' else 1, key=f"{curr}_md_sel", help=help_mode, disabled=ui_locked)
    
    old_mode = p.get('last_mode', 'Frame')
    if old_mode != new_mode_sel:
        # Handle Mode Switching Logic
        if new_mode_sel == 'Superstructure':
            st.session_state[curr]['backup'] = {
                'h_list': copy.deepcopy(p['h_list']),
                'Iw_list': copy.deepcopy(p['Iw_list']),
                'soil': copy.deepcopy(p['soil']),
                'surcharge': copy.deepcopy(p['surcharge'])
            }
            p['h_list'] = [0.0] * len(p['h_list'])
            p['Iw_list'] = [0.0] * len(p['Iw_list'])
            p['soil'] = []
            p['surcharge'] = []
        elif new_mode_sel == 'Frame':
            b = st.session_state[curr].get('backup', {})
            if b:
                 p['h_list'] = b.get('h_list', p['h_list'])
                 p['Iw_list'] = b.get('Iw_list', p['Iw_list'])
                 p['soil'] = b.get('soil', p['soil'])
                 p['surcharge'] = b.get('surcharge', p['surcharge'])
        p['mode'] = new_mode_sel
        p['last_mode'] = new_mode_sel
        st.rerun()
    
    help_mat = "Choose method for Elastic Modulus (E) definition."
    e_mode = st.radio("Material Definition", ["Eurocode (f_ck)", "Manual (E-Modulus)"], horizontal=True, index=0 if p['e_mode']=='Eurocode' else 1, key=f"{curr}_emode", help=help_mat, disabled=ui_locked)
    p['e_mode'] = "Eurocode" if "Eurocode" in e_mode else "Manual"

    kfi_opts = [0.9, 1.0, 1.1]
    curr_kfi = p.get('KFI', 1.0)
    idx_kfi = kfi_opts.index(curr_kfi) if curr_kfi in kfi_opts else 1
    help_KFI = "Partial factor for consequence class. Applied to all loads."
    p['KFI'] = st.selectbox("KFI (Consequence Class)", kfi_opts, index=idx_kfi, key=f"{curr}_kfi", disabled=ui_locked, help=help_KFI)
    
    gg_opts = [0.9, 1.0, 1.10, 1.25]
    c_gg, c_gj = st.columns(2)
    gg_val = p.get('gamma_g', 1.0)
    idx_gg = gg_opts.index(gg_val) if gg_val in gg_opts else len(gg_opts)
    
    help_gg = "Partial factor for permanent loads (Self-weight). Applied to the 'Selfweight' load case."
    gg_sel = c_gg.selectbox(r"$\gamma_{g}$ (Self-weight)", gg_opts + ["Custom"], index=min(idx_gg, len(gg_opts)), key=f"{curr}_gg_sel", disabled=ui_locked, help=help_gg)
    if gg_sel == "Custom": p['gamma_g'] = c_gg.number_input(r"Custom $\gamma_{g}$", value=float(gg_val), key=f"{curr}_gg_cust", disabled=ui_locked)
    else: p['gamma_g'] = float(gg_sel)

    gj_opts = [1.0, 1.1]
    gj_val = p.get('gamma_j', 1.0)
    idx_gj = gj_opts.index(gj_val) if gj_val in gj_opts else len(gj_opts)
    
    help_gj = "Partial factor for permanent soil loads (Earth Pressure). Applied to the 'Soil' load case."
    gj_sel = c_gj.selectbox(r"$\gamma_{j}$ (Soil)", gj_opts + ["Custom"], index=min(idx_gj, len(gj_opts)), key=f"{curr}_gj_sel", disabled=ui_locked, help=help_gj)
    if gj_sel == "Custom": p['gamma_j'] = c_gj.number_input(r"Custom $\gamma_{j}$", value=float(gj_val), key=f"{curr}_gj_cust", disabled=ui_locked)
    else: p['gamma_j'] = float(gj_sel)

    gam_opts = [0.56, 1.0, 1.05, 1.20, 1.25, 1.40]
    c_ga, c_gb = st.columns(2)
    gam_valA = p.get('gamma_veh', 1.0)
    idx_gamA = gam_opts.index(gam_valA) if gam_valA in gam_opts else len(gam_opts)
    
    help_ga = "Partial factor for variable traffic Load Model A. Applied to 'Vehicle A' (with Dynamic Factor) and 'Surcharge' (static)."
    gam_selA = c_ga.selectbox(r"$\gamma_{veh,A}$", gam_opts + ["Custom"], index=min(idx_gamA, len(gam_opts)), key=f"{curr}_gamA_sel", disabled=ui_locked, help=help_ga)
    if gam_selA == "Custom": p['gamma_veh'] = c_ga.number_input(r"Custom $\gamma_{A}$", value=float(gam_valA), key=f"{curr}_gamA_cust", disabled=ui_locked)
    else: p['gamma_veh'] = float(gam_selA)

    gam_valB = p.get('gamma_vehB', 1.0)
    idx_gamB = gam_opts.index(gam_valB) if gam_valB in gam_opts else len(gam_opts)
    
    help_gb = "Partial factor for variable traffic Load Model B. Applied to 'Vehicle B' (with Dynamic Factor)."
    gam_selB = c_gb.selectbox(r"$\gamma_{veh,B}$", gam_opts + ["Custom"], index=min(idx_gamB, len(gam_opts)), key=f"{curr}_gamB_sel", disabled=ui_locked, help=help_gb)
    if gam_selB == "Custom": p['gamma_vehB'] = c_gb.number_input(r"Custom $\gamma_{B}$", value=float(gam_valB), key=f"{curr}_gamB_cust", disabled=ui_locked)
    else: p['gamma_vehB'] = float(gam_selB)

    phi_mode = st.radio("Dynamic Factor (Phi)", ["Calculate", "Manual"], horizontal=True, index=0 if p.get('phi_mode', 'Calculate') == 'Calculate' else 1, key=f"{curr}_phim", disabled=ui_locked)
    p['phi_mode'] = phi_mode
    if phi_mode == "Manual":
        p['phi'] = st.number_input("Phi Value", value=p.get('phi', 1.0), key=f"{curr}_phiv", disabled=ui_locked)
    
    phi_log_placeholder = st.empty()

with st.sidebar.expander("Geometry, Stiffness & Static Loads", expanded=False):
    n_spans = st.number_input("Number of Spans", 1, 10, p['num_spans'], key=f"{curr}_nsp", disabled=ui_locked)
    p['num_spans'] = n_spans
    
    is_ec = (p['e_mode'] == 'Eurocode')
    lbl_mat = r"$f_{ck}$ [MPa]" if is_ec else r"$E$ [GPa]"
    
    st.markdown("---")
    st.markdown("**Spans (L, H, SW, Material)**")
    
    # Input Loop (Spans)
    for i in range(n_spans):
        # Tooltip Help Strings (Only show on first iteration)
        help_L = "Span length [m]" if i == 0 else None
        help_H = "Section Height/Depth [m]. Used to calculate stiffness I." if i == 0 else None
        help_SW = "Load from selfweight and other permanent loads, such as soil and surfacing [kN/m]." if i == 0 else None
        help_Mat = "Characteristic concrete cylinder strength [MPa]" if (i==0 and is_ec) else ("Young's Modulus [GPa]" if i==0 else None)

        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        p['L_list'][i] = c1.number_input(f"L{i+1} [m]", value=float(p['L_list'][i]), key=f"{curr}_l{i}", disabled=ui_locked, help=help_L)
        
        # Check if profiler data exists (Advanced Config Check)
        key = f"span_geom_{i}"
        if key not in p: p[key] = {'type': 1, 'shape': 0, 'vals': [p['Is_list'][i]]*3, 'locked': False}
        s_geom = p[key]
        
        # Locked if user marked it 'locked' via Profiler OR if complex config
        is_adv = (s_geom.get('locked', False)) or (s_geom['shape'] != 0) or (s_geom['type'] != 1) or (s_geom.get('align_type', 0) != 0)
        
        if not is_adv:
            val = c2.number_input(f"H{i+1} [m]", value=float(p['Is_list'][i]), format="%.3f", key=f"{curr}_i{i}", disabled=ui_locked, help=help_H)
            p['Is_list'][i] = val
            s_geom['vals'] = [val, val, val]
        else:
            c2.text_input(f"H{i+1} [m]", "See Profiler", disabled=True, key=f"{curr}_i{i}_dis", help="Controlled by Section Profiler")

        p['sw_list'][i] = c3.number_input(f"SW{i+1} [kN/m]", value=float(p['sw_list'][i]), key=f"{curr}_s{i}", disabled=ui_locked, help=help_SW)
        
        if is_ec:
            val_in = c4.number_input(f"{lbl_mat}", value=float(p['fck_span_list'][i]), key=f"{curr}_fck_s{i}", disabled=ui_locked, help=help_Mat)
            p['fck_span_list'][i] = val_in
            E_gpa = 22.0 * ((val_in + 8)/10.0)**0.3
            p['E_span_list'][i] = E_gpa * 1e6
        else:
            val_in = c4.number_input(f"{lbl_mat}", value=float(p['E_custom_span'][i]), key=f"{curr}_Eman_s{i}", disabled=ui_locked, help=help_Mat)
            p['E_custom_span'][i] = val_in
            p['E_span_list'][i] = val_in * 1e6
        
    is_super = (p['mode'] == 'Superstructure')
    st.markdown("---")
    st.markdown("**Walls (H_wall, H_sect, Surcharge, Material)**")
    
    # Input Loop (Walls)
    for i in range(n_spans + 1):
        # Tooltip Help Strings (Only show on first iteration)
        help_Hw = "Vertical height of the wall [m]" if i == 0 else None
        help_Hs = "Wall Section Thickness/Height [m]" if i == 0 else None
        help_Surch = "The horizontal load resulting from vehicle surcharge, placed over the full height of the wall. Dynamic Factors are not applied to this load, but the partial coefficient for vehicle A is applied in ULS." if i == 0 else None
        help_Mat = "Characteristic concrete cylinder strength [MPa]" if (i==0 and is_ec) else ("Young's Modulus [GPa]" if i==0 else None)

        st.caption(f"Wall {i+1}")
        c1, c2, c3, c4 = st.columns([1,1,1,1])
        p['h_list'][i] = c1.number_input(f"H_wall [m]", value=float(p['h_list'][i]), disabled=(is_super or ui_locked), key=f"{curr}_h{i}", help=help_Hw)
        
        key = f"wall_geom_{i}"
        if key not in p: p[key] = {'type': 1, 'shape': 0, 'vals': [p['Iw_list'][i]]*3, 'locked': False}
        w_geom = p[key]
        
        is_adv_w = (w_geom.get('locked', False)) or (w_geom['shape'] != 0) or (w_geom['type'] != 1)

        if not is_adv_w:
            val_w = c2.number_input(r"H_sect [m]", value=float(p['Iw_list'][i]), format="%.3f", disabled=(is_super or ui_locked), key=f"{curr}_iw{i}", help=help_Hs)
            p['Iw_list'][i] = val_w
            w_geom['vals'] = [val_w, val_w, val_w]
        else:
            c2.text_input(f"H_sect", "See Profiler", disabled=True, key=f"{curr}_iw{i}_dis", help="Controlled by Section Profiler")
        
        sur = next((x for x in p['surcharge'] if x['wall_idx']==i), None)
        val_q = sur['q'] if sur else 0.0
        new_q = c3.number_input(f"Surcharge [kN/m]", value=float(val_q), disabled=(is_super or ui_locked), key=f"{curr}_sq{i}", help=help_Surch)
        
        if not is_super:
            p['surcharge'] = [x for x in p['surcharge'] if x['wall_idx'] != i]
            if new_q != 0: 
                p['surcharge'].append({'wall_idx':i, 'face':'R', 'q':new_q, 'h':p['h_list'][i]})

        if is_ec:
            val_in = c4.number_input(f"{lbl_mat}", value=float(p['fck_wall_list'][i]), disabled=(is_super or ui_locked), key=f"{curr}_fck_w{i}", help=help_Mat)
            p['fck_wall_list'][i] = val_in
            E_gpa = 22.0 * ((val_in + 8)/10.0)**0.3
            p['E_wall_list'][i] = E_gpa * 1e6
        else:
            val_in = c4.number_input(f"{lbl_mat}", value=float(p['E_custom_wall'][i]), disabled=(is_super or ui_locked), key=f"{curr}_Eman_w{i}", help=help_Mat)
            p['E_custom_wall'][i] = val_in
            p['E_wall_list'][i] = val_in * 1e6

        ex_SoilLeft = next((x for x in p['soil'] if x['wall_idx']==i and x['face']=='L'), None)
        ex_SoilRight = next((x for x in p['soil'] if x['wall_idx']==i and x['face']=='R'), None)
        
        c_sl, c_sr = st.columns(2)
        # Help for Soil
        help_Hs = "Height of soil layer [m]" if i == 0 else None
        help_qb = "Earth pressure at bottom of layer [kN/m]" if i == 0 else None
        help_qt = "Earth pressure at top of layer [kN/m]" if i == 0 else None

        h_L = c_sl.number_input("H_soil_left [m]", value=ex_SoilLeft['h'] if ex_SoilLeft else 0.0, disabled=(is_super or ui_locked), key=f"{curr}_shl{i}", help=help_Hs)
        qL_bot = c_sl.number_input(r"q_bot [kN/m]", value=ex_SoilLeft['q_bot'] if ex_SoilLeft else 0.0, disabled=(is_super or ui_locked), key=f"{curr}_sqlb{i}", help=help_qb)
        qL_top = c_sl.number_input(r"q_top [kN/m]", value=ex_SoilLeft['q_top'] if ex_SoilLeft else 0.0, disabled=(is_super or ui_locked), key=f"{curr}_sqlt{i}", help=help_qt)
        
        h_R = c_sr.number_input("H_soil_right [m]", value=ex_SoilRight['h'] if ex_SoilRight else 0.0, disabled=(is_super or ui_locked), key=f"{curr}_shr{i}", help=help_Hs)
        qR_bot = c_sr.number_input(r"q_bot [kN/m]", value=ex_SoilRight['q_bot'] if ex_SoilRight else 0.0, disabled=(is_super or ui_locked), key=f"{curr}_sqrb{i}", help=help_qb)
        qR_top = c_sr.number_input(r"q_top [kN/m]", value=ex_SoilRight['q_top'] if ex_SoilRight else 0.0, disabled=(is_super or ui_locked), key=f"{curr}_sqrt{i}", help=help_qt)

        if not is_super:
            p['soil'] = [x for x in p['soil'] if x['wall_idx']!=i]
            if h_L > 0: p['soil'].append({'wall_idx':i, 'face':'L', 'q_bot':qL_bot, 'q_top':qL_top, 'h':h_L})
            if h_R > 0: p['soil'].append({'wall_idx':i, 'face':'R', 'q_bot':qR_bot, 'q_top':qR_top, 'h':h_R})

    st.markdown("---")
    with st.sidebar.expander("🛠️ Section Profiler (Advanced)", expanded=False):
        st.caption("Configure variable stiffness, height profiles, or vertical alignment.")
        
        elem_options = [f"Span {i+1}" for i in range(n_spans)] + ([f"Wall {i+1}" for i in range(n_spans+1)] if not is_super else [])
        sel_el = st.selectbox("Edit Element:", elem_options, key=f"{curr}_prof_sel", disabled=ui_locked)
        
        is_span_selected = "Span" in sel_el
        idx = int(sel_el.split(" ")[1]) - 1
        
        if is_span_selected:
            target_geom = p[f"span_geom_{idx}"]
            target_simple_list = p['Is_list']
        else:
            target_geom = p[f"wall_geom_{idx}"]
            target_simple_list = p['Iw_list']
        
        # --- LOCK MANAGEMENT ---
        is_currently_locked = target_geom.get('locked', False)
        is_simple_shape = (target_geom['shape'] == 0) and (target_geom['type'] == 1) and (target_geom.get('align_type', 0) == 0)

        # Sanity Check: If it says locked, but config is simple, show the Reset button (Logic below).
        # We removed the auto-lock 'else' block here.
        
        # UI Control for Locking
        c_lock1, c_lock2 = st.columns([3, 1])
        if is_currently_locked:
            c_lock1.warning("⚠️ Simple Input Locked")
            if c_lock2.button("Reset", key=f"{curr}_unlock_{sel_el}", help="Reverts to Simple Input (unlocks field above)"):
                target_geom['locked'] = False
                target_geom['shape'] = 0
                target_geom['type'] = 1
                target_geom['align_type'] = 0
                target_geom['incline_mode'] = 0
                target_geom['incline_val'] = 0.0
                st.rerun()
        
        # NOTE: We attach 'trigger_lock' to on_change events below to catch explicit edits.

        c_p1, c_p2 = st.columns(2)
        new_type = c_p1.radio(
            "Definition Mode:", ["Inertia (I)", "Height (H)"], 
            index=target_geom['type'], 
            key=f"{curr}_prof_type_{sel_el}", horizontal=True, disabled=ui_locked,
            on_change=trigger_lock, args=(target_geom,)
        )
        target_geom['type'] = 0 if "Inertia" in new_type else 1
        
        new_shape = c_p2.radio(
            "Profile Shape:", ["Constant", "Linear (Taper)", "3-Point (Start/Mid/End)"], 
            index=target_geom['shape'], 
            key=f"{curr}_prof_shape_{sel_el}", horizontal=True, disabled=ui_locked,
            on_change=trigger_lock, args=(target_geom,)
        )
        shape_map = {"Constant": 0, "Linear (Taper)": 1, "3-Point (Start/Mid/End)": 2}
        target_geom['shape'] = shape_map[new_shape]
        
        vals = target_geom['vals']
        c_v1, c_v2, c_v3 = st.columns(3)
        lbl_v = r"I [$\text{m}^4$]" if target_geom['type']==0 else "H [m]"
        
        # SYNC: If unlocked and simple, ensure we see the simple input values
        if not is_currently_locked and is_simple_shape:
             st.session_state[f"{curr}_prof_v1_{sel_el}"] = vals[0]
             st.session_state[f"{curr}_prof_v2_{sel_el}"] = vals[1]
             st.session_state[f"{curr}_prof_v3_{sel_el}"] = vals[2]

        v1 = c_v1.number_input(
            f"Start {lbl_v}", value=float(vals[0]), format="%.4f", 
            key=f"{curr}_prof_v1_{sel_el}", disabled=ui_locked,
            on_change=trigger_lock, args=(target_geom,)
        )
        v2 = vals[1]
        if target_geom['shape'] == 2:
            v2 = c_v2.number_input(
                f"Mid {lbl_v}", value=float(vals[1]), format="%.4f", 
                key=f"{curr}_prof_v2_{sel_el}", disabled=ui_locked,
                on_change=trigger_lock, args=(target_geom,)
            )
        v3 = vals[2]
        if target_geom['shape'] >= 1:
            v3 = c_v3.number_input(
                f"End {lbl_v}", value=float(vals[2]), format="%.4f", 
                key=f"{curr}_prof_v3_{sel_el}", disabled=ui_locked,
                on_change=trigger_lock, args=(target_geom,)
            )
            
        target_geom['vals'] = [v1, v2, v3]
        
        # If in simple mode (just height constant), sync back to simple list for legacy logic
        if target_geom['type'] == 1:
            target_simple_list[idx] = v1

        if is_span_selected:
            st.markdown("#### 📐 Alignment (Vertical Geometry)")
            if 'align_type' not in target_geom: target_geom['align_type'] = 0
            if 'incline_mode' not in target_geom: target_geom['incline_mode'] = 0
            if 'incline_val' not in target_geom: target_geom['incline_val'] = 0.0

            al_opts = ["Straight (Horizontal)", "Inclined"]
            new_align = st.radio(
                "Span Profile:", al_opts, index=target_geom['align_type'], horizontal=True, 
                key=f"{curr}_align_t_{sel_el}", disabled=ui_locked,
                on_change=trigger_lock, args=(target_geom,)
            )
            target_geom['align_type'] = al_opts.index(new_align)
            
            if target_geom['align_type'] == 1:
                inc_opts = ["Slope (%)", "Delta Height (End - Start) [m]"]
                new_inc_mode = st.radio(
                    "Define Inclination by:", inc_opts, index=target_geom['incline_mode'], horizontal=True, 
                    key=f"{curr}_inc_m_{sel_el}", disabled=ui_locked,
                    on_change=trigger_lock, args=(target_geom,)
                )
                target_geom['incline_mode'] = inc_opts.index(new_inc_mode)
                
                lbl_inc = "Slope [%]" if target_geom['incline_mode'] == 0 else "Delta H [m]"
                target_geom['incline_val'] = st.number_input(
                    lbl_inc, value=float(target_geom['incline_val']), format="%.2f", 
                    key=f"{curr}_inc_v_{sel_el}", disabled=ui_locked,
                    on_change=trigger_lock, args=(target_geom,)
                )

# --- BOUNDARY CONDITIONS TAB ---
with st.sidebar.expander("Boundary Conditions", expanded=False):
    num_supports = n_spans + 1
    current_supports = p.get('supports', [])
    
    if len(current_supports) != num_supports:
        new_list = []
        for i in range(num_supports):
            if i < len(current_supports):
                new_list.append(current_supports[i])
            else:
                if p['mode'] == 'Frame':
                    new_list.append({'type': 'Fixed', 'k': [1e14, 1e14, 1e14]})
                else:
                    if i == 0: new_list.append({'type': 'Pinned', 'k': [1e14, 1e14, 0.0]})
                    else: new_list.append({'type': 'Roller (X-Free)', 'k': [0.0, 1e14, 0.0]})
        p['supports'] = new_list
    
    presets = {
        "Fixed": [1e14, 1e14, 1e14], "Pinned": [1e14, 1e14, 0.0],
        "Roller (X-Free)": [0.0, 1e14, 0.0], "Roller (Y-Free)": [1e14, 0.0, 0.0],
        "Custom Spring": None
    }
    
    for i in range(num_supports):
        supp_name = f"Wall {i+1} Base" if p['mode'] == 'Frame' else f"Support {i+1}"
        st.markdown(f"**{supp_name}**")
        curr_s = p['supports'][i]
        curr_type = curr_s.get('type', 'Fixed')
        if curr_type not in presets: curr_type = 'Custom Spring'
        
        sel_type = st.selectbox(f"Type {i+1}", list(presets.keys()), index=list(presets.keys()).index(curr_type), key=f"{curr}_supp_t_{i}", label_visibility="collapsed", disabled=ui_locked)
        
        new_k = curr_s['k']
        if sel_type != "Custom Spring":
            new_k = presets[sel_type]
            p['supports'][i]['type'] = sel_type
            p['supports'][i]['k'] = new_k
        else:
            p['supports'][i]['type'] = "Custom Spring"
            col_k1, col_k2, col_k3 = st.columns(3)
            kx = col_k1.number_input(f"Kx", value=float(curr_s['k'][0]), format="%.1e", key=f"{curr}_kx_{i}", disabled=ui_locked)
            ky = col_k2.number_input(f"Ky", value=float(curr_s['k'][1]), format="%.1e", key=f"{curr}_ky_{i}", disabled=ui_locked)
            km = col_k3.number_input(f"Km", value=float(curr_s['k'][2]), format="%.1e", key=f"{curr}_km_{i}", disabled=ui_locked)
            p['supports'][i]['k'] = [kx, ky, km]

# --- VEHICLES ---
with st.sidebar.expander("Vehicle Definitions", expanded=False):
    # Retrieve vehicle library via Data Module
    veh_options, veh_data = data_mod.get_vehicle_library()
    veh_help_txt = "Standard LM3 vehicles are defined in accordance with DS/EN 1991-2, DK:NA (bridges):2017."
    
    def handle_veh_inputs(prefix, key_class, key_loads, key_space, struct_key):
        sess_key = f"{curr}_{prefix}_class"
        
        # FIXED: Robust initialization using identifying logic if session key missing
        if sess_key not in st.session_state:
            curr_loads = p[struct_key].get('loads', [])
            curr_space = p[struct_key].get('spacing', [])
            st.session_state[sess_key] = data_mod.identify_vehicle_class(curr_loads, curr_space)
        
        sel_class = st.selectbox(f"Class {prefix}", veh_options, key=sess_key, disabled=ui_locked, help=veh_help_txt)
        input_key_l = f"{curr}_{prefix}_loads_input"
        input_key_s = f"{curr}_{prefix}_space_input"
        if input_key_l not in st.session_state: st.session_state[input_key_l] = p[key_loads]
        if input_key_s not in st.session_state: st.session_state[input_key_s] = p[key_space]

        last_key = f"{key_class}_last"
        if last_key not in st.session_state: st.session_state[last_key] = sel_class
        
        if sel_class != st.session_state[last_key]:
            st.session_state[last_key] = sel_class
            if sel_class == "Custom":
                p[key_loads] = ""; p[key_space] = ""; p[struct_key]['loads'] = []; p[struct_key]['spacing'] = []
                st.session_state[input_key_l] = ""; st.session_state[input_key_s] = ""
                st.rerun()
            elif sel_class in veh_data:
                p[key_loads] = veh_data[sel_class]['loads']; p[key_space] = veh_data[sel_class]['spacing']
                st.session_state[input_key_l] = p[key_loads]; st.session_state[input_key_s] = p[key_space] 
                st.rerun()
        
        # --- TOOLTIP CONFIGURATION ---
        help_loads = "Define axle loads in tonnes [t], separated by commas. Example: '10, 10, 15'"
        help_space = "Define spacing between axles in meters [m]. Must start with 0. The list length must equal the number of loads. Example: '0, 1.5, 3.0'"
        
        p[key_loads] = st.text_input(f"Loads {prefix} [t]", key=input_key_l, disabled=ui_locked, help=help_loads)
        p[key_space] = st.text_input(f"Space {prefix} [m]", key=input_key_s, disabled=ui_locked, help=help_space)
        
        valid_veh = False
        err_msg = ""
        
        try:
            if p[key_loads].strip():
                try:
                    l_arr = [float(x) for x in p[key_loads].split(',') if x.strip()]
                except ValueError:
                    l_arr = []
                    err_msg = "Load input contains non-numeric values."

                try:
                    s_arr = [float(x) for x in p[key_space].split(',') if x.strip()]
                except ValueError:
                    s_arr = []
                    err_msg = "Spacing input contains non-numeric values."
                
                if not err_msg:
                    if len(l_arr) != len(s_arr):
                        err_msg = f"Mismatch: {len(l_arr)} loads vs {len(s_arr)} spacings."
                    elif len(s_arr) > 0 and s_arr[0] != 0:
                        err_msg = "Spacing must start with 0."
                    elif len(l_arr) == 0:
                        err_msg = "Vehicle definition empty."
                    else:
                        valid_veh = True
                        p[struct_key]['loads'] = l_arr; p[struct_key]['spacing'] = s_arr
            else:
                p[struct_key]['loads'] = []; p[struct_key]['spacing'] = []
        except Exception as e:
            err_msg = f"Parsing Error: {str(e)}"
        
        if not valid_veh:
            if p[key_loads].strip(): 
                st.error(f"Invalid Vehicle {prefix}: {err_msg}")
                p[struct_key]['loads'] = [] # Ensure invalid data doesn't propagate to solver
            else: 
                st.caption("No vehicle defined.")
        else: 
            st.success(f"Vehicle {prefix} Valid")

    st.markdown("**Vehicle A**")
    handle_veh_inputs("A", f"{curr}_vehA_class", 'vehicle_loads', 'vehicle_space', 'vehicle')
    st.markdown("---")
    st.markdown("**Vehicle B**")
    handle_veh_inputs("B", f"{curr}_vehB_class", 'vehicleB_loads', 'vehicleB_space', 'vehicleB')

# ==========================================
# SOLVER EXECUTION
# ==========================================

def safe_solve(params):
    try:
        # Returns: results, nodes, model_props, error_flag
        return solver.run_raw_analysis(params)
    except ValueError as e:
        return None, None, None, str(e)

raw_res_A, nodes_A, props_A, err_A = safe_solve(st.session_state['sysA'])
raw_res_B, nodes_B, props_B, err_B = safe_solve(st.session_state['sysB'])

# Persist Model Props for Report Generator
st.session_state['model_props_A'] = props_A
st.session_state['model_props_B'] = props_B

if err_A and isinstance(err_A, str): st.error(f"System A Error: {err_A}")
if err_B and isinstance(err_B, str): st.error(f"System B Error: {err_B}")

if p.get('phi_mode') == 'Calculate' and raw_res_A and raw_res_B:
    active_raw_res = raw_res_A if curr == 'sysA' else raw_res_B
    phi_val = active_raw_res.get('phi_calc', 1.0)
    with phi_log_placeholder.container():
        st.markdown(f"**Calculated Phi:** {phi_val:.3f}")
        with st.expander("Phi Calculation Log", expanded=False):
            st.markdown("The determinant length of the static system is calculated in accordance with DS/EN 1991-2:2023, Table 8.2 (NDP). The dynamic factor is then calculated in accordance with DS/EN 1991-2 DK/NA Bridges:2017, A.2.3.5 based on the determinant length.")
            for log_line in active_raw_res.get('phi_log', []): st.caption(log_line)

# ==========================================
# REPORT GENERATION TRIGGER
# ==========================================

if st.session_state.is_generating_report:
    buffer = io.BytesIO()
    meta = {
        'proj_no': st.session_state.rep_pno,
        'proj_name': st.session_state.rep_pname,
        'rev': st.session_state.rep_rev,
        'author': st.session_state.rep_author,
        'checker': st.session_state.rep_check,
        'approver': st.session_state.rep_appr,
        'comments': st.session_state.rep_comm
    }
    
    current_prog = prog_bar.progress(0, text="Initializing Report...")
    def update_progress(p):
        val = max(0.0, min(1.0, float(p)))
        current_prog.progress(val, text=f"Rendering Plots: {int(val*100)}%")

    try:
        rep_gen = report_mod.BricosReportGenerator(
            buffer, meta, st.session_state,
            raw_res_A, raw_res_B, nodes_A, nodes_B,
            version=APP_VERSION,
            progress_callback=update_progress
        )
        rep_gen.generate()
        buffer.seek(0)
        st.session_state['report_buffer'] = buffer
        st.success("Report Generated!")
        
    except Exception as e:
        st.error(f"Report Generation Failed: {e}")
    
    finally:
        st.session_state.is_generating_report = False
        prog_bar.empty()
        st.rerun()

# ==========================================
# RESULTS UI
# ==========================================

results_ui.render_results_section(st.session_state['sysA'], st.session_state['sysB'], raw_res_A, raw_res_B, nodes_A, nodes_B)