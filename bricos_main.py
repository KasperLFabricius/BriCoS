import streamlit as st
import pandas as pd
import numpy as np
import json
import copy
import os
import bricos_solver as solver
import bricos_viz as viz

# ==========================================
# UI SETUP & CONFIGURATION
# ==========================================

st.set_page_config(layout="wide", page_title="BriCoS v0.30")

st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 0rem;}
    div[data-testid="stExpander"] div[role="button"] p {font-size: 1rem; font-weight: bold;}
    .stSelectbox label { font-size: 0.9rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- LOGO DISPLAY ---
if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", width='stretch')

st.title("BriCoS v0.30 - Bridge Comparison Software")

# --- PERSISTENT STATE INITIALIZATION ---
if 'keep_view_case' not in st.session_state: st.session_state.keep_view_case = "Total Envelope"
if 'keep_active_veh_step' not in st.session_state: st.session_state.keep_active_veh_step = "Vehicle A"
if 'keep_step_view_sys' not in st.session_state: st.session_state.keep_step_view_sys = "System A"

def calc_I(h_mm):
    return (1.0 * (h_mm/1000.0)**3) / 12.0

def load_vehicle_from_csv(target_name):
    """Attempts to read a specific vehicle from CSV. Returns dict or None."""
    try:
        if os.path.exists("vehicles.csv"):
            df = pd.read_csv("vehicles.csv")
            # Strip whitespace from columns
            df.columns = [c.strip() for c in df.columns]
            
            # Look for name
            row = df[df['Name'] == target_name]
            if not row.empty:
                l_str = str(row.iloc[0]['Loads'])
                s_str = str(row.iloc[0]['Spacing'])
                
                # Parse to arrays
                l_arr = [float(x) for x in l_str.split(',')]
                s_arr = [float(x) for x in s_str.split(',')]
                
                return {
                    'loads': l_arr, 'spacing': s_arr,
                    'l_str': l_str, 's_str': s_str
                }
    except:
        pass
    return None

def get_def():
    # Updated Defaults based on User Request
    I_def = calc_I(500)
    
    # Attempt to load Class 100
    def_veh = load_vehicle_from_csv("Class 100")
    
    if def_veh:
        veh_obj = {'loads': def_veh['loads'], 'spacing': def_veh['spacing']}
        veh_l_str = def_veh['l_str']
        veh_s_str = def_veh['s_str']
    else:
        # Fallback if CSV missing or Class 100 not found
        veh_obj = {'loads': [100.0], 'spacing': [0.0]}
        veh_l_str = "100.0"
        veh_s_str = "0.0"

    return {
        'mode': 'Frame', 
        'E': 33e6, # Approx C30
        'num_spans': 1,
        'L_list': [10.0]*10,
        'Is_list': [I_def]*10,
        'sw_list': [20.0]*10,
        'h_list': [8.0]*11,
        'Iw_list': [I_def]*11,
        # Material Properties (C30 -> fck=30)
        'e_mode': 'Eurocode',
        'fck_span_list': [30.0]*10,
        'fck_wall_list': [30.0]*11,
        'E_custom_span': [33.0]*10, 
        'E_custom_wall': [33.0]*11,
        'E_span_list': [33e6]*10,
        'E_wall_list': [33e6]*11,
        
        'supports': [], 
        'soil': [], # Populated in init
        'surcharge': [], 
        
        # Default Vehicle A
        'vehicle': veh_obj, 
        'vehicle_loads': veh_l_str, 
        'vehicle_space': veh_s_str,
        
        # Default Vehicle B (Empty/Custom)
        'vehicleB': {'loads': [], 'spacing': []},
        'vehicleB_loads': "", 
        'vehicleB_space': "",
        
        'KFI': 1.1, 
        'gamma_g': 1.0, 'gamma_j': 1.0, 
        'gamma_veh': 1.4, 'gamma_vehB': 1.05, 
        'phi': 1.0, 'scale_manual': 2.0,
        'phi_mode': 'Calculate',
        'mesh_size': 0.5, 'step_size': 0.2,
        'name': 'System',
        'last_mode': 'Frame',
        'combine_surcharge_vehicle': False,
        'vehicle_direction': 'Forward'
    }

def get_clear(name_suffix, current_mode):
    return {
        'mode': current_mode, 
        'E': 30e6, 'num_spans': 1,
        'L_list': [0.0]*10, 'Is_list': [0.0]*10, 'sw_list': [0.0]*10,
        'h_list': [0.0]*11, 'Iw_list': [0.0]*11,
        'e_mode': 'Eurocode',
        'fck_span_list': [30.0]*10, 'fck_wall_list': [30.0]*11,
        'E_custom_span': [33.0]*10, 'E_custom_wall': [33.0]*11,
        'E_span_list': [33e6]*10, 'E_wall_list': [33e6]*11,
        
        'supports': [],
        'soil': [],
        'surcharge': [], 
        'vehicle': {'loads': [], 'spacing': []},
        'vehicle_loads': "", 'vehicle_space': "",
        'vehicleB': {'loads': [], 'spacing': []},
        'vehicleB_loads': "", 'vehicleB_space': "",
        'KFI': 1.0, 
        'gamma_g': 1.0, 'gamma_j': 1.0, 
        'gamma_veh': 1.0, 'gamma_vehB': 1.0, 'phi': 1.0, 'scale_manual': 2.0,
        'phi_mode': 'Manual', 
        'mesh_size': 0.5, 'step_size': 0.2,
        'name': f"System {name_suffix}",
        'last_mode': current_mode,
        'combine_surcharge_vehicle': False,
        'vehicle_direction': 'Forward'
    }

# --- INITIALIZATION WITH SPECIFIC DEFAULTS ---
if 'sysA' not in st.session_state: 
    # System A: 1 Span
    d = get_def()
    d['num_spans'] = 1
    d['name'] = "System A"
    d['soil'] = [
        # Wall 0 (Left Wall):
        # - Soil Left (Outside) -> Full Height -> face='L'
        # - Soil Right (Inside) -> Half Height -> face='R'
        {'wall_idx': 0, 'face': 'L', 'h': 8.0, 'q_top': 0.0, 'q_bot': 20.0}, 
        {'wall_idx': 0, 'face': 'R', 'h': 4.0, 'q_top': 0.0, 'q_bot': 10.0},
        
        # Wall 1 (Right Wall):
        # - Soil Left (Inside) -> Half Height -> face='L'
        # - Soil Right (Outside) -> Full Height -> face='R'
        {'wall_idx': 1, 'face': 'L', 'h': 4.0, 'q_top': 0.0, 'q_bot': 10.0},
        {'wall_idx': 1, 'face': 'R', 'h': 8.0, 'q_top': 0.0, 'q_bot': 20.0}
    ]
    st.session_state['sysA'] = d

if 'sysB' not in st.session_state: 
    # System B: 2 Spans
    d = get_def()
    d['num_spans'] = 2
    d['name'] = "System B"
    d['soil'] = [
        # Wall 0 (Left): Outside Full (L), Inside Half (R)
        {'wall_idx': 0, 'face': 'L', 'h': 8.0, 'q_top': 0.0, 'q_bot': 20.0},
        {'wall_idx': 0, 'face': 'R', 'h': 4.0, 'q_top': 0.0, 'q_bot': 10.0},
        
        # Wall 2 (Right): Inside Half (L), Outside Full (R)
        {'wall_idx': 2, 'face': 'L', 'h': 4.0, 'q_top': 0.0, 'q_bot': 10.0},
        {'wall_idx': 2, 'face': 'R', 'h': 8.0, 'q_top': 0.0, 'q_bot': 20.0}
    ]
    st.session_state['sysB'] = d

if 'name' not in st.session_state['sysA']: st.session_state['sysA']['name'] = "System A"
if 'name' not in st.session_state['sysB']: st.session_state['sysB']['name'] = "System B"

# Migration Logic
for sys_k in ['sysA', 'sysB']:
    if 'gamma_g' not in st.session_state[sys_k]: st.session_state[sys_k]['gamma_g'] = 1.0
    if 'gamma_j' not in st.session_state[sys_k]: st.session_state[sys_k]['gamma_j'] = 1.0
    if 'vehicleB' not in st.session_state[sys_k]:
        st.session_state[sys_k]['vehicleB'] = {'loads': [], 'spacing': []}
        st.session_state[sys_k]['vehicleB_loads'] = ""
        st.session_state[sys_k]['vehicleB_space'] = ""
        st.session_state[sys_k]['gamma_vehB'] = 1.05
    if 'last_mode' not in st.session_state[sys_k]:
         st.session_state[sys_k]['last_mode'] = st.session_state[sys_k]['mode']
    if 'supports' not in st.session_state[sys_k]:
         st.session_state[sys_k]['supports'] = []
    
    # Material Migration
    if 'e_mode' not in st.session_state[sys_k]: st.session_state[sys_k]['e_mode'] = 'Eurocode'
    if 'fck_span_list' not in st.session_state[sys_k]: st.session_state[sys_k]['fck_span_list'] = [30.0]*10
    if 'fck_wall_list' not in st.session_state[sys_k]: st.session_state[sys_k]['fck_wall_list'] = [30.0]*11
    if 'E_custom_span' not in st.session_state[sys_k]: st.session_state[sys_k]['E_custom_span'] = [33.0]*10
    if 'E_custom_wall' not in st.session_state[sys_k]: st.session_state[sys_k]['E_custom_wall'] = [33.0]*11
    if 'E_span_list' not in st.session_state[sys_k]: st.session_state[sys_k]['E_span_list'] = [33e6]*10
    if 'E_wall_list' not in st.session_state[sys_k]: st.session_state[sys_k]['E_wall_list'] = [33e6]*11
    
    # Remove old keys
    st.session_state[sys_k].pop('k_rot', None)
    st.session_state[sys_k].pop('k_rot_super', None)

if 'uploader_key' not in st.session_state: st.session_state.uploader_key = 0

if st.session_state['sysA'].get('scale_manual', 0) < 0.1: st.session_state['sysA']['scale_manual'] = 2.0
if st.session_state['sysB'].get('scale_manual', 0) < 0.1: st.session_state['sysB']['scale_manual'] = 2.0

st.sidebar.header("Configuration")

# --- RESET DATA SECTION ---
with st.sidebar.expander("Reset Data", expanded=False):
    if 'reset_mode' not in st.session_state: st.session_state.reset_mode = None
    if 'reset_action' not in st.session_state: st.session_state.reset_action = None 
    
    c_res, c_clr = st.columns(2)
    with c_res:
        st.caption("Restore Defaults")
        if st.button("Restore A"):
            st.session_state.reset_mode, st.session_state.reset_action = "A", "restore"
            st.rerun()
        if st.button("Restore B"):
            st.session_state.reset_mode, st.session_state.reset_action = "B", "restore"
            st.rerun()
        if st.button("Restore All"):
            st.session_state.reset_mode, st.session_state.reset_action = "ALL", "restore"
            st.rerun()

    with c_clr:
        st.caption("Clear Data (Zero)")
        if st.button("Clear A"):
            st.session_state.reset_mode, st.session_state.reset_action = "A", "clear"
            st.rerun()
        if st.button("Clear B"):
            st.session_state.reset_mode, st.session_state.reset_action = "B", "clear"
            st.rerun()
        if st.button("Clear All"):
            st.session_state.reset_mode, st.session_state.reset_action = "ALL", "clear"
            st.rerun()
        
    if st.session_state.reset_mode:
        action_text = "Restore Defaults to" if st.session_state.reset_action == "restore" else "Clear All Data from"
        st.warning(f"⚠️ {action_text} {st.session_state.reset_mode}? Unsaved data will be lost.")
        
        c_yes, c_no = st.columns(2)
        if c_yes.button("Confirm Action"):
            mode = st.session_state.reset_mode
            action = st.session_state.reset_action
            
            def reset_system_state(target_key, new_data):
                # 1. CLEANUP: Delete ALL widget states for this system first.
                # This ensures input widgets (including Vehicle selectors) re-initialize 
                # from the new_data on the next run.
                prefix = f"{target_key}_"
                keys_to_del = [k for k in st.session_state.keys() if k.startswith(prefix)]
                for k in keys_to_del: del st.session_state[k]

                # 2. SET DATA: Apply the new data dictionary
                st.session_state[target_key] = new_data
                
                # 3. SYNC HELPERS: Re-establish keys needed for sidebar rendering logic
                st.session_state[f"{target_key}_nsp"] = new_data['num_spans']
                st.session_state[f"{target_key}_md_sel"] = new_data['mode']
                st.session_state[f"{target_key}_kfi"] = new_data['KFI']
                # Note: Vehicle keys are NOT set here; they will auto-init in 'handle_veh_inputs' on rerun.

            if mode == "A" or mode == "ALL":
                current_mode = st.session_state['sysA']['mode']
                data = {**get_def(), 'num_spans':1, 'name': "System A"}
                # Re-apply A specific soil
                data['soil'] = [
                    {'wall_idx': 0, 'face': 'L', 'h': 8.0, 'q_top': 0.0, 'q_bot': 20.0}, 
                    {'wall_idx': 0, 'face': 'R', 'h': 4.0, 'q_top': 0.0, 'q_bot': 10.0}, 
                    {'wall_idx': 1, 'face': 'L', 'h': 4.0, 'q_top': 0.0, 'q_bot': 10.0}, 
                    {'wall_idx': 1, 'face': 'R', 'h': 8.0, 'q_top': 0.0, 'q_bot': 20.0}
                ]
                if action == "clear": data = get_clear("A", current_mode)
                reset_system_state("sysA", data)
                
            if mode == "B" or mode == "ALL":
                current_mode = st.session_state['sysB']['mode']
                data = {**get_def(), 'num_spans':2, 'name': "System B"}
                # Re-apply B specific soil
                data['soil'] = [
                    {'wall_idx': 0, 'face': 'L', 'h': 8.0, 'q_top': 0.0, 'q_bot': 20.0},
                    {'wall_idx': 0, 'face': 'R', 'h': 4.0, 'q_top': 0.0, 'q_bot': 10.0},
                    {'wall_idx': 2, 'face': 'L', 'h': 4.0, 'q_top': 0.0, 'q_bot': 10.0},
                    {'wall_idx': 2, 'face': 'R', 'h': 8.0, 'q_top': 0.0, 'q_bot': 20.0}
                ]
                if action == "clear": data = get_clear("B", current_mode)
                reset_system_state("sysB", data)
                
            st.session_state.reset_mode = None
            st.session_state.reset_action = None
            st.rerun()
            
        if c_no.button("Cancel"):
            st.session_state.reset_mode = None
            st.session_state.reset_action = None
            st.rerun()

# --- FILE OPERATIONS ---
with st.sidebar.expander("File Operations (Save/Load)", expanded=False):
    def to_csv():
        rows = []
        for sys_name in ['sysA', 'sysB']:
            data = st.session_state[sys_name]
            for k, v in data.items():
                if k == 'backup': continue 
                val_str = json.dumps(v, default=str)
                rows.append({'System': sys_name, 'Parameter': k, 'Value': val_str})
        df = pd.DataFrame(rows)
        return df.to_csv(index=False).encode('utf-8')

    st.download_button("Download Configuration (.csv)", to_csv(), "brico_config.csv", "text/csv")

    uploaded_file = st.file_uploader("Upload Configuration (.csv)", type="csv", key=f"uploader_{st.session_state.uploader_key}")
    if uploaded_file is not None:
        try:
            df_load = pd.read_csv(uploaded_file)
            if 'System' in df_load.columns and 'Parameter' in df_load.columns:
                for _, row in df_load.iterrows():
                    sys_n = row['System']
                    if sys_n in ['sysA', 'sysB']:
                        try:
                            val = json.loads(row['Value'])
                            st.session_state[sys_n][row['Parameter']] = val
                        except: pass
                
                keys_to_clear = [k for k in st.session_state.keys() if k.startswith('sysA_') or k.startswith('sysB_')]
                for k in keys_to_clear: del st.session_state[k]
                st.session_state.uploader_key += 1
                st.success("Configuration loaded! UI will update.")
                st.rerun()
            else: st.error("Invalid CSV format.")
        except Exception as e: st.error(f"Error loading file: {e}")

# --- COPY SYSTEM FEATURE ---
with st.sidebar.expander("Copy Data", expanded=False):
    if 'copy_confirm_mode' not in st.session_state: st.session_state.copy_confirm_mode = None
    c_cp1, c_cp2 = st.columns(2)
    if c_cp1.button("Copy A → B"):
        st.session_state.copy_confirm_mode = "A2B"
        st.rerun()
    if c_cp2.button("Copy B → A"):
        st.session_state.copy_confirm_mode = "B2A"
        st.rerun()

    if st.session_state.copy_confirm_mode == "A2B":
        st.warning("⚠️ Overwrite System B?")
        c_yes, c_no = st.columns(2)
        if c_yes.button("Confirm"):
            nm = st.session_state['sysB']['name']
            st.session_state['sysB'] = copy.deepcopy(st.session_state['sysA'])
            st.session_state['sysB']['name'] = nm
            prefix = "sysB_"
            keys_to_del = [k for k in st.session_state.keys() if k.startswith(prefix)]
            for k in keys_to_del: del st.session_state[k]
            st.session_state.copy_confirm_mode = None
            st.rerun()
        if c_no.button("Cancel"):
            st.session_state.copy_confirm_mode = None
            st.rerun()

    elif st.session_state.copy_confirm_mode == "B2A":
        st.warning("⚠️ Overwrite System A?")
        c_yes, c_no = st.columns(2)
        if c_yes.button("Confirm"):
            nm = st.session_state['sysA']['name']
            st.session_state['sysA'] = copy.deepcopy(st.session_state['sysB'])
            st.session_state['sysA']['name'] = nm
            prefix = "sysA_"
            keys_to_del = [k for k in st.session_state.keys() if k.startswith(prefix)]
            for k in keys_to_del: del st.session_state[k]
            st.session_state.copy_confirm_mode = None
            st.rerun()
        if c_no.button("Cancel"):
            st.session_state.copy_confirm_mode = None
            st.rerun()

c_nA, c_nB = st.sidebar.columns(2)
st.session_state['sysA']['name'] = c_nA.text_input("Name Sys A", st.session_state['sysA']['name'])
st.session_state['sysB']['name'] = c_nB.text_input("Name Sys B", st.session_state['sysB']['name'])

sys_map = {"sysA": f"{st.session_state['sysA']['name']} (Blue)", "sysB": f"{st.session_state['sysB']['name']} (Red)"}
active_sys_key = st.sidebar.radio("Active System:", ["sysA", "sysB"], format_func=lambda x: sys_map[x], horizontal=True)

if active_sys_key == 'sysA':
    st.markdown("""<style>[data-testid="stSidebar"] { background-color: #F0F8FF; }</style>""", unsafe_allow_html=True)
else:
    st.markdown("""<style>[data-testid="stSidebar"] { background-color: #FFF5F5; }</style>""", unsafe_allow_html=True)

curr = active_sys_key
p = st.session_state[curr]

# --- MAIN INPUTS ---
with st.sidebar.expander("Design Factors & Type", expanded=True):
    help_mode = "Choose 'Frame' for full interaction (Walls + Slab) or 'Superstructure' for a simplified slab-on-supports analysis."
    new_mode_sel = st.selectbox("Model Type", ["Frame", "Superstructure"], index=0 if p['mode']=='Frame' else 1, key=f"{curr}_md_sel", help=help_mode)
    
    old_mode = p.get('last_mode', 'Frame')
    if old_mode != new_mode_sel:
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
    
    help_mat = "Choose method for Elastic Modulus (E) definition:\n- Eurocode: Calculate E from f_ck (Ecm = 22 * (fcm/10)^0.3)\n- Manual: Enter E directly in GPa."
    e_mode = st.radio("Material Definition", ["Eurocode (f_ck)", "Manual (E-Modulus)"], horizontal=True, index=0 if p['e_mode']=='Eurocode' else 1, key=f"{curr}_emode", help=help_mat)
    p['e_mode'] = "Eurocode" if "Eurocode" in e_mode else "Manual"

    help_kfi = "Consequence Class Factor (KFI) applied to all loads."
    p['KFI'] = st.selectbox("KFI (Consequence Class)", [0.9, 1.0, 1.1], index=2, key=f"{curr}_kfi", help=help_kfi)
    
    gg_opts = [0.9, 1.0, 1.10, 1.25]
    c_gg, c_gj = st.columns(2)
    gg_val = p.get('gamma_g', 1.0)
    idx_gg = gg_opts.index(gg_val) if gg_val in gg_opts else len(gg_opts)
    
    help_gg = "Partial factor for permanent self-weight loads."
    gg_sel = c_gg.selectbox(r"$\gamma_{g}$ (Self-weight)", gg_opts + ["Custom"], index=min(idx_gg, len(gg_opts)), key=f"{curr}_gg_sel", help=help_gg)
    if gg_sel == "Custom": p['gamma_g'] = c_gg.number_input(r"Custom $\gamma_{g}$", value=float(gg_val), key=f"{curr}_gg_cust")
    else: p['gamma_g'] = float(gg_sel)

    gj_opts = [1.0, 1.1]
    gj_val = p.get('gamma_j', 1.0)
    idx_gj = gj_opts.index(gj_val) if gj_val in gj_opts else len(gj_opts)
    
    help_gj = "Partial factor for permanent soil loads (Earth Pressure)."
    gj_sel = c_gj.selectbox(r"$\gamma_{j}$ (Soil)", gj_opts + ["Custom"], index=min(idx_gj, len(gj_opts)), key=f"{curr}_gj_sel", help=help_gj)
    if gj_sel == "Custom": p['gamma_j'] = c_gj.number_input(r"Custom $\gamma_{j}$", value=float(gj_val), key=f"{curr}_gj_cust")
    else: p['gamma_j'] = float(gj_sel)

    gam_opts = [0.56, 1.0, 1.05, 1.25, 1.40]
    c_ga, c_gb = st.columns(2)
    gam_valA = p.get('gamma_veh', 1.0)
    idx_gamA = gam_opts.index(gam_valA) if gam_valA in gam_opts else len(gam_opts)
    
    help_gamA = "Partial factor for variable traffic loads (Vehicle A)."
    gam_selA = c_ga.selectbox(r"$\gamma_{veh,A}$", gam_opts + ["Custom"], index=min(idx_gamA, len(gam_opts)), key=f"{curr}_gamA_sel", help=help_gamA)
    if gam_selA == "Custom": p['gamma_veh'] = c_ga.number_input(r"Custom $\gamma_{A}$", value=float(gam_valA), key=f"{curr}_gamA_cust")
    else: p['gamma_veh'] = float(gam_selA)

    gam_valB = p.get('gamma_vehB', 1.0)
    idx_gamB = gam_opts.index(gam_valB) if gam_valB in gam_opts else len(gam_opts)
    
    help_gamB = "Partial factor for variable traffic loads (Vehicle B)."
    gam_selB = c_gb.selectbox(r"$\gamma_{veh,B}$", gam_opts + ["Custom"], index=min(idx_gamB, len(gam_opts)), key=f"{curr}_gamB_sel", help=help_gamB)
    if gam_selB == "Custom": p['gamma_vehB'] = c_gb.number_input(r"Custom $\gamma_{B}$", value=float(gam_valB), key=f"{curr}_gamB_cust")
    else: p['gamma_vehB'] = float(gam_selB)

    help_phi = "Dynamic Amplification Factor calculation method. Manual allows fixed input; Calculate uses Eurocode logic based on span lengths."
    phi_mode = st.radio("Dynamic Factor (Phi)", ["Calculate", "Manual"], horizontal=True, index=0 if p.get('phi_mode', 'Calculate') == 'Calculate' else 1, key=f"{curr}_phim", help=help_phi)
    p['phi_mode'] = phi_mode
    if phi_mode == "Manual":
        p['phi'] = st.number_input("Phi Value", value=p.get('phi', 1.0), key=f"{curr}_phiv")
    
    phi_log_placeholder = st.empty()

with st.sidebar.expander("Geometry, Stiffness & Static Loads", expanded=True):
    n_spans = st.number_input("Number of Spans", 1, 10, p['num_spans'], key=f"{curr}_nsp")
    p['num_spans'] = n_spans
    
    # Material Header logic
    is_ec = (p['e_mode'] == 'Eurocode')
    lbl_mat = "f_ck [MPa]" if is_ec else "E [GPa]"
    help_mat_col = "Concrete Cylinder Strength (f_ck)." if is_ec else "Elastic Modulus (Young's Modulus)."
    
    st.markdown("---")
    st.markdown("**Spans (L, I, SW, Material)**")
    
    # Ensure lists are long enough
    while len(p['fck_span_list']) < 10: p['fck_span_list'].append(35.0)
    while len(p['E_custom_span']) < 10: p['E_custom_span'].append(34.0)
    while len(p['E_span_list']) < 10: p['E_span_list'].append(34e6)

    # --- ADVANCED SECTION HELPER ---
    def get_geom_ui_data(prefix_key, i, default_val):
        key = f"{prefix_key}_{i}"
        if key not in p:
            # Init Default: Type=0(I), Shape=0(Const), Val=Default
            p[key] = {
                'type': 0, 'shape': 0, 'vals': [default_val, default_val, default_val],
                'align_type': 0, 'incline_mode': 0, 'incline_val': 0.0
            }
        return p[key]

    for i in range(n_spans):
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        p['L_list'][i] = c1.number_input(f"L{i+1} [m]", value=float(p['L_list'][i]), key=f"{curr}_l{i}", help="Length of Span" if i==0 else None)
        
        # Check if advanced geom is active (shape != 0)
        s_geom = get_geom_ui_data('span_geom', i, p['Is_list'][i])
        is_adv = (s_geom['shape'] != 0) or (s_geom['type'] != 0) or (s_geom.get('align_type', 0) != 0)
        
        # Simple View Logic: Update list from geom vals
        if not is_adv:
            val = c2.number_input(f"I{i+1} [m⁴]", value=float(p['Is_list'][i]), format="%.4f", key=f"{curr}_i{i}", help="Inertia" if i==0 else None)
            p['Is_list'][i] = val
            s_geom['vals'] = [val, val, val] # Sync
        else:
            c2.text_input(f"I{i+1}", "See Profiler", disabled=True, key=f"{curr}_i{i}_dis")

        # UPDATED SELFWEIGHT HELP
        p['sw_list'][i] = c3.number_input(f"SW{i+1} [kN/m]", value=float(p['sw_list'][i]), key=f"{curr}_s{i}", help="Distributed load from permanent dead loads, such as structure selfweight, surfacing, soil and another permanent loads." if i==0 else None)
        
        # MATERIAL
        if is_ec:
            val_in = c4.number_input(f"{lbl_mat}", value=float(p['fck_span_list'][i]), key=f"{curr}_fck_s{i}", help=help_mat_col if i==0 else None)
            p['fck_span_list'][i] = val_in
            E_gpa = 22.0 * ((val_in + 8)/10.0)**0.3
            p['E_span_list'][i] = E_gpa * 1e6
        else:
            val_in = c4.number_input(f"{lbl_mat}", value=float(p['E_custom_span'][i]), key=f"{curr}_Eman_s{i}", help=help_mat_col if i==0 else None)
            p['E_custom_span'][i] = val_in
            p['E_span_list'][i] = val_in * 1e6
        
    is_super = (p['mode'] == 'Superstructure')
    st.markdown("---")
    st.markdown("**Walls (H, I, Surcharge, Material)**")
    if is_super: st.caption("Mode: Superstructure. Walls and lateral loads are disabled.")
    
    # Ensure wall lists long enough
    while len(p['fck_wall_list']) < 11: p['fck_wall_list'].append(35.0)
    while len(p['E_custom_wall']) < 11: p['E_custom_wall'].append(34.0)
    while len(p['E_wall_list']) < 11: p['E_wall_list'].append(34e6)

    for i in range(n_spans + 1):
        st.caption(f"Wall {i+1}")
        c1, c2, c3, c4 = st.columns([1,1,1,1])
        p['h_list'][i] = c1.number_input(f"H [m]", value=float(p['h_list'][i]), disabled=is_super, key=f"{curr}_h{i}", help="Wall Height" if i==0 else None)
        
        w_geom = get_geom_ui_data('wall_geom', i, p['Iw_list'][i])
        is_adv_w = (w_geom['shape'] != 0) or (w_geom['type'] != 0)

        if not is_adv_w:
            val_w = c2.number_input(f"I [m⁴]", value=float(p['Iw_list'][i]), format="%.4f", disabled=is_super, key=f"{curr}_iw{i}", help="Inertia" if i==0 else None)
            p['Iw_list'][i] = val_w
            w_geom['vals'] = [val_w, val_w, val_w]
        else:
            c2.text_input(f"I", "See Profiler", disabled=True, key=f"{curr}_iw{i}_dis")
        
        sur = next((x for x in p['surcharge'] if x['wall_idx']==i), None)
        val_q = sur['q'] if sur else 0.0
        # UPDATED SURCHARGE HELP
        new_q = c3.number_input(f"Surch. [kN/m]", value=float(val_q), disabled=is_super, key=f"{curr}_sq{i}", help="Vehicle Surcharge, acting uniformly on the wall. Dynamic Factor is never added to surcharge load. For combination with vertical vehicle loads, see Analysis settings." if i==0 else None)
        
        if not is_super:
            p['surcharge'] = [x for x in p['surcharge'] if x['wall_idx'] != i]
            if new_q != 0: 
                p['surcharge'].append({'wall_idx':i, 'face':'R', 'q':new_q, 'h':p['h_list'][i]})

        # MATERIAL COLUMN WALL
        if is_ec:
            val_in = c4.number_input(f"{lbl_mat}", value=float(p['fck_wall_list'][i]), disabled=is_super, key=f"{curr}_fck_w{i}", help=help_mat_col if i==0 else None)
            p['fck_wall_list'][i] = val_in
            E_gpa = 22.0 * ((val_in + 8)/10.0)**0.3
            p['E_wall_list'][i] = E_gpa * 1e6
        else:
            val_in = c4.number_input(f"{lbl_mat}", value=float(p['E_custom_wall'][i]), disabled=is_super, key=f"{curr}_Eman_w{i}", help=help_mat_col if i==0 else None)
            p['E_custom_wall'][i] = val_in
            p['E_wall_list'][i] = val_in * 1e6

        # --- UPDATED SOIL UI LOGIC (CORRECTED) ---
        # Soil Left (User Input) -> Pushes Right -> face='L'
        # Soil Right (User Input) -> Pushes Left -> face='R'
        
        ex_SoilLeft = next((x for x in p['soil'] if x['wall_idx']==i and x['face']=='L'), None)
        ex_SoilRight = next((x for x in p['soil'] if x['wall_idx']==i and x['face']=='R'), None)
        
        c_sl, c_sr = st.columns(2)
        
        # Left Column = H_soil_left (Writes to face='L')
        h_L = c_sl.number_input("H_soil_left [m]", value=ex_SoilLeft['h'] if ex_SoilLeft else 0.0, disabled=is_super, key=f"{curr}_shl{i}", help="Soil on Left Face (Pushing Right)")
        qL_bot = c_sl.number_input("q_bot [kN/m²]", value=ex_SoilLeft['q_bot'] if ex_SoilLeft else 0.0, disabled=is_super, key=f"{curr}_sqlb{i}", help="Pressure Bot")
        qL_top = c_sl.number_input("q_top [kN/m²]", value=ex_SoilLeft['q_top'] if ex_SoilLeft else 0.0, disabled=is_super, key=f"{curr}_sqlt{i}", help="Pressure Top")
        
        # Right Column = H_soil_right (Writes to face='R')
        h_R = c_sr.number_input("H_soil_right [m]", value=ex_SoilRight['h'] if ex_SoilRight else 0.0, disabled=is_super, key=f"{curr}_shr{i}", help="Soil on Right Face (Pushing Left)")
        qR_bot = c_sr.number_input("q_bot [kN/m²]", value=ex_SoilRight['q_bot'] if ex_SoilRight else 0.0, disabled=is_super, key=f"{curr}_sqrb{i}")
        qR_top = c_sr.number_input("q_top [kN/m²]", value=ex_SoilRight['q_top'] if ex_SoilRight else 0.0, disabled=is_super, key=f"{curr}_sqrt{i}")

        if not is_super:
            p['soil'] = [x for x in p['soil'] if x['wall_idx']!=i]
            if h_L > 0: p['soil'].append({'wall_idx':i, 'face':'L', 'q_bot':qL_bot, 'q_top':qL_top, 'h':h_L})
            if h_R > 0: p['soil'].append({'wall_idx':i, 'face':'R', 'q_bot':qR_bot, 'q_top':qR_top, 'h':h_R})

    # --- SECTION PROFILER ---
    st.markdown("---")
    # CHANGED: Wrapped in Expander
    with st.sidebar.expander("🛠️ Section Profiler (Advanced)", expanded=False):
        st.caption("Configure variable stiffness, height profiles, or vertical alignment.")
        
        # Selector
        elem_options = [f"Span {i+1}" for i in range(n_spans)] + ([f"Wall {i+1}" for i in range(n_spans+1)] if not is_super else [])
        sel_el = st.selectbox("Edit Element:", elem_options, key=f"{curr}_prof_sel")
        
        # Parse Selection
        is_span_selected = "Span" in sel_el
        if is_span_selected:
            idx = int(sel_el.split(" ")[1]) - 1
            target_geom = p[f"span_geom_{idx}"]
            target_simple_list = p['Is_list']
        else:
            idx = int(sel_el.split(" ")[1]) - 1
            target_geom = p[f"wall_geom_{idx}"]
            target_simple_list = p['Iw_list']
            
        c_p1, c_p2 = st.columns(2)
        new_type = c_p1.radio("Definition Mode:", ["Inertia (I)", "Height (H)"], index=target_geom['type'], key=f"{curr}_prof_type", horizontal=True)
        target_geom['type'] = 0 if "Inertia" in new_type else 1
        
        new_shape = c_p2.radio("Profile Shape:", ["Constant", "Linear (Taper)", "3-Point (Start/Mid/End)"], index=target_geom['shape'], key=f"{curr}_prof_shape", horizontal=True)
        shape_map = {"Constant": 0, "Linear (Taper)": 1, "3-Point (Start/Mid/End)": 2}
        target_geom['shape'] = shape_map[new_shape]
        
        vals = target_geom['vals']
        c_v1, c_v2, c_v3 = st.columns(3)
        
        lbl_v = "I [m⁴]" if target_geom['type']==0 else "H [m]"
        v1 = c_v1.number_input(f"Start {lbl_v}", value=float(vals[0]), format="%.4f", key=f"{curr}_prof_v1")
        
        v2 = vals[1]
        if target_geom['shape'] == 2:
            v2 = c_v2.number_input(f"Mid {lbl_v}", value=float(vals[1]), format="%.4f", key=f"{curr}_prof_v2")
        
        v3 = vals[2]
        if target_geom['shape'] >= 1:
            v3 = c_v3.number_input(f"End {lbl_v}", value=float(vals[2]), format="%.4f", key=f"{curr}_prof_v3")
            
        target_geom['vals'] = [v1, v2, v3]
        
        # Update simple list (backward compat)
        target_simple_list[idx] = v1 if target_geom['type']==0 else (1.0 * v1**3)/12.0

        # --- ALIGNMENT SECTION (Spans Only) ---
        if is_span_selected:
            st.markdown("#### 📐 Alignment (Vertical Geometry)")
            if 'align_type' not in target_geom: target_geom['align_type'] = 0
            if 'incline_mode' not in target_geom: target_geom['incline_mode'] = 0
            if 'incline_val' not in target_geom: target_geom['incline_val'] = 0.0

            al_opts = ["Straight (Horizontal)", "Inclined"]
            new_align = st.radio("Span Profile:", al_opts, index=target_geom['align_type'], horizontal=True, key=f"{curr}_align_t")
            target_geom['align_type'] = al_opts.index(new_align)
            
            if target_geom['align_type'] == 1:
                inc_opts = ["Slope (%)", "Delta Height (End - Start) [m]"]
                new_inc_mode = st.radio("Define Inclination by:", inc_opts, index=target_geom['incline_mode'], horizontal=True, key=f"{curr}_inc_m")
                target_geom['incline_mode'] = inc_opts.index(new_inc_mode)
                
                lbl_inc = "Slope [%]" if target_geom['incline_mode'] == 0 else "Delta H [m]"
                help_inc = "Positive slope/height goes UP. Negative goes DOWN."
                target_geom['incline_val'] = st.number_input(lbl_inc, value=float(target_geom['incline_val']), format="%.2f", help=help_inc, key=f"{curr}_inc_v")


# --- NEW BOUNDARY CONDITIONS TAB ---
with st.sidebar.expander("Boundary Conditions", expanded=False):
    # Dynamic list management for supports
    num_supports = n_spans + 1
    current_supports = p.get('supports', [])
    
    # Ensure list length matches current geometry
    if len(current_supports) != num_supports:
        new_list = []
        for i in range(num_supports):
            if i < len(current_supports):
                new_list.append(current_supports[i])
            else:
                # Default Logic
                if p['mode'] == 'Frame':
                    # Fixed Base
                    new_list.append({'type': 'Fixed', 'k': [1e14, 1e14, 1e14]})
                else:
                    if i == 0:
                        new_list.append({'type': 'Pinned', 'k': [1e14, 1e14, 0.0]})
                    else:
                        new_list.append({'type': 'Roller (X-Free)', 'k': [0.0, 1e14, 0.0]})
        p['supports'] = new_list
    
    # Render Inputs
    presets = {
        "Fixed": [1e14, 1e14, 1e14],
        "Pinned": [1e14, 1e14, 0.0],
        "Roller (X-Free)": [0.0, 1e14, 0.0],
        "Roller (Y-Free)": [1e14, 0.0, 0.0],
        "Custom Spring": None
    }
    
    for i in range(num_supports):
        supp_name = f"Wall {i+1} Base" if p['mode'] == 'Frame' else f"Support {i+1}"
        st.markdown(f"**{supp_name}**")
        
        # Get current state
        curr_s = p['supports'][i]
        curr_type = curr_s.get('type', 'Fixed')
        if curr_type not in presets: curr_type = 'Custom Spring'
        
        sel_type = st.selectbox(f"Type {i+1}", list(presets.keys()), index=list(presets.keys()).index(curr_type), key=f"{curr}_supp_t_{i}", label_visibility="collapsed")
        
        new_k = curr_s['k']
        if sel_type != "Custom Spring":
            new_k = presets[sel_type]
            p['supports'][i]['type'] = sel_type
            p['supports'][i]['k'] = new_k
        else:
            p['supports'][i]['type'] = "Custom Spring"
            col_k1, col_k2, col_k3 = st.columns(3)
            kx = col_k1.number_input(f"Kx", value=float(curr_s['k'][0]), format="%.1e", key=f"{curr}_kx_{i}", help="Horizontal Stiffness [kN/m]")
            ky = col_k2.number_input(f"Ky", value=float(curr_s['k'][1]), format="%.1e", key=f"{curr}_ky_{i}", help="Vertical Stiffness [kN/m]")
            km = col_k3.number_input(f"Km", value=float(curr_s['k'][2]), format="%.1e", key=f"{curr}_km_{i}", help="Rotational Stiffness [kNm/rad]")
            p['supports'][i]['k'] = [kx, ky, km]

@st.cache_data
def get_vehicle_library():
    options = ["Custom"]
    data = {}
    try:
        df_v = pd.read_csv("vehicles.csv")
        if 'Name' in df_v.columns and 'Loads' in df_v.columns and 'Spacing' in df_v.columns:
            for idx, row in df_v.iterrows():
                v_name = row['Name']
                options.append(v_name)
                data[v_name] = {'loads': str(row['Loads']), 'spacing': str(row['Spacing'])}
    except: pass
    return options, data

with st.sidebar.expander("Vehicle Definitions", expanded=True):
    veh_options, veh_data = get_vehicle_library()
    
    def handle_veh_inputs(prefix, key_class, key_loads, key_space, struct_key):
        # Allow default override via session state (from Reset/Init logic)
        sess_key = f"{curr}_{prefix}_class"
        
        # CHANGE: Differentiate default based on Prefix
        default_class = "Class 100" if prefix == "A" else "Custom"
        if sess_key not in st.session_state: st.session_state[sess_key] = default_class
        
        # Sync Class 100 on First Load if needed (Only for A)
        if prefix == "A" and st.session_state[sess_key] == "Class 100" and st.session_state[sess_key] in veh_data and not p[struct_key]['loads']:
             p[key_loads] = veh_data["Class 100"]['loads']
             p[key_space] = veh_data["Class 100"]['spacing']
        
        sel_class = st.selectbox(f"Class {prefix}", veh_options, key=sess_key)
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
        
        h_loads = "Comma-separated axle loads [tonnes]. Example: 10,10,12"
        h_space = "Comma-separated distances [m] between axles. First value must be 0. Example: 0, 1.5, 3.0"
        
        p[key_loads] = st.text_input(f"Loads {prefix} [t]", key=input_key_l, help=h_loads)
        p[key_space] = st.text_input(f"Space {prefix} [m]", key=input_key_s, help=h_space)
        
        valid_veh = False
        msg = ""
        try:
            if p[key_loads].strip():
                l_arr = [float(x) for x in p[key_loads].split(',')]
                s_arr = [float(x) for x in p[key_space].split(',')]
                if len(l_arr) != len(s_arr): msg = "Error: Mismatch in number of Loads vs Spacings."
                elif len(s_arr) > 0 and s_arr[0] != 0: msg = "Error: First spacing value must be 0."
                elif len(l_arr) == 0: msg = "Empty."
                else:
                    valid_veh = True
                    p[struct_key]['loads'] = l_arr; p[struct_key]['spacing'] = s_arr
            else:
                p[struct_key]['loads'] = []; p[struct_key]['spacing'] = []
        except: msg = "Error: Invalid number format."
        
        if not valid_veh:
            if p[key_loads].strip(): st.error(f"Invalid Vehicle: {msg}"); p[struct_key]['loads'] = []
            else: st.caption("No vehicle defined (skips analysis).")
        else: st.success("Vehicle Valid")

    st.markdown("**Vehicle A**")
    handle_veh_inputs("A", f"{curr}_vehA_class", 'vehicle_loads', 'vehicle_space', 'vehicle')
    st.markdown("---")
    st.markdown("**Vehicle B**")
    handle_veh_inputs("B", f"{curr}_vehB_class", 'vehicleB_loads', 'vehicleB_space', 'vehicleB')

# --- NEW: ANALYSIS SETTINGS EXPANDER (Moved BEFORE execution to fix lag) ---
with st.sidebar.expander("Analysis & Result Settings", expanded=True):
    help_dir = "Forward: Left to Right. Reverse: Right to Left (axles inverted). Both: Envelope of both directions."
    # Determine default index based on current state
    curr_dir = st.session_state['sysA'].get('vehicle_direction', 'Forward')
    dir_opts = ["Forward", "Reverse", "Both"]
    idx_dir = dir_opts.index(curr_dir) if curr_dir in dir_opts else 0
    
    dir_sel = st.radio("Vehicle Direction", dir_opts, horizontal=True, index=idx_dir, key="veh_dir_radio_sidebar", help=help_dir)
    st.session_state['sysA']['vehicle_direction'] = dir_sel
    st.session_state['sysB']['vehicle_direction'] = dir_sel
    
    st.markdown("---")
    help_combo = "Define how the Traffic Surcharge (on walls) and the Main Vehicle (on deck) interact.\n- Exclusive: Load is max(Vehicle, Surcharge).\n- Simultaneous: Load is Vehicle + Surcharge."
    
    # Determine default surcharge index
    is_sim = st.session_state['sysA'].get('combine_surcharge_vehicle', False)
    combo_idx = 1 if is_sim else 0
    
    surch_sel = st.radio("Surcharge Combination", ["Exclusive (Vehicle OR Surcharge)", "Simultaneous (Vehicle + Surcharge)"], index=combo_idx, horizontal=True, key="surcharge_combo_radio_sidebar", help=help_combo)
    
    is_simultaneous = (surch_sel == "Simultaneous (Vehicle + Surcharge)")
    st.session_state['sysA']['combine_surcharge_vehicle'] = is_simultaneous
    st.session_state['sysB']['combine_surcharge_vehicle'] = is_simultaneous
    
    # RESULT TYPE REMOVED FROM HERE, MOVED TO MAIN UI

if "common_mesh_slider" in st.session_state:
    m_val = st.session_state["common_mesh_slider"]
    st.session_state['sysA']['mesh_size'] = m_val
    st.session_state['sysB']['mesh_size'] = m_val

if "common_step_slider" in st.session_state:
    s_val = st.session_state["common_step_slider"]
    st.session_state['sysA']['step_size'] = s_val
    st.session_state['sysB']['step_size'] = s_val

# --- ABOUT SECTION ---
st.sidebar.markdown("---")
with st.sidebar.expander("About", expanded=False):
    st.markdown("**BriCoS v0.30**")
    st.write("Author: Kasper Lindskov Fabricius")
    st.write("Email: Kasper.LindskovFabricius@sweco.dk")
    st.write("A specialized Finite Element Analysis (FEM) tool for rapid bridge analysis and comparison.")

# --- EXECUTION & RESULTS ---
view_options = ["Total Envelope", "Selfweight", "Soil", "Surcharge", "Vehicle Envelope", "Vehicle Steps"]
def set_view_case(): st.session_state.keep_view_case = st.session_state.view_case_selector
try: v_idx = view_options.index(st.session_state.keep_view_case)
except ValueError: v_idx = 0

# Result Mode Logic (MOVED TO MAIN)
# Default initialization if not present
if 'result_mode' not in st.session_state: st.session_state['result_mode'] = "Design (ULS)"

# CALL SOLVER WRAPPED IN TRY/EXCEPT FOR STABILITY
def safe_solve(params):
    try:
        return solver.run_raw_analysis(params)
    except ValueError as e:
        return None, None, str(e)

# Now safe_solve is called AFTER sidebar inputs are processed
raw_res_A, nodes_A, err_A = safe_solve(st.session_state['sysA'])
raw_res_B, nodes_B, err_B = safe_solve(st.session_state['sysB'])

# Display Errors if any
if err_A and isinstance(err_A, str): st.error(f"System A Error: {err_A}")
if err_B and isinstance(err_B, str): st.error(f"System B Error: {err_B}")

# Proceed only if results exist
has_res_A = (raw_res_A is not None)
has_res_B = (raw_res_B is not None)

c1, c2, c3, c4 = st.columns([1,1,1,2])
man_scale = c2.number_input("Target Diagram Height [m]", value=st.session_state['sysA'].get('scale_manual', 2.0), format="%.2f")
st.session_state['sysA']['scale_manual'] = man_scale
st.session_state['sysB']['scale_manual'] = man_scale

c_mesh, c_step = st.columns(2)
def_mesh = st.session_state['sysA'].get('mesh_size', 0.5)
def_step = st.session_state['sysA'].get('step_size', 0.2)
c_mesh.slider("Mesh Size [m]", 0.01, 2.0, def_mesh, 0.01, key="common_mesh_slider")
c_step.slider("Vehicle Step [m]", 0.01, 2.0, def_step, 0.01, key="common_step_slider")

view_case = c4.selectbox("Load Case", view_options, index=v_idx, key="view_case_selector", on_change=set_view_case)

show_sys_mode = "Both"
if view_case != "Vehicle Steps":
    c_sys_tog, _ = st.columns([1,1])
    with c_sys_tog:
        tog_map = {"Both": "Both", "System A": st.session_state['sysA']['name'], "System B": st.session_state['sysB']['name']}
        show_sys_mode = st.radio("Active Systems View", ["Both", "System A", "System B"], format_func=lambda x: tog_map[x], horizontal=True, key="sys_view_toggle")

# MOVED RESULT TYPE SELECTOR HERE
st.write("") # Spacer
curr_res_mode = st.session_state.get('result_mode', "Design (ULS)")
res_opts = ["Design (ULS)", "Characteristic (SLS)", "Characteristic (No Dynamic Factor)"]
try: res_idx = res_opts.index(curr_res_mode)
except: res_idx = 0
st.session_state['result_mode'] = st.radio("Result Type", res_opts, index=res_idx, horizontal=True, key="result_mode_main_ui")
result_mode_val = st.session_state['result_mode']

# Label Checkbox
show_labels = c3.checkbox("Labels", value=True)

# Combine Results (Only if raw results exist)
res_A = solver.combine_results(raw_res_A, st.session_state['sysA'], result_mode_val) if has_res_A else {}
res_B = solver.combine_results(raw_res_B, st.session_state['sysB'], result_mode_val) if has_res_B else {}

if p.get('phi_mode', 'Calculate') == 'Calculate' and has_res_A: # Default to showing A log if avail
    active_raw_res = raw_res_A if curr == 'sysA' and has_res_A else (raw_res_B if has_res_B else {})
    phi_val = active_raw_res.get('phi_calc', 1.0)
    with phi_log_placeholder.container():
        st.markdown(f"**Calculated Phi:** {phi_val:.3f}")
        with st.expander("Phi Calculation Log", expanded=False):
            for log_line in active_raw_res.get('phi_log', []): st.caption(log_line)

# Prepare Plot Data
rA, rB = {}, {}
step_view_sys = "System A"
active_veh_step = "Vehicle A"
veh_key_res = ""

if view_case == "Vehicle Steps":
    st.markdown("---")
    
    # Check if we need a direction toggle
    is_both_active = (st.session_state['sysA']['vehicle_direction'] == 'Both')
    step_dir_suffix = ""
    
    # If "Both" is active, show the toggle
    if is_both_active:
        c_veh_tog, c_dir_tog, c_step_slide, c_step_tog = st.columns([1, 1, 2, 1])
        step_dir_sel = c_dir_tog.radio("Step Direction:", ["Forward", "Reverse"], horizontal=True, key="step_dir_radio")
        if step_dir_sel == "Reverse": step_dir_suffix = "_Rev"
    else:
        # If not "Both", determine if we are in Reverse mode implicitly
        if st.session_state['sysA']['vehicle_direction'] == 'Reverse':
             step_dir_suffix = "_Rev"
        c_veh_tog, c_step_slide, c_step_tog = st.columns([1, 2, 1])
    
    def set_anim_veh(): st.session_state.keep_active_veh_step = st.session_state.anim_veh_radio
    try: av_idx = ["Vehicle A", "Vehicle B"].index(st.session_state.keep_active_veh_step)
    except ValueError: av_idx = 0
    active_veh_step = c_veh_tog.radio("Anim Vehicle:", ["Vehicle A", "Vehicle B"], index=av_idx, horizontal=True, key="anim_veh_radio", on_change=set_anim_veh)
    
    # Construct key based on selection
    base_key = "Vehicle Steps A" if active_veh_step == "Vehicle A" else "Vehicle Steps B"
    veh_key_res = f"{base_key}{step_dir_suffix}"
    
    list_A = res_A.get(veh_key_res, [])
    list_B = res_B.get(veh_key_res, [])
    
    if not list_A and not list_B:
        st.warning(f"No valid steps/vehicle definition found for {active_veh_step} ({'Reverse' if '_Rev' in veh_key_res else 'Forward'}).")
    else:
        max_steps = max(1, len(list_A), len(list_B))
        step_idx = c_step_slide.slider("Step Index", 0, max_steps-1, 0, key="veh_step_slider_persistent")
        
        def set_step_sys(): st.session_state.keep_step_view_sys = st.session_state.step_sys_radio
        try: ss_idx = ["Both", "System A", "System B"].index(st.session_state.keep_step_view_sys)
        except ValueError: ss_idx = 0
        step_tog_map = {"Both": "Both", "System A": st.session_state['sysA']['name'], "System B": st.session_state['sysB']['name']}
        step_view_sys = c_step_tog.radio("View System:", ["Both", "System A", "System B"], index=ss_idx, format_func=lambda x: step_tog_map[x], horizontal=True, key="step_sys_radio", on_change=set_step_sys)
        
        st.markdown("---")
        def get_step(res, idx, k_res, f_factor):
            s_list = res.get(k_res, [])
            if idx < len(s_list):
                step_data = s_list[idx]['res']
                out = {}
                for k, v in step_data.items():
                    # Populate Max/Min keys for step view (Max=Min=Current) to allow unified table generation
                    out[k] = {**v, 
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
        f_B = res_B['f_vehA'] if active_veh_step == "Vehicle A" and has_res_B else 1.0
        # If toggled to B
        if active_veh_step == "Vehicle B":
             f_A = res_A['f_vehB'] if has_res_A else 1.0
             f_B = res_B['f_vehB'] if has_res_B else 1.0
             
        rA = get_step(res_A, step_idx, veh_key_res, f_A)
        rB = get_step(res_B, step_idx, veh_key_res, f_B)
else:
    key_map = {"Total Envelope": "Total Envelope", "Selfweight": "Selfweight", "Soil": "Soil", "Surcharge": "Surcharge", "Vehicle Envelope": "Vehicle Envelope"}
    target_key = key_map.get(view_case, "Total Envelope")
    rA = res_A.get(target_key, {})
    rB = res_B.get(target_key, {})

t1, t2, t3 = st.tabs(["Visualization", "Tabular Data", "Summary"])
name_A = st.session_state['sysA']['name']
name_B = st.session_state['sysB']['name']

with t1:
    if view_case == "Vehicle Steps":
        has_content = len(res_A.get(veh_key_res, [])) > 0 or len(res_B.get(veh_key_res, [])) > 0
        if not has_content: st.info("Visualization unavailable: No valid vehicle steps.")
        else:
            show_A_step = (step_view_sys == "Both" or step_view_sys == "System A")
            show_B_step = (step_view_sys == "Both" or step_view_sys == "System B")
            st.subheader("Bending Moment [kNm]")
            st.plotly_chart(viz.create_plotly_fig(nodes_A, rA, rB, 'M', man_scale, "", show_A_step, show_B_step, show_labels, view_case, name_A, name_B, geom_A=res_A.get('Selfweight'), geom_B=res_B.get('Selfweight')), width='stretch', key="chart_M_step")
            st.subheader("Shear Force [kN]")
            st.plotly_chart(viz.create_plotly_fig(nodes_A, rA, rB, 'V', man_scale, "", show_A_step, show_B_step, show_labels, view_case, name_A, name_B, geom_A=res_A.get('Selfweight'), geom_B=res_B.get('Selfweight')), width='stretch', key="chart_V_step")
            st.subheader("Normal Force [kN]")
            st.plotly_chart(viz.create_plotly_fig(nodes_A, rA, rB, 'N', man_scale, "", show_A_step, show_B_step, show_labels, view_case, name_A, name_B, geom_A=res_A.get('Selfweight'), geom_B=res_B.get('Selfweight')), width='stretch', key="chart_N_step")
            st.subheader("Deformation [mm]")
            st.plotly_chart(viz.create_plotly_fig(nodes_A, rA, rB, 'Def', man_scale, "", show_A_step, show_B_step, show_labels, view_case, name_A, name_B, geom_A=res_A.get('Selfweight'), geom_B=res_B.get('Selfweight')), width='stretch', key="chart_D_step")
    else:
        show_A = (show_sys_mode == "Both" or show_sys_mode == "System A")
        show_B = (show_sys_mode == "Both" or show_sys_mode == "System B")
        # Fix check for None nodes
        geom_invalid_A = (nodes_A is None) or (len(nodes_A)==0)
        geom_invalid_B = (nodes_B is None) or (len(nodes_B)==0)
        
        if geom_invalid_A and geom_invalid_B: 
             st.warning("⚠️ No structural geometry or valid analysis.")
        else:
             if (not rA) and (not rB): st.warning(f"⚠️ No results found for **{view_case}**.")

             st.subheader("Bending Moment [kNm]")
             st.plotly_chart(viz.create_plotly_fig(nodes_A, rA, rB, 'M', man_scale, "", show_A, show_B, show_labels, view_case, name_A, name_B, geom_A=res_A.get('Selfweight'), geom_B=res_B.get('Selfweight')), width='stretch', key="chart_M")
             st.subheader("Shear Force [kN]")
             st.plotly_chart(viz.create_plotly_fig(nodes_A, rA, rB, 'V', man_scale, "", show_A, show_B, show_labels, view_case, name_A, name_B, geom_A=res_A.get('Selfweight'), geom_B=res_B.get('Selfweight')), width='stretch', key="chart_V")
             st.subheader("Normal Force [kN]")
             st.plotly_chart(viz.create_plotly_fig(nodes_A, rA, rB, 'N', man_scale, "", show_A, show_B, show_labels, view_case, name_A, name_B, geom_A=res_A.get('Selfweight'), geom_B=res_B.get('Selfweight')), width='stretch', key="chart_N")
             st.subheader("Deformation [mm]")
             st.plotly_chart(viz.create_plotly_fig(nodes_A, rA, rB, 'Def', man_scale, "", show_A, show_B, show_labels, view_case, name_A, name_B, geom_A=res_A.get('Selfweight'), geom_B=res_B.get('Selfweight')), width='stretch', key="chart_D")

with t2:
    st.markdown(f"### Detailed Data ({view_case})")
    
    detailed_rows = []
    
    def process_detailed(r_dict, sys_name):
        if not r_dict: return
        for eid, data in r_dict.items():
            x_vals = data.get('x', [])
            n_pts = len(x_vals)
            
            # Helper to safely get array or fill with zeros
            def get_arr(key):
                arr = data.get(key)
                if arr is None: return np.zeros(n_pts)
                return arr
            
            # If viewing steps, data structure is simpler (M, V, N)
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
                # Envelopes
                m_max = get_arr('M_max'); m_min = get_arr('M_min')
                v_max = get_arr('V_max'); v_min = get_arr('V_min')
                n_max = get_arr('N_max'); n_min = get_arr('N_min')
                dx_max = get_arr('def_x_max'); dx_min = get_arr('def_x_min')
                dy_max = get_arr('def_y_max'); dy_min = get_arr('def_y_min')
                
                # Fallback for single-case results like SW which might use M instead of M_max
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
    process_detailed(rB, name_B)
    
    if detailed_rows:
        df_detailed = pd.DataFrame(detailed_rows)
        st.dataframe(df_detailed, width='stretch')
        st.download_button(
            "Download Detailed Data (.csv)", 
            df_detailed.to_csv(index=False).encode('utf-8'), 
            f"bricos_detailed_{view_case.replace(' ', '_')}.csv", 
            "text/csv"
        )
    else:
        st.info("No detailed data available for this view.")

with t3:
    st.markdown("### Max/Min Forces per Element")
    def table_gen(res, name):
        d = []
        if not res: return []
        for eid, v in res.items():
            if 'M_max' in v:
                d.append({
                    "Sys": name, "Elem": eid,
                    "M_max": np.max(v['M_max']), "M_min": np.min(v['M_min']),
                    "V_max": np.max(v['V_max']), "V_min": np.min(v['V_min']),
                    "N_max": np.max(v['N_max']), "N_min": np.min(v['N_min']),
                })
            elif 'M' in v:
                d.append({
                    "Sys": name, "Elem": eid,
                    "M_max": np.max(v['M']), "M_min": np.min(v['M']),
                    "V_max": np.max(v['V']), "V_min": np.min(v['V']),
                    "N_max": np.max(v['N']), "N_min": np.min(v['N']),
                })
        return d
    rows = table_gen(rA, name_A) + table_gen(rB, name_B)
    if rows: st.dataframe(pd.DataFrame(rows).round(1))
    else: st.info("No data for this view.")
    
    if view_case != "Vehicle Steps":
        st.markdown("### Total Support Reactions (Global X/Y)")
        def react_table(reacts, nodes, sys_key, display_name):
            d = []
            if not reacts or not nodes: return []
            for nid, f_vec in reacts.items():
                if nid not in nodes: continue
                y_coord = nodes[nid][1]
                is_support = False
                label = ""
                if st.session_state[sys_key]['mode'] == 'Frame':
                    if y_coord < -0.1: is_support = True; wall_idx = nid - 100 + 1; label = f"Wall {wall_idx} Base"
                else:
                    is_support = True; sup_idx = nid - 200 + 1; label = f"Support {sup_idx}"
                if is_support:
                    d.append({"Sys": display_name, "Node": nid, "Loc": label, "Rx (kN)": f_vec[0], "Ry (kN)": f_vec[1], "Mz (kNm)": f_vec[2]})
            return d
        r_rows = react_table(res_A.get('Reactions', {}), nodes_A, "sysA", name_A) + react_table(res_B.get('Reactions', {}), nodes_B, "sysB", name_B)
        st.dataframe(pd.DataFrame(r_rows).round(1))