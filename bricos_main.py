import streamlit as st
import pandas as pd
import numpy as np
import json
import copy
import os
import io # Added for Report Buffer

import bricos_solver as solver
import bricos_viz as viz
import bricos_report # New Report Module

# ==========================================
# UI SETUP & CONFIGURATION
# ==========================================

st.set_page_config(layout="wide", page_title="BriCoS v0.30")

# --- CSS FOR STICKY CONTROLS & LAYOUT ---
# Updated to use :has() selector for robust container targeting
st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 3rem;}
    div[data-testid="stExpander"] div[role="button"] p {font-size: 1rem; font-weight: bold;}\r
    .stSelectbox label { font-size: 0.9rem; font-weight: bold; }
    
    /* Sticky Sidebar Container */
    div[data-testid="stVerticalBlock"]:has(div#sticky-sidebar-marker) {
        position: sticky;
        top: 0rem;
        z-index: 1000;
        background-color: inherit; /* Inherit Blue/Red tint */
        padding-top: 10px;
        padding-bottom: 15px;
        border-bottom: 2px solid rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    
    /* Sticky Results Toolbar (Main Pane) */
    div[data-testid="stVerticalBlock"]:has(div#sticky-results-marker) {
        position: sticky;
        top: 3.75rem; /* Matches Streamlit Header Height */
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
                l_arr = [float(x) for x in l_str.split(',') if x.strip()]
                s_arr = [float(x) for x in s_str.split(',') if x.strip()]
                
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
        'vehicle_direction': 'Forward',

        # Shear Deformations (Issue J)
        'use_shear_def': False,
        'b_eff': 1.0,
        'nu': 0.2
    }

def get_clear(name_suffix, current_mode):
    # Strictly cleared state: 0 Geometry, 1.0 Factors, Manual Phi=1.0
    return {
        'mode': current_mode, 
        'E': 30e6, 'num_spans': 1,
        'L_list': [0.0]*10, 'Is_list': [0.0]*10, 'sw_list': [0.0]*10,
        'h_list': [0.0]*11, 'Iw_list': [0.0]*11,
        'e_mode': 'Manual', # Reset to Manual E to avoid implicit fck calcs
        'fck_span_list': [30.0]*10, 'fck_wall_list': [30.0]*11,
        'E_custom_span': [30.0]*10, 'E_custom_wall': [30.0]*11,
        'E_span_list': [30e6]*10, 'E_wall_list': [30e6]*11,
        
        'supports': [],
        'soil': [],
        'surcharge': [], 
        
        # Cleared Vehicles
        'vehicle': {'loads': [], 'spacing': []},
        'vehicle_loads': "", 'vehicle_space': "",
        'vehicleB': {'loads': [], 'spacing': []},
        'vehicleB_loads': "", 'vehicleB_space': "",
        
        # Unity Factors
        'KFI': 1.0, 
        'gamma_g': 1.0, 'gamma_j': 1.0, 
        'gamma_veh': 1.0, 'gamma_vehB': 1.0, 
        
        # Manual Phi = 1.0
        'phi': 1.0, 
        'phi_mode': 'Manual',
        
        'scale_manual': 2.0, 
        'mesh_size': 0.5, 'step_size': 0.2,
        'name': f"System {name_suffix}",
        'last_mode': current_mode,
        'combine_surcharge_vehicle': False,
        'vehicle_direction': 'Forward',
        
        # Shear Deformations (Issue J)
        'use_shear_def': False,
        'b_eff': 1.0,
        'nu': 0.2
    }

# --- FORCE UPDATE HELPER ---
def force_ui_update(sys_key, data):
    """
    Explicitly synchronizes the Streamlit session_state keys with the provided data dict.
    """
    
    # 1. Main Config Keys
    st.session_state[f"{sys_key}_md_sel"] = data.get('mode', 'Frame')
    st.session_state[f"{sys_key}_emode"] = "Eurocode (f_ck)" if data.get('e_mode') == "Eurocode" else "Manual (E-Modulus)"
    st.session_state[f"{sys_key}_kfi"] = data.get('KFI', 1.0)
    st.session_state[f"{sys_key}_phim"] = data.get('phi_mode', 'Calculate')
    st.session_state[f"{sys_key}_phiv"] = data.get('phi', 1.0)
    st.session_state[f"{sys_key}_nsp"] = data.get('num_spans', 1)
    
    # Shear Deformation Keys
    st.session_state[f"{sys_key}_use_shear"] = data.get('use_shear_def', False)
    st.session_state[f"{sys_key}_beff"] = data.get('b_eff', 1.0)
    st.session_state[f"{sys_key}_nu"] = data.get('nu', 0.2)

    # 2. Factors
    st.session_state[f"{sys_key}_gg_cust"] = data.get('gamma_g', 1.0)
    st.session_state[f"{sys_key}_gj_cust"] = data.get('gamma_j', 1.0)
    st.session_state[f"{sys_key}_gamA_cust"] = data.get('gamma_veh', 1.0)
    st.session_state[f"{sys_key}_gamB_cust"] = data.get('gamma_vehB', 1.0)

    # 3. Spans
    nsp = data.get('num_spans', 1)
    for i in range(10): 
        if i < len(data['L_list']): st.session_state[f"{sys_key}_l{i}"] = data['L_list'][i]
        if i < len(data['Is_list']): st.session_state[f"{sys_key}_i{i}"] = data['Is_list'][i]
        if i < len(data['sw_list']): st.session_state[f"{sys_key}_s{i}"] = data['sw_list'][i]
        if i < len(data['fck_span_list']): st.session_state[f"{sys_key}_fck_s{i}"] = data['fck_span_list'][i]
        if i < len(data['E_custom_span']): st.session_state[f"{sys_key}_Eman_s{i}"] = data['E_custom_span'][i]
        st.session_state[f"{sys_key}_i{i}_dis"] = "See Profiler"

    # 4. Walls
    for i in range(11):
        if i < len(data['h_list']): st.session_state[f"{sys_key}_h{i}"] = data['h_list'][i]
        if i < len(data['Iw_list']): st.session_state[f"{sys_key}_iw{i}"] = data['Iw_list'][i]
        if i < len(data['fck_wall_list']): st.session_state[f"{sys_key}_fck_w{i}"] = data['fck_wall_list'][i]
        if i < len(data['E_custom_wall']): st.session_state[f"{sys_key}_Eman_w{i}"] = data['E_custom_wall'][i]
        
        surch_val = 0.0
        for s in data.get('surcharge', []):
            if s['wall_idx'] == i: surch_val = s['q']
        st.session_state[f"{sys_key}_sq{i}"] = surch_val
        
        hL, qL_b, qL_t = 0.0, 0.0, 0.0
        hR, qR_b, qR_t = 0.0, 0.0, 0.0
        
        for s in data.get('soil', []):
            if s['wall_idx'] == i:
                if s['face'] == 'L':
                    hL = s.get('h', 0.0); qL_b = s.get('q_bot', 0.0); qL_t = s.get('q_top', 0.0)
                elif s['face'] == 'R':
                    hR = s.get('h', 0.0); qR_b = s.get('q_bot', 0.0); qR_t = s.get('q_top', 0.0)
        
        st.session_state[f"{sys_key}_shl{i}"] = hL
        st.session_state[f"{sys_key}_sqlb{i}"] = qL_b
        st.session_state[f"{sys_key}_sqlt{i}"] = qL_t
        st.session_state[f"{sys_key}_shr{i}"] = hR
        st.session_state[f"{sys_key}_sqrb{i}"] = qR_b
        st.session_state[f"{sys_key}_sqrt{i}"] = qR_t

    # 5. Vehicles
    st.session_state[f"{sys_key}_A_loads_input"] = data.get('vehicle_loads', "")
    st.session_state[f"{sys_key}_A_space_input"] = data.get('vehicle_space', "")
    st.session_state[f"{sys_key}_B_loads_input"] = data.get('vehicleB_loads', "")
    st.session_state[f"{sys_key}_B_space_input"] = data.get('vehicleB_space', "")
    
    if f"{sys_key}_vehA_class" in st.session_state: del st.session_state[f"{sys_key}_vehA_class"]
    if f"{sys_key}_vehB_class" in st.session_state: del st.session_state[f"{sys_key}_vehB_class"]
    
    # 6. Supports
    prefix = f"{sys_key}_"
    supp_keys = [k for k in st.session_state.keys() if k.startswith(prefix) and "_k" in k]
    for k in supp_keys: del st.session_state[k]
    
    for i, supp in enumerate(data.get('supports', [])):
        if supp['type'] == 'Custom Spring':
            st.session_state[f"{sys_key}_kx_{i}"] = supp['k'][0]
            st.session_state[f"{sys_key}_ky_{i}"] = supp['k'][1]
            st.session_state[f"{sys_key}_km_{i}"] = supp['k'][2]

def clean_transient_keys():
    """
    Removes temporary UI keys (especially for Profiler and Viz) to prevent 
    stale widget states from overwriting loaded data on re-run.
    """
    keys_to_clear = []
    for k in st.session_state.keys():
        # Clear Profiler transient keys (prof_type, prof_shape, etc.)
        if "_prof_" in k or "_align_t" in k or "_inc_" in k:
            keys_to_clear.append(k)
    
    for k in keys_to_clear:
        del st.session_state[k]

# --- INITIALIZATION WITH SPECIFIC DEFAULTS ---
if 'sysA' not in st.session_state: 
    # System A: 1 Span
    d = get_def()
    d['num_spans'] = 1
    d['name'] = "System A"
    d['soil'] = [
        {'wall_idx': 0, 'face': 'L', 'h': 8.0, 'q_top': 0.0, 'q_bot': 20.0}, 
        {'wall_idx': 0, 'face': 'R', 'h': 4.0, 'q_top': 0.0, 'q_bot': 10.0},
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
        {'wall_idx': 0, 'face': 'L', 'h': 8.0, 'q_top': 0.0, 'q_bot': 20.0},
        {'wall_idx': 0, 'face': 'R', 'h': 4.0, 'q_top': 0.0, 'q_bot': 10.0},
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
    
    # Timoshenko Migration
    if 'use_shear_def' not in st.session_state[sys_k]: st.session_state[sys_k]['use_shear_def'] = False
    if 'b_eff' not in st.session_state[sys_k]: st.session_state[sys_k]['b_eff'] = 1.0
    if 'nu' not in st.session_state[sys_k]: st.session_state[sys_k]['nu'] = 0.2
    
    st.session_state[sys_k].pop('k_rot', None)
    st.session_state[sys_k].pop('k_rot_super', None)

if 'uploader_key' not in st.session_state: st.session_state.uploader_key = 0

if st.session_state['sysA'].get('scale_manual', 0) < 0.1: st.session_state['sysA']['scale_manual'] = 2.0
if st.session_state['sysB'].get('scale_manual', 0) < 0.1: st.session_state['sysB']['scale_manual'] = 2.0

# ---------------------------------------------
# MOVED SECTIONS (TOP OF SIDEBAR)
# ---------------------------------------------

# --- ABOUT SECTION ---
with st.sidebar.expander("About", expanded=False):
    st.markdown("**BriCoS v0.30**")
    st.write("Author: Kasper Lindskov Fabricius")
    st.write("Email: Kasper.LindskovFabricius@sweco.dk")
    st.write("A specialized Finite Element Analysis (FEM) tool for rapid bridge analysis and comparison.")

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
                st.session_state[target_key] = new_data
                force_ui_update(target_key, new_data)

            if mode == "A" or mode == "ALL":
                current_mode = st.session_state['sysA']['mode']
                if action == "clear":
                    data = get_clear("A", current_mode)
                else:
                    data = {**get_def(), 'num_spans':1, 'name': "System A"}
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
                    data = get_clear("B", current_mode)
                else:
                    data = {**get_def(), 'num_spans':2, 'name': "System B"}
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

# --- FILE OPERATIONS ---
with st.sidebar.expander("File Operations (Save/Load)", expanded=False):
    # Initialize report keys if missing to ensure they exist for saving
    rep_keys = ['rep_pno', 'rep_pname', 'rep_rev', 'rep_author', 'rep_check', 'rep_appr', 'rep_comm']
    for rk in rep_keys:
        if rk not in st.session_state: st.session_state[rk] = ""

    def to_csv():
        rows = []
        # Save Systems
        for sys_name in ['sysA', 'sysB']:
            data = st.session_state[sys_name]
            for k, v in data.items():
                if k == 'backup': continue 
                val_str = json.dumps(v, default=str)
                rows.append({'System': sys_name, 'Parameter': k, 'Value': val_str})
        
        # Save Global / Report Settings
        global_keys = rep_keys + ['result_mode']
        for gk in global_keys:
            if gk in st.session_state:
                val_str = json.dumps(st.session_state[gk], default=str)
                rows.append({'System': 'Global', 'Parameter': gk, 'Value': val_str})

        df = pd.DataFrame(rows)
        return df.to_csv(index=False).encode('utf-8')

    st.download_button("Download Configuration (.csv)", to_csv(), "brico_config.csv", "text/csv")

    uploaded_file = st.file_uploader("Upload Configuration (.csv)", type="csv", key=f"uploader_{st.session_state.uploader_key}")
    if uploaded_file is not None:
        try:
            df_load = pd.read_csv(uploaded_file)
            if 'System' in df_load.columns and 'Parameter' in df_load.columns:
                
                # Clean up transient UI keys (Profiler/Viz) to prevent stale state
                clean_transient_keys()
                
                for _, row in df_load.iterrows():
                    sys_n = row['System']
                    try:
                        val = json.loads(row['Value'])
                        if sys_n in ['sysA', 'sysB']:
                            st.session_state[sys_n][row['Parameter']] = val
                        elif sys_n == 'Global':
                            # Restore global/report settings directly to session state
                            st.session_state[row['Parameter']] = val
                    except: pass
                
                force_ui_update('sysA', st.session_state['sysA'])
                force_ui_update('sysB', st.session_state['sysB'])

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
            force_ui_update('sysB', st.session_state['sysB'])
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
            force_ui_update('sysA', st.session_state['sysA'])
            st.session_state.copy_confirm_mode = None
            st.rerun()
        if c_no.button("Cancel"):
            st.session_state.copy_confirm_mode = None
            st.rerun()

# --- ANALYSIS SETTINGS ---
with st.sidebar.expander("Analysis & Result Settings", expanded=False):
    help_dir = "Forward: Left to Right. Reverse: Right to Left (axles inverted). Both: Envelope of both directions."
    curr_dir = st.session_state['sysA'].get('vehicle_direction', 'Forward')
    dir_opts = ["Forward", "Reverse", "Both"]
    idx_dir = dir_opts.index(curr_dir) if curr_dir in dir_opts else 0
    
    dir_sel = st.radio("Vehicle Direction", dir_opts, horizontal=True, index=idx_dir, key="veh_dir_radio_sidebar", help=help_dir)
    st.session_state['sysA']['vehicle_direction'] = dir_sel
    st.session_state['sysB']['vehicle_direction'] = dir_sel
    
    st.markdown("---")
    help_combo = "Define how the Traffic Surcharge (on walls) and the Main Vehicle (on deck) interact.\n- Exclusive: Load is max(Vehicle, Surcharge).\n- Simultaneous: Load is Vehicle + Surcharge."
    
    is_sim = st.session_state['sysA'].get('combine_surcharge_vehicle', False)
    combo_idx = 1 if is_sim else 0
    
    surch_sel = st.radio("Surcharge Combination", ["Exclusive (Vehicle OR Surcharge)", "Simultaneous (Vehicle + Surcharge)"], index=combo_idx, horizontal=True, key="surcharge_combo_radio_sidebar", help=help_combo)
    
    is_simultaneous = (surch_sel == "Simultaneous (Vehicle + Surcharge)")
    st.session_state['sysA']['combine_surcharge_vehicle'] = is_simultaneous
    st.session_state['sysB']['combine_surcharge_vehicle'] = is_simultaneous

    st.markdown("---")
    # --- SHEAR DEFORMATION SETTINGS (Issue J) ---
    st.markdown("**Shear Deformations (Timoshenko)**")
    
    help_shear = "Enables shear deformation consideration in the stiffness matrix. Recommended for deep beams and piers."
    
    # We use System A's value as the 'driver' for the shared sidebar control, but we must update both.
    # To handle potential desync, we prefer a single widget updating both.
    use_shear = st.checkbox("Enable Shear Deformations", value=st.session_state['sysA'].get('use_shear_def', False), key="shear_toggle_sidebar", help=help_shear)
    
    st.session_state['sysA']['use_shear_def'] = use_shear
    st.session_state['sysB']['use_shear_def'] = use_shear

    col_beff, col_nu = st.columns(2)
    
    help_beff = "Effective shear width [m]. Typically web thickness or width of rectangular section. Affects Shear Area (As = 5/6 * b_eff * h)."
    help_nu = "Poisson's Ratio (ν). Used to calculate Shear Modulus G = E / (2*(1+ν))."
    
    # Shared input values from Sys A
    val_beff = st.session_state['sysA'].get('b_eff', 1.0)
    val_nu = st.session_state['sysA'].get('nu', 0.2)
    
    new_beff = col_beff.number_input("Effective Width (b_eff) [m]", value=float(val_beff), min_value=0.01, step=0.1, help=help_beff, key="beff_input_sidebar")
    new_nu = col_nu.number_input("Poisson's Ratio (ν)", value=float(val_nu), min_value=0.0, max_value=0.5, step=0.05, help=help_nu, key="nu_input_sidebar")
    
    st.session_state['sysA']['b_eff'] = new_beff
    st.session_state['sysB']['b_eff'] = new_beff
    st.session_state['sysA']['nu'] = new_nu
    st.session_state['sysB']['nu'] = new_nu

    st.markdown("---")
    st.markdown("**Calculation Precision**")
    c_mesh, c_step = st.columns(2)
    def_mesh = st.session_state['sysA'].get('mesh_size', 0.5)
    def_step = st.session_state['sysA'].get('step_size', 0.2)
    m_val = c_mesh.slider("Mesh Size [m]", 0.01, 2.0, def_mesh, 0.01, key="common_mesh_slider")
    s_val = c_step.slider("Vehicle Step [m]", 0.01, 2.0, def_step, 0.01, key="common_step_slider")

if "common_mesh_slider" in st.session_state:
    st.session_state['sysA']['mesh_size'] = m_val
    st.session_state['sysB']['mesh_size'] = m_val

if "common_step_slider" in st.session_state:
    st.session_state['sysA']['step_size'] = s_val
    st.session_state['sysB']['step_size'] = s_val

# --- REPORT GENERATION (NEW) ---
with st.sidebar.expander("Report Generation", expanded=False):
    # Initialize report keys if missing
    if 'rep_pno' not in st.session_state: st.session_state.rep_pno = ""
    if 'rep_pname' not in st.session_state: st.session_state.rep_pname = ""
    if 'rep_rev' not in st.session_state: st.session_state.rep_rev = "A"
    if 'rep_author' not in st.session_state: st.session_state.rep_author = ""
    if 'rep_check' not in st.session_state: st.session_state.rep_check = ""
    if 'rep_appr' not in st.session_state: st.session_state.rep_appr = ""
    if 'rep_comm' not in st.session_state: st.session_state.rep_comm = ""

    # Corrected: Only using key= to bind, removing conflicting assignment
    st.text_input("Project No.", key="rep_pno")
    st.text_input("Project Name", key="rep_pname")
    
    c_r1, c_r2 = st.columns(2)
    c_r1.text_input("Revision", key="rep_rev")
    c_r2.text_input("Author", key="rep_author")
    
    c_r3, c_r4 = st.columns(2)
    c_r3.text_input("Checker", key="rep_check")
    c_r4.text_input("Approver", key="rep_appr")
    
    st.text_area("Comments", height=100, key="rep_comm")
    
    if st.button("Generate PDF Report", type="primary"):
        with st.spinner("Rendering Report (this may take a moment)..."):
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
            try:
                rep_gen = bricos_report.BricosReportGenerator(buffer, meta, st.session_state)
                rep_gen.generate()
                buffer.seek(0)
                st.session_state['report_buffer'] = buffer
                st.success("Report Generated!")
            except Exception as e:
                st.error(f"Report Generation Failed: {e}")
    
    if 'report_buffer' in st.session_state:
        st.download_button(
            label="Download Report PDF",
            data=st.session_state['report_buffer'],
            file_name=f"BriCoS_Report_{st.session_state.rep_pno}.pdf",
            mime="application/pdf"
        )

# ---------------------------------------------
# STICKY SIDEBAR: ACTIVE SYSTEM & NAMES
# ---------------------------------------------
st.sidebar.header("Configuration")

with st.sidebar.container():
    st.markdown('<div id="sticky-sidebar-marker"></div>', unsafe_allow_html=True)
    c_nA, c_nB = st.columns(2)
    st.session_state['sysA']['name'] = c_nA.text_input("Name Sys A", st.session_state['sysA']['name'])
    st.session_state['sysB']['name'] = c_nB.text_input("Name Sys B", st.session_state['sysB']['name'])

    sys_map = {"sysA": f"{st.session_state['sysA']['name']} (Blue)", "sysB": f"{st.session_state['sysB']['name']} (Red)"}
    active_sys_key = st.radio("Active System:", ["sysA", "sysB"], format_func=lambda x: sys_map[x], horizontal=True)

    if active_sys_key == 'sysA':
        st.markdown("""<style>[data-testid="stSidebar"] { background-color: #F0F8FF; }</style>""", unsafe_allow_html=True)
    else:
        st.markdown("""<style>[data-testid="stSidebar"] { background-color: #FFF5F5; }</style>""", unsafe_allow_html=True)

curr = active_sys_key
p = st.session_state[curr]

# --- MAIN INPUTS (REMAINING) ---
with st.sidebar.expander("Design Factors & Type", expanded=False):
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
    kfi_opts = [0.9, 1.0, 1.1]
    curr_kfi = p.get('KFI', 1.0)
    idx_kfi = kfi_opts.index(curr_kfi) if curr_kfi in kfi_opts else 1
    p['KFI'] = st.selectbox("KFI (Consequence Class)", kfi_opts, index=idx_kfi, key=f"{curr}_kfi", help=help_kfi)
    
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

with st.sidebar.expander("Geometry, Stiffness & Static Loads", expanded=False):
    n_spans = st.number_input("Number of Spans", 1, 10, p['num_spans'], key=f"{curr}_nsp")
    p['num_spans'] = n_spans
    
    is_ec = (p['e_mode'] == 'Eurocode')
    lbl_mat = "f_ck [MPa]" if is_ec else "E [GPa]"
    help_mat_col = "Concrete Cylinder Strength (f_ck)." if is_ec else "Elastic Modulus (Young's Modulus)."
    
    st.markdown("---")
    st.markdown("**Spans (L, I, SW, Material)**")
    
    while len(p['fck_span_list']) < 10: p['fck_span_list'].append(35.0)
    while len(p['E_custom_span']) < 10: p['E_custom_span'].append(34.0)
    while len(p['E_span_list']) < 10: p['E_span_list'].append(34e6)

    def get_geom_ui_data(prefix_key, i, default_val):
        key = f"{prefix_key}_{i}"
        if key not in p:
            p[key] = {
                'type': 0, 'shape': 0, 'vals': [default_val, default_val, default_val],
                'align_type': 0, 'incline_mode': 0, 'incline_val': 0.0
            }
        return p[key]

    for i in range(n_spans):
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        p['L_list'][i] = c1.number_input(f"L{i+1} [m]", value=float(p['L_list'][i]), key=f"{curr}_l{i}", help="Length of Span" if i==0 else None)
        
        s_geom = get_geom_ui_data('span_geom', i, p['Is_list'][i])
        is_adv = (s_geom['shape'] != 0) or (s_geom['type'] != 0) or (s_geom.get('align_type', 0) != 0)
        
        if not is_adv:
            val = c2.number_input(f"I{i+1} [m⁴]", value=float(p['Is_list'][i]), format="%.4f", key=f"{curr}_i{i}", help="Inertia" if i==0 else None)
            p['Is_list'][i] = val
            s_geom['vals'] = [val, val, val]
        else:
            c2.text_input(f"I{i+1}", "See Profiler", disabled=True, key=f"{curr}_i{i}_dis")

        p['sw_list'][i] = c3.number_input(f"SW{i+1} [kN/m]", value=float(p['sw_list'][i]), key=f"{curr}_s{i}", help="Distributed load from permanent dead loads." if i==0 else None)
        
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
        new_q = c3.number_input(f"Lat. Load [kN/m]", value=float(val_q), disabled=is_super, key=f"{curr}_sq{i}", help="Vehicle Surcharge / Lateral Load" if i==0 else None)
        
        if not is_super:
            p['surcharge'] = [x for x in p['surcharge'] if x['wall_idx'] != i]
            if new_q != 0: 
                p['surcharge'].append({'wall_idx':i, 'face':'R', 'q':new_q, 'h':p['h_list'][i]})

        if is_ec:
            val_in = c4.number_input(f"{lbl_mat}", value=float(p['fck_wall_list'][i]), disabled=is_super, key=f"{curr}_fck_w{i}", help=help_mat_col if i==0 else None)
            p['fck_wall_list'][i] = val_in
            E_gpa = 22.0 * ((val_in + 8)/10.0)**0.3
            p['E_wall_list'][i] = E_gpa * 1e6
        else:
            val_in = c4.number_input(f"{lbl_mat}", value=float(p['E_custom_wall'][i]), disabled=is_super, key=f"{curr}_Eman_w{i}", help=help_mat_col if i==0 else None)
            p['E_custom_wall'][i] = val_in
            p['E_wall_list'][i] = val_in * 1e6

        ex_SoilLeft = next((x for x in p['soil'] if x['wall_idx']==i and x['face']=='L'), None)
        ex_SoilRight = next((x for x in p['soil'] if x['wall_idx']==i and x['face']=='R'), None)
        
        c_sl, c_sr = st.columns(2)
        
        h_L = c_sl.number_input("H_soil_left [m]", value=ex_SoilLeft['h'] if ex_SoilLeft else 0.0, disabled=is_super, key=f"{curr}_shl{i}", help="Soil on Left Face (Pushing Right)")
        qL_bot = c_sl.number_input("q_bot [kN/m²]", value=ex_SoilLeft['q_bot'] if ex_SoilLeft else 0.0, disabled=is_super, key=f"{curr}_sqlb{i}")
        qL_top = c_sl.number_input("q_top [kN/m²]", value=ex_SoilLeft['q_top'] if ex_SoilLeft else 0.0, disabled=is_super, key=f"{curr}_sqlt{i}")
        
        h_R = c_sr.number_input("H_soil_right [m]", value=ex_SoilRight['h'] if ex_SoilRight else 0.0, disabled=is_super, key=f"{curr}_shr{i}", help="Soil on Right Face (Pushing Left)")
        qR_bot = c_sr.number_input("q_bot [kN/m²]", value=ex_SoilRight['q_bot'] if ex_SoilRight else 0.0, disabled=is_super, key=f"{curr}_sqrb{i}")
        qR_top = c_sr.number_input("q_top [kN/m²]", value=ex_SoilRight['q_top'] if ex_SoilRight else 0.0, disabled=is_super, key=f"{curr}_sqrt{i}")

        if not is_super:
            p['soil'] = [x for x in p['soil'] if x['wall_idx']!=i]
            if h_L > 0: p['soil'].append({'wall_idx':i, 'face':'L', 'q_bot':qL_bot, 'q_top':qL_top, 'h':h_L})
            if h_R > 0: p['soil'].append({'wall_idx':i, 'face':'R', 'q_bot':qR_bot, 'q_top':qR_top, 'h':h_R})

    st.markdown("---")
    with st.sidebar.expander("🛠️ Section Profiler (Advanced)", expanded=False):
        st.caption("Configure variable stiffness, height profiles, or vertical alignment.")
        
        elem_options = [f"Span {i+1}" for i in range(n_spans)] + ([f"Wall {i+1}" for i in range(n_spans+1)] if not is_super else [])
        sel_el = st.selectbox("Edit Element:", elem_options, key=f"{curr}_prof_sel")
        
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
        # ISSUE K FIX: Appending sel_el to keys to ensure widget state refreshes on selection change
        new_type = c_p1.radio("Definition Mode:", ["Inertia (I)", "Height (H)"], index=target_geom['type'], key=f"{curr}_prof_type_{sel_el}", horizontal=True)
        target_geom['type'] = 0 if "Inertia" in new_type else 1
        
        new_shape = c_p2.radio("Profile Shape:", ["Constant", "Linear (Taper)", "3-Point (Start/Mid/End)"], index=target_geom['shape'], key=f"{curr}_prof_shape_{sel_el}", horizontal=True)
        shape_map = {"Constant": 0, "Linear (Taper)": 1, "3-Point (Start/Mid/End)": 2}
        target_geom['shape'] = shape_map[new_shape]
        
        vals = target_geom['vals']
        c_v1, c_v2, c_v3 = st.columns(3)
        
        lbl_v = "I [m⁴]" if target_geom['type']==0 else "H [m]"
        v1 = c_v1.number_input(f"Start {lbl_v}", value=float(vals[0]), format="%.4f", key=f"{curr}_prof_v1_{sel_el}")
        
        v2 = vals[1]
        if target_geom['shape'] == 2:
            v2 = c_v2.number_input(f"Mid {lbl_v}", value=float(vals[1]), format="%.4f", key=f"{curr}_prof_v2_{sel_el}")
        
        v3 = vals[2]
        if target_geom['shape'] >= 1:
            v3 = c_v3.number_input(f"End {lbl_v}", value=float(vals[2]), format="%.4f", key=f"{curr}_prof_v3_{sel_el}")
            
        target_geom['vals'] = [v1, v2, v3]
        
        target_simple_list[idx] = v1 if target_geom['type']==0 else (1.0 * v1**3)/12.0

        if is_span_selected:
            st.markdown("#### 📐 Alignment (Vertical Geometry)")
            if 'align_type' not in target_geom: target_geom['align_type'] = 0
            if 'incline_mode' not in target_geom: target_geom['incline_mode'] = 0
            if 'incline_val' not in target_geom: target_geom['incline_val'] = 0.0

            al_opts = ["Straight (Horizontal)", "Inclined"]
            new_align = st.radio("Span Profile:", al_opts, index=target_geom['align_type'], horizontal=True, key=f"{curr}_align_t_{sel_el}")
            target_geom['align_type'] = al_opts.index(new_align)
            
            if target_geom['align_type'] == 1:
                inc_opts = ["Slope (%)", "Delta Height (End - Start) [m]"]
                new_inc_mode = st.radio("Define Inclination by:", inc_opts, index=target_geom['incline_mode'], horizontal=True, key=f"{curr}_inc_m_{sel_el}")
                target_geom['incline_mode'] = inc_opts.index(new_inc_mode)
                
                lbl_inc = "Slope [%]" if target_geom['incline_mode'] == 0 else "Delta H [m]"
                help_inc = "Positive slope/height goes UP. Negative goes DOWN."
                target_geom['incline_val'] = st.number_input(lbl_inc, value=float(target_geom['incline_val']), format="%.2f", help=help_inc, key=f"{curr}_inc_v_{sel_el}")


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
        "Fixed": [1e14, 1e14, 1e14],
        "Pinned": [1e14, 1e14, 0.0],
        "Roller (X-Free)": [0.0, 1e14, 0.0],
        "Roller (Y-Free)": [1e14, 0.0, 0.0],
        "Custom Spring": None
    }
    
    for i in range(num_supports):
        supp_name = f"Wall {i+1} Base" if p['mode'] == 'Frame' else f"Support {i+1}"
        st.markdown(f"**{supp_name}**")
        
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

with st.sidebar.expander("Vehicle Definitions", expanded=False):
    veh_options, veh_data = get_vehicle_library()
    
    def handle_veh_inputs(prefix, key_class, key_loads, key_space, struct_key):
        sess_key = f"{curr}_{prefix}_class"
        default_class = "Class 100" if prefix == "A" else "Custom"
        if sess_key not in st.session_state: st.session_state[sess_key] = default_class
        
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
                # Safe Parsing to prevent crashes on trailing commas
                l_arr = [float(x) for x in p[key_loads].split(',') if x.strip()]
                s_arr = [float(x) for x in p[key_space].split(',') if x.strip()]
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

# --- EXECUTION & RESULTS ---
view_options = ["Total Envelope", "Selfweight", "Soil", "Surcharge", "Vehicle Envelope", "Vehicle Steps"]
def set_view_case(): st.session_state.keep_view_case = st.session_state.view_case_selector
try: v_idx = view_options.index(st.session_state.keep_view_case)
except ValueError: v_idx = 0

if 'result_mode' not in st.session_state: st.session_state['result_mode'] = "Design (ULS)"

def safe_solve(params):
    try:
        return solver.run_raw_analysis(params)
    except ValueError as e:
        return None, None, str(e)

raw_res_A, nodes_A, err_A = safe_solve(st.session_state['sysA'])
raw_res_B, nodes_B, err_B = safe_solve(st.session_state['sysB'])

if err_A and isinstance(err_A, str): st.error(f"System A Error: {err_A}")
if err_B and isinstance(err_B, str): st.error(f"System B Error: {err_B}")

has_res_A = (raw_res_A is not None) and (nodes_A is not None)
has_res_B = (raw_res_B is not None) and (nodes_B is not None)

# --- VISUAL CONTROL SETTINGS ---
# Unified 2-row layout as requested

r1_col1, r1_col2 = st.columns([3, 1])
with r1_col1:
    man_scale = st.slider("Target Diagram Height [m]", 0.5, 10.0, float(st.session_state['sysA'].get('scale_manual', 2.0)), 0.1)
with r1_col2:
    show_labels = st.checkbox("Labels", value=True)

r2_col1, r2_col2 = st.columns([3, 1])
with r2_col1:
    support_size = st.slider("Support Size", 0.1, 2.0, 0.5, 0.1)
with r2_col2:
    show_supports = st.checkbox("Show Supports", value=True)

# Update Session State
st.session_state['sysA']['scale_manual'] = man_scale
st.session_state['sysB']['scale_manual'] = man_scale

# --- STICKY RESULTS TOOLBAR ---
with st.container():
    st.markdown('<div id="sticky-results-marker"></div>', unsafe_allow_html=True)
    c_res_tool1, c_res_tool2, c_res_tool3 = st.columns([2, 2, 2])

    view_case = c_res_tool1.selectbox("Load Case", view_options, index=v_idx, key="view_case_selector", on_change=set_view_case)

    show_sys_mode = "Both"
    if view_case != "Vehicle Steps":
        tog_map = {"Both": "Both", "System A": st.session_state['sysA']['name'], "System B": st.session_state['sysB']['name']}
        show_sys_mode = c_res_tool2.radio("Active Systems View", ["Both", "System A", "System B"], format_func=lambda x: tog_map[x], horizontal=True, key="sys_view_toggle")

    curr_res_mode = st.session_state.get('result_mode', "Design (ULS)")
    res_opts = ["Design (ULS)", "Characteristic (SLS)", "Characteristic (No Dynamic Factor)"]
    try: res_idx = res_opts.index(curr_res_mode)
    except: res_idx = 0
    st.session_state['result_mode'] = c_res_tool3.radio("Result Type", res_opts, index=res_idx, horizontal=True, key="result_mode_main_ui")
    result_mode_val = st.session_state['result_mode']

# -----------------------------

res_A = solver.combine_results(raw_res_A, st.session_state['sysA'], result_mode_val) if has_res_A else {}
res_B = solver.combine_results(raw_res_B, st.session_state['sysB'], result_mode_val) if has_res_B else {}

if p.get('phi_mode', 'Calculate') == 'Calculate' and has_res_A:
    active_raw_res = raw_res_A if curr == 'sysA' and has_res_A else (raw_res_B if has_res_B else {})
    phi_val = active_raw_res.get('phi_calc', 1.0)
    with phi_log_placeholder.container():
        st.markdown(f"**Calculated Phi:** {phi_val:.3f}")
        with st.expander("Phi Calculation Log", expanded=False):
            for log_line in active_raw_res.get('phi_log', []): st.caption(log_line)

rA, rB = {}, {}
step_view_sys = "System A"
active_veh_step = "Vehicle A"
veh_key_res = ""

if view_case == "Vehicle Steps":
    st.markdown("---")
    
    # ISSUE H FIX: Only show Reverse option if available
    is_both_active = (st.session_state['sysA']['vehicle_direction'] == 'Both')
    is_reverse_only = (st.session_state['sysA']['vehicle_direction'] == 'Reverse')
    step_dir_suffix = ""
    
    if is_both_active:
        c_veh_tog, c_dir_tog, c_step_slide, c_step_tog = st.columns([1, 1, 2, 1])
        step_dir_sel = c_dir_tog.radio("Step Direction:", ["Forward", "Reverse"], horizontal=True, key="step_dir_radio")
        if step_dir_sel == "Reverse": step_dir_suffix = "_Rev"
    elif is_reverse_only:
        c_veh_tog, c_step_slide, c_step_tog = st.columns([1, 2, 1])
        step_dir_suffix = "_Rev"
    else:
        # Forward Only
        c_veh_tog, c_step_slide, c_step_tog = st.columns([1, 2, 1])
        # step_dir_suffix remains ""
    
    def set_anim_veh(): st.session_state.keep_active_veh_step = st.session_state.anim_veh_radio
    try: av_idx = ["Vehicle A", "Vehicle B"].index(st.session_state.keep_active_veh_step)
    except ValueError: av_idx = 0
    active_veh_step = c_veh_tog.radio("Anim Vehicle:", ["Vehicle A", "Vehicle B"], index=av_idx, horizontal=True, key="anim_veh_radio", on_change=set_anim_veh)
    
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
            st.plotly_chart(viz.create_plotly_fig(
                nodes_A, rA, rB, 'M', man_scale, "", 
                show_A_step, show_B_step, show_labels, view_case, 
                name_A, name_B, 
                geom_A=res_A.get('Selfweight'), geom_B=res_B.get('Selfweight'),
                params_A=st.session_state['sysA'], params_B=st.session_state['sysB'],
                show_supports=show_supports, support_size=support_size
            ), width='stretch', key="chart_M_step")
            
            st.subheader("Shear Force [kN]")
            st.plotly_chart(viz.create_plotly_fig(
                nodes_A, rA, rB, 'V', man_scale, "", 
                show_A_step, show_B_step, show_labels, view_case, 
                name_A, name_B, 
                geom_A=res_A.get('Selfweight'), geom_B=res_B.get('Selfweight'),
                params_A=st.session_state['sysA'], params_B=st.session_state['sysB'],
                show_supports=show_supports, support_size=support_size
            ), width='stretch', key="chart_V_step")
            
            st.subheader("Normal Force [kN]")
            st.plotly_chart(viz.create_plotly_fig(
                nodes_A, rA, rB, 'N', man_scale, "", 
                show_A_step, show_B_step, show_labels, view_case, 
                name_A, name_B, 
                geom_A=res_A.get('Selfweight'), geom_B=res_B.get('Selfweight'),
                params_A=st.session_state['sysA'], params_B=st.session_state['sysB'],
                show_supports=show_supports, support_size=support_size
            ), width='stretch', key="chart_N_step")
            
            st.subheader("Deformation [mm]")
            st.plotly_chart(viz.create_plotly_fig(
                nodes_A, rA, rB, 'Def', man_scale, "", 
                show_A_step, show_B_step, show_labels, view_case, 
                name_A, name_B, 
                geom_A=res_A.get('Selfweight'), geom_B=res_B.get('Selfweight'),
                params_A=st.session_state['sysA'], params_B=st.session_state['sysB'],
                show_supports=show_supports, support_size=support_size
            ), width='stretch', key="chart_D_step")
    else:
        show_A = (show_sys_mode == "Both" or show_sys_mode == "System A")
        show_B = (show_sys_mode == "Both" or show_sys_mode == "System B")
        
        geom_invalid_A = (nodes_A is None) or (len(nodes_A)==0)
        geom_invalid_B = (nodes_B is None) or (len(nodes_B)==0)
        
        if geom_invalid_A and geom_invalid_B: 
             st.warning("⚠️ No structural geometry defined. Please configure Spans/Walls in the sidebar.")
        else:
             if (not rA) and (not rB): st.warning(f"⚠️ No results found for **{view_case}**.")

             st.subheader("Bending Moment [kNm]")
             st.plotly_chart(viz.create_plotly_fig(
                 nodes_A, rA, rB, 'M', man_scale, "", 
                 show_A, show_B, show_labels, view_case, 
                 name_A, name_B, 
                 geom_A=res_A.get('Selfweight'), geom_B=res_B.get('Selfweight'),
                 params_A=st.session_state['sysA'], params_B=st.session_state['sysB'],
                 show_supports=show_supports, support_size=support_size
             ), width='stretch', key="chart_M")
             
             st.subheader("Shear Force [kN]")
             st.plotly_chart(viz.create_plotly_fig(
                 nodes_A, rA, rB, 'V', man_scale, "", 
                 show_A, show_B, show_labels, view_case, 
                 name_A, name_B, 
                 geom_A=res_A.get('Selfweight'), geom_B=res_B.get('Selfweight'),
                 params_A=st.session_state['sysA'], params_B=st.session_state['sysB'],
                 show_supports=show_supports, support_size=support_size
             ), width='stretch', key="chart_V")
             
             st.subheader("Normal Force [kN]")
             st.plotly_chart(viz.create_plotly_fig(
                 nodes_A, rA, rB, 'N', man_scale, "", 
                 show_A, show_B, show_labels, view_case, 
                 name_A, name_B, 
                 geom_A=res_A.get('Selfweight'), geom_B=res_B.get('Selfweight'),
                 params_A=st.session_state['sysA'], params_B=st.session_state['sysB'],
                 show_supports=show_supports, support_size=support_size
             ), width='stretch', key="chart_N")
             
             st.subheader("Deformation [mm]")
             st.plotly_chart(viz.create_plotly_fig(
                 nodes_A, rA, rB, 'Def', man_scale, "", 
                 show_A, show_B, show_labels, view_case, 
                 name_A, name_B, 
                 geom_A=res_A.get('Selfweight'), geom_B=res_B.get('Selfweight'),
                 params_A=st.session_state['sysA'], params_B=st.session_state['sysB'],
                 show_supports=show_supports, support_size=support_size
             ), width='stretch', key="chart_D")

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
    st.subheader(f"Comparsion Summary ({view_case})")
    
    # ----------------------------------------------
    # 1. HELPER FUNCTIONS
    # ----------------------------------------------
    def get_peaks(r_dict, key_max, key_min):
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

    # Updated Algebraic Comparison Logic
    def calc_diff(val_a, val_b, is_max_case=True):
        if val_a is None or val_b is None: return np.nan
        
        # Guard for zero division
        denom = abs(val_a)
        if denom < 1e-6:
            if abs(val_b) < 1e-6: return 0.0
            return 9999.0 # Placeholder for Infinity
        
        if is_max_case:
            # For MAX: Algebraic Increase = Red.
            # (B - A) > 0 -> Increase -> Red
            # (B - A) < 0 -> Decrease -> Green
            diff = (val_b - val_a)
            return (diff / denom) * 100.0
        else:
            # For MIN: Algebraic Decrease (More Negative) = Red.
            # (A - B) > 0 => A > B => B is smaller/more negative -> Red
            diff = (val_a - val_b)
            return (diff / denom) * 100.0

    # Styling: Positive = Red (Worse), Negative = Green (Better)
    def color_diff(val):
        if pd.isna(val): return ""
        if val > 0.05: return 'color: red; font-weight: bold' 
        if val < -0.05: return 'color: green; font-weight: bold' 
        return 'color: gray'

    # FORMATTING FUNCTION FOR CAP
    def fmt_pct_cap(val):
        if pd.isna(val): return "--"
        # Defensive check: ensure val is numeric before comparing
        if not isinstance(val, (int, float)): return str(val)
        
        if val > 999.0: return ">999%"
        if val < -999.0: return "<-999%"
        return "{:+.1f}%".format(val)

    all_elems = sorted(list(set(rA.keys()) | set(rB.keys())), key=lambda x: (x[0], int(x[1:])))

    # ----------------------------------------------
    # 2. SEPARATE TABLES GENERATION
    # ----------------------------------------------
    
    def render_summary_table(title, metrics_list, elem_filter=None):
        st.markdown(f"##### {title}")
        rows = []
        
        for eid in all_elems:
            # Apply Element Filter (e.g., only Spans for Def Y)
            if elem_filter:
                if elem_filter == "Span" and not eid.startswith("S"): continue
                if elem_filter == "Wall" and not eid.startswith("W"): continue
                
            row_dat = {"Element": eid}
            dataA = rA.get(eid, {})
            dataB = rB.get(eid, {})
            
            # If element not in system, skip row for that system? No, keep comparison as --
            
            for k_max, k_min, label in metrics_list:
                is_def = "def" in k_max
                scale = 1000.0 if is_def else 1.0
                
                a_mx, a_mn = get_peaks(dataA, k_max, k_min)
                if a_mx is not None: a_mx *= scale; a_mn *= scale
                
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
            
            rows.append(row_dat)
            
        if not rows:
            st.caption("No elements found.")
            return

        df = pd.DataFrame(rows)
        pct_cols = [c for c in df.columns if "%" in c]
        
        # Apply formatting ONLY to percentage columns using subset
        st.dataframe(
            df.style.map(color_diff, subset=pct_cols).format(fmt_pct_cap, subset=pct_cols, na_rep="--"),
            height=200, width='stretch'
        )

    # A. Bending Moment
    render_summary_table("Bending Moment", [("M_max", "M_min", "M [kNm]")])
    
    # B. Shear Force
    render_summary_table("Shear Force", [("V_max", "V_min", "V [kN]")])
    
    # C. Normal Force
    render_summary_table("Normal Force", [("N_max", "N_min", "N [kN]")])
    
    # D. Deformations (Special Filtering)
    # Combined Table strategy: We iterate elements and pick correct def type
    st.markdown("##### Deformations (Spans: Vertical, Walls: Horizontal)")
    def_rows = []
    for eid in all_elems:
        row_dat = {"Element": eid}
        dataA = rA.get(eid, {})
        dataB = rB.get(eid, {})
        
        # Determine Type
        is_wall = eid.startswith("W")
        k_max = "def_x_max" if is_wall else "def_y_max"
        k_min = "def_x_min" if is_wall else "def_y_min"
        label = "Def X [mm]" if is_wall else "Def Y [mm]"
        
        a_mx, a_mn = get_peaks(dataA, k_max, k_min)
        if a_mx is not None: a_mx *= 1000; a_mn *= 1000
        
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
        
        row_dat["Type"] = "Wall (Horiz)" if is_wall else "Span (Vert)"
        def_rows.append(row_dat)
        
    if def_rows:
        df_def = pd.DataFrame(def_rows)
        # Reorder to put Type second
        cols = ['Element', 'Type'] + [c for c in df_def.columns if c not in ['Element', 'Type']]
        df_def = df_def[cols]
        
        pct_cols_d = [c for c in df_def.columns if "%" in c]
        # Apply formatting ONLY to percentage columns using subset
        st.dataframe(
            df_def.style.map(color_diff, subset=pct_cols_d).format(fmt_pct_cap, subset=pct_cols_d, na_rep="--"),
            height=200, width='stretch'
        )
    
    # ----------------------------------------------
    # 3. REACTIONS
    # ----------------------------------------------
    st.markdown("##### Envelope Support Reactions")

    def get_reaction_envelope(res_dict, nodes_dict):
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
            
            # Start Node
            def get_val(key, idx):
                if key in dat: return dat[key][idx] 
                elif key.replace("_max","") in dat: 
                     return dat[key.replace("_max","")][idx]
                return 0.0

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
            is_supp_start = (y_start < -0.01) if p['mode'] == 'Frame' else (dat['ni_id'] >= 200) 
            if is_supp_start: add_to_node(dat['ni_id'], fx_mx, fx_mn, fy_mx, fy_mn, m_mx, m_mn)

            # End Node
            n_mx = get_val('N_max', -1); n_mn = get_val('N_min', -1)
            v_mx = get_val('V_max', -1); v_mn = get_val('V_min', -1)
            m_mx = get_val('M_max', -1); m_mn = get_val('M_min', -1)
            
            n_mx, n_mn = -n_mn, -n_mx 
            v_mx, v_mn = -v_mn, -v_mx
            m_mx, m_mn = -m_mn, -m_mx
            
            fx_mx, fx_mn = get_bounds(c, s)
            fy_mx, fy_mn = get_bounds(s, -c)
            
            y_end = nodes_dict[dat['nj_id']][1]
            is_supp_end = (y_end < -0.01) if p['mode'] == 'Frame' else (dat['nj_id'] >= 200)
            if is_supp_end: add_to_node(dat['nj_id'], fx_mx, fx_mn, fy_mx, fy_mn, m_mx, m_mn)
                 
        return reacts

    reactsA = get_reaction_envelope(rA, nodes_A)
    reactsB = get_reaction_envelope(rB, nodes_B)
    
    all_react_nodes = sorted(list(set(reactsA.keys()) | set(reactsB.keys())))
    r_rows = []
    
    for nid in all_react_nodes:
        label = f"Node {nid}"
        if nid >= 200: label = f"Support {nid-200+1}"
        elif nid >= 100: label = f"Wall {nid-100+1} Base"
        
        row = {"Location": label}
        dA = reactsA.get(nid, {})
        dB = reactsB.get(nid, {})
        
        for comp in ['Rx', 'Ry', 'Mz']:
            for bnd in ['max', 'min']:
                key = f"{comp}_{bnd}"
                valA = dA.get(key)
                valB = dB.get(key)
                
                col_A = f"{comp} ({bnd}) A"
                col_B = f"{comp} ({bnd}) B"
                col_P = f"{comp} ({bnd}) %"
                
                row[col_A] = f"{valA:.1f}" if valA is not None else "--"
                row[col_B] = f"{valB:.1f}" if valB is not None else "--"
                row[col_P] = calc_diff(valA, valB, is_max_case=(bnd=='max'))
        
        r_rows.append(row)
    
    if r_rows:
        df_react = pd.DataFrame(r_rows)
        pct_cols_r = [c for c in df_react.columns if "%" in c]
            
        # Apply formatting ONLY to percentage columns using subset
        st.dataframe(
            df_react.style.map(color_diff, subset=pct_cols_r).format(fmt_pct_cap, subset=pct_cols_r, na_rep="--"),
            width='stretch'
        )
    else:
        st.info("No reaction data found (check supports).")