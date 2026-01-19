import streamlit as st
import pandas as pd
import numpy as np
import json
import copy
import os
import sys
import time

# ==========================================
# GLOBAL CONFIGURATION
# ==========================================

APP_VERSION = "0.31"
AUTOSAVE_FILE = "latest_session.csv"

# ==========================================
# PATH & RESOURCE HELPERS
# ==========================================

def resource_path(relative_path):
    """ 
    Get absolute path to bundled resources (images, static csv).
    In EXE: Points to sys._MEIPASS (Temp folder).
    In DEV: Points to the directory containing this script.
    """
    if getattr(sys, 'frozen', False):
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    else:
        # Robustly get the folder where bricos_data.py is located
        base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_path, relative_path)

def get_writable_path(filename):
    """
    Get path for writing persistent user data.
    """
    if getattr(sys, 'frozen', False):
        base_path = os.path.dirname(sys.executable)
    else:
        # Use script directory to ensure autosaves go to the project folder
        # regardless of where the terminal command was run.
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    return os.path.join(base_path, filename)

# ==========================================
# DATA HELPERS & CALCULATIONS
# ==========================================

def calc_I(h_mm):
    return (1.0 * (h_mm/1000.0)**3) / 12.0

@st.cache_data
def get_vehicle_library():
    """
    Reads the vehicles.csv file and returns:
    1. options: List of vehicle names for the dropdown.
    2. data: Dictionary mapping names to raw load/spacing strings.
    """
    options = ["Custom"]
    data = {}
    try:
        csv_path = resource_path("vehicles.csv")
        if os.path.exists(csv_path):
            df_v = pd.read_csv(csv_path)
            # Sanitization: Strip whitespace from headers
            df_v.columns = [c.strip() for c in df_v.columns]
            
            if 'Name' in df_v.columns and 'Loads' in df_v.columns and 'Spacing' in df_v.columns:
                for _, row in df_v.iterrows():
                    v_name = str(row['Name']).strip()
                    options.append(v_name)
                    data[v_name] = {
                        'loads': str(row['Loads']), 
                        'spacing': str(row['Spacing'])
                    }
    except Exception as e:
        # Fail silently in UI but allow debugging if needed
        print(f"Vehicle Load Error: {e}")
        pass
        
    return options, data

def load_vehicle_from_csv(target_name):
    """Attempts to read a specific vehicle from the cached library. Returns parsed dict or None."""
    _, lib_data = get_vehicle_library()
    
    if target_name in lib_data:
        raw = lib_data[target_name]
        try:
            l_str = raw['loads']
            s_str = raw['spacing']
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

def sanitize_input_data(data):
    """
    Ensures input dictionaries are clean and devoid of placeholder zeros.
    """
    # 1. Clean Span Lists
    nsp = data.get('num_spans', 1)
    
    # 2. Geometry Defaults
    for i in range(10):
        k = f"span_geom_{i}"
        if k in data:
            g = data[k]
            if not g.get('vals'): g['vals'] = [0.0, 0.0, 0.0]
            # If main list has value but geom is 0, sync them
            if i < len(data['Is_list']) and g['vals'] == [0.0, 0.0, 0.0]:
                val = data['Is_list'][i]
                g['vals'] = [val, val, val]
                
    for i in range(11):
        k = f"wall_geom_{i}"
        if k in data:
            g = data[k]
            if not g.get('vals'): g['vals'] = [0.0, 0.0, 0.0]
            if i < len(data['Iw_list']) and g['vals'] == [0.0, 0.0, 0.0]:
                val = data['Iw_list'][i]
                g['vals'] = [val, val, val]

    # 3. Ensure Material Lists are populated
    while len(data['fck_span_list']) < 10: data['fck_span_list'].append(30.0)
    while len(data['E_span_list']) < 10: data['E_span_list'].append(30e6)
    while len(data['fck_wall_list']) < 11: data['fck_wall_list'].append(30.0)
    while len(data['E_wall_list']) < 11: data['E_wall_list'].append(30e6)
    
    return data

# ==========================================
# DEFAULT STATES & TEMPLATES
# ==========================================

def get_def():
    # Updated Defaults based on User Request
    I_def = calc_I(500)
    
    # Attempt to load Class 100 using the new centralized logic
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

    base_def = {
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
        'soil': [], 
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
        
        'mesh_size': 2.0,
        'step_size': 0.2,
        'vehicle_direction': 'Both',
        'use_shear_def': True,
        'b_eff': 3.0,
        
        'name': 'System',
        'last_mode': 'Frame',
        'combine_surcharge_vehicle': False,
        'nu': 0.2
    }
    return sanitize_input_data(base_def)

def get_clear(name_suffix, current_mode):
    # Strictly cleared state: 0 Geometry, 1.0 Factors, Manual Phi=1.0
    return {
        'mode': current_mode, 
        'E': 30e6, 'num_spans': 1,
        'L_list': [0.0]*10, 'Is_list': [0.0]*10, 'sw_list': [0.0]*10,
        'h_list': [0.0]*11, 'Iw_list': [0.0]*11,
        'e_mode': 'Manual', 
        'fck_span_list': [30.0]*10, 'fck_wall_list': [30.0]*11,
        'E_custom_span': [30.0]*10, 'E_custom_wall': [30.0]*11,
        'E_span_list': [30e6]*10, 'E_wall_list': [30e6]*11,
        
        'supports': [],
        'soil': [],
        'surcharge': [], 
        
        'vehicle': {'loads': [], 'spacing': []},
        'vehicle_loads': "", 'vehicle_space': "",
        'vehicleB': {'loads': [], 'spacing': []},
        'vehicleB_loads': "", 'vehicleB_space': "",
        
        'KFI': 1.0, 
        'gamma_g': 1.0, 'gamma_j': 1.0, 
        'gamma_veh': 1.0, 'gamma_vehB': 1.0, 
        
        'phi': 1.0, 
        'phi_mode': 'Manual',
        
        'scale_manual': 2.0, 
        'mesh_size': 0.5, 'step_size': 0.2,
        'name': f"System {name_suffix}",
        'last_mode': current_mode,
        'combine_surcharge_vehicle': False,
        'vehicle_direction': 'Forward',
        
        'use_shear_def': False,
        'b_eff': 1.0,
        'nu': 0.2
    }

# ==========================================
# STATE MANAGEMENT
# ==========================================

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
    
    # 2. Shear Deformation Keys & Analysis Settings
    if sys_key == "sysA":
        st.session_state["shear_toggle_sidebar"] = data.get('use_shear_def', False)
        st.session_state["beff_input_sidebar"] = data.get('b_eff', 1.0)
        st.session_state["nu_input_sidebar"] = data.get('nu', 0.2)
        st.session_state["common_mesh_slider"] = data.get('mesh_size', 0.5)
        st.session_state["common_step_slider"] = data.get('step_size', 0.2)
    
    # 3. Factors
    st.session_state[f"{sys_key}_gg_cust"] = data.get('gamma_g', 1.0)
    st.session_state[f"{sys_key}_gj_cust"] = data.get('gamma_j', 1.0)
    st.session_state[f"{sys_key}_gamA_cust"] = data.get('gamma_veh', 1.0)
    st.session_state[f"{sys_key}_gamB_cust"] = data.get('gamma_vehB', 1.0)

    # 4. Spans & Profiler Keys
    shape_map_rev = {0: "Constant", 1: "Linear (Taper)", 2: "3-Point (Start/Mid/End)"}
    type_map_rev = {0: "Inertia (I)", 1: "Height (H)"}
    align_map_rev = {0: "Straight (Horizontal)", 1: "Inclined"}
    inc_mode_rev = {0: "Slope (%)", 1: "Delta Height (End - Start) [m]"}
    
    for i in range(10): 
        if i < len(data['L_list']): st.session_state[f"{sys_key}_l{i}"] = data['L_list'][i]
        if i < len(data['Is_list']): st.session_state[f"{sys_key}_i{i}"] = data['Is_list'][i]
        if i < len(data['sw_list']): st.session_state[f"{sys_key}_s{i}"] = data['sw_list'][i]
        if i < len(data['fck_span_list']): st.session_state[f"{sys_key}_fck_s{i}"] = data['fck_span_list'][i]
        if i < len(data['E_custom_span']): st.session_state[f"{sys_key}_Eman_s{i}"] = data['E_custom_span'][i]
        st.session_state[f"{sys_key}_i{i}_dis"] = "See Profiler"

        geom_key = f"span_geom_{i}"
        if geom_key in data:
            g = data[geom_key]
            el_name = f"Span {i+1}"
            st.session_state[f"{sys_key}_prof_type_{el_name}"] = type_map_rev.get(g.get('type', 0), "Inertia (I)")
            st.session_state[f"{sys_key}_prof_shape_{el_name}"] = shape_map_rev.get(g.get('shape', 0), "Constant")
            
            vals = g.get('vals', [0.0, 0.0, 0.0])
            st.session_state[f"{sys_key}_prof_v1_{el_name}"] = vals[0]
            st.session_state[f"{sys_key}_prof_v2_{el_name}"] = vals[1]
            st.session_state[f"{sys_key}_prof_v3_{el_name}"] = vals[2]
            
            st.session_state[f"{sys_key}_align_t_{el_name}"] = align_map_rev.get(g.get('align_type', 0), "Straight (Horizontal)")
            st.session_state[f"{sys_key}_inc_m_{el_name}"] = inc_mode_rev.get(g.get('incline_mode', 0), "Slope (%)")
            st.session_state[f"{sys_key}_inc_v_{el_name}"] = g.get('incline_val', 0.0)

    # 5. Walls & Profiler Keys
    for i in range(11):
        if i < len(data['h_list']): st.session_state[f"{sys_key}_h{i}"] = data['h_list'][i]
        if i < len(data['Iw_list']): st.session_state[f"{sys_key}_iw{i}"] = data['Iw_list'][i]
        if i < len(data['fck_wall_list']): st.session_state[f"{sys_key}_fck_w{i}"] = data['fck_wall_list'][i]
        if i < len(data['E_custom_wall']): st.session_state[f"{sys_key}_Eman_w{i}"] = data['E_custom_wall'][i]
        
        geom_key = f"wall_geom_{i}"
        if geom_key in data:
            g = data[geom_key]
            el_name = f"Wall {i+1}"
            st.session_state[f"{sys_key}_prof_type_{el_name}"] = type_map_rev.get(g.get('type', 0), "Inertia (I)")
            st.session_state[f"{sys_key}_prof_shape_{el_name}"] = shape_map_rev.get(g.get('shape', 0), "Constant")
            
            vals = g.get('vals', [0.0, 0.0, 0.0])
            st.session_state[f"{sys_key}_prof_v1_{el_name}"] = vals[0]
            st.session_state[f"{sys_key}_prof_v2_{el_name}"] = vals[1]
            st.session_state[f"{sys_key}_prof_v3_{el_name}"] = vals[2]

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

    # 6. Vehicles
    st.session_state[f"{sys_key}_A_loads_input"] = data.get('vehicle_loads', "")
    st.session_state[f"{sys_key}_A_space_input"] = data.get('vehicle_space', "")
    st.session_state[f"{sys_key}_B_loads_input"] = data.get('vehicleB_loads', "")
    st.session_state[f"{sys_key}_B_space_input"] = data.get('vehicleB_space', "")
    
    if f"{sys_key}_vehA_class" in st.session_state: del st.session_state[f"{sys_key}_vehA_class"]
    if f"{sys_key}_vehB_class" in st.session_state: del st.session_state[f"{sys_key}_vehB_class"]
    
    # 7. Supports
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
    Removes temporary UI keys.
    """
    keys_to_clear = []
    for k in st.session_state.keys():
        if "_prof_" in k or "_align_t" in k or "_inc_" in k:
            keys_to_clear.append(k)
    
    for k in keys_to_clear:
        del st.session_state[k]

def generate_csv_data():
    """Generates the CSV string for saving the current session state."""
    rep_keys = ['rep_pno', 'rep_pname', 'rep_rev', 'rep_author', 'rep_check', 'rep_appr', 'rep_comm']
    
    rows = []
    for sys_name in ['sysA', 'sysB']:
        if sys_name in st.session_state:
            data = st.session_state[sys_name]
            for k, v in data.items():
                if k == 'backup': continue 
                val_str = json.dumps(v, default=str)
                rows.append({'System': sys_name, 'Parameter': k, 'Value': val_str})
    
    global_keys = rep_keys + ['result_mode']
    for gk in global_keys:
        if gk in st.session_state:
            val_str = json.dumps(st.session_state[gk], default=str)
            rows.append({'System': 'Global', 'Parameter': gk, 'Value': val_str})

    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode('utf-8')

def load_data_from_df(df_load):
    """Loads session state from a DataFrame."""
    if 'System' in df_load.columns and 'Parameter' in df_load.columns:
        clean_transient_keys()
        
        for _, row in df_load.iterrows():
            sys_n = row['System']
            try:
                val = json.loads(row['Value'])
                if sys_n in ['sysA', 'sysB']:
                    if sys_n not in st.session_state:
                         st.session_state[sys_n] = get_def() 
                    st.session_state[sys_n][row['Parameter']] = val
                elif sys_n == 'Global':
                    st.session_state[row['Parameter']] = val
            except: pass
        
        if 'sysA' in st.session_state:
            st.session_state['sysA'] = sanitize_input_data(st.session_state['sysA'])
            force_ui_update('sysA', st.session_state['sysA'])
        if 'sysB' in st.session_state:
            st.session_state['sysB'] = sanitize_input_data(st.session_state['sysB'])
            force_ui_update('sysB', st.session_state['sysB'])
        return True
    return False

def initialize_session_state():
    """
    Main initialization logic.
    """
    if 'keep_view_case' not in st.session_state: st.session_state.keep_view_case = "Total Envelope"
    if 'keep_active_veh_step' not in st.session_state: st.session_state.keep_active_veh_step = "Vehicle A"
    if 'keep_step_view_sys' not in st.session_state: st.session_state.keep_step_view_sys = "System A"
    if 'is_generating_report' not in st.session_state: st.session_state.is_generating_report = False
    if 'last_autosave_time' not in st.session_state: st.session_state.last_autosave_time = time.time()
    if 'autosave_interval' not in st.session_state: st.session_state.autosave_interval = 5

    if 'sysA' not in st.session_state:
        autosave_path = get_writable_path(AUTOSAVE_FILE)
        loaded_from_autosave = False
        
        if os.path.exists(autosave_path):
            try:
                df_auto = pd.read_csv(autosave_path)
                st.session_state['sysA'] = get_def()
                st.session_state['sysB'] = get_def()
                if load_data_from_df(df_auto):
                    loaded_from_autosave = True
            except Exception:
                pass 

        if not loaded_from_autosave:
            d = get_def()
            d['num_spans'] = 1
            d['name'] = "System A"
            d['soil'] = [
                {'wall_idx': 0, 'face': 'L', 'h': 8.0, 'q_top': 0.0, 'q_bot': 20.0}, 
                {'wall_idx': 0, 'face': 'R', 'h': 4.0, 'q_top': 0.0, 'q_bot': 10.0},
                {'wall_idx': 1, 'face': 'L', 'h': 4.0, 'q_top': 0.0, 'q_bot': 10.0},
                {'wall_idx': 1, 'face': 'R', 'h': 8.0, 'q_top': 0.0, 'q_bot': 20.0}
            ]
            st.session_state['sysA'] = sanitize_input_data(d)

    if 'sysB' not in st.session_state: 
        d = get_def()
        d['num_spans'] = 2
        d['name'] = "System B"
        d['soil'] = [
            {'wall_idx': 0, 'face': 'L', 'h': 8.0, 'q_top': 0.0, 'q_bot': 20.0},
            {'wall_idx': 0, 'face': 'R', 'h': 4.0, 'q_top': 0.0, 'q_bot': 10.0},
            {'wall_idx': 2, 'face': 'L', 'h': 4.0, 'q_top': 0.0, 'q_bot': 10.0},
            {'wall_idx': 2, 'face': 'R', 'h': 8.0, 'q_top': 0.0, 'q_bot': 20.0}
        ]
        st.session_state['sysB'] = sanitize_input_data(d)

    if 'name' not in st.session_state['sysA']: st.session_state['sysA']['name'] = "System A"
    if 'name' not in st.session_state['sysB']: st.session_state['sysB']['name'] = "System B"

    # Migration
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
        
        if 'e_mode' not in st.session_state[sys_k]: st.session_state[sys_k]['e_mode'] = 'Eurocode'
        if 'fck_span_list' not in st.session_state[sys_k]: st.session_state[sys_k]['fck_span_list'] = [30.0]*10
        if 'fck_wall_list' not in st.session_state[sys_k]: st.session_state[sys_k]['fck_wall_list'] = [30.0]*11
        if 'E_custom_span' not in st.session_state[sys_k]: st.session_state[sys_k]['E_custom_span'] = [33.0]*10
        if 'E_custom_wall' not in st.session_state[sys_k]: st.session_state[sys_k]['E_custom_wall'] = [33.0]*11
        if 'E_span_list' not in st.session_state[sys_k]: st.session_state[sys_k]['E_span_list'] = [33e6]*10
        if 'E_wall_list' not in st.session_state[sys_k]: st.session_state[sys_k]['E_wall_list'] = [33e6]*11
        
        if 'use_shear_def' not in st.session_state[sys_k]: st.session_state[sys_k]['use_shear_def'] = False
        if 'b_eff' not in st.session_state[sys_k]: st.session_state[sys_k]['b_eff'] = 1.0
        if 'nu' not in st.session_state[sys_k]: st.session_state[sys_k]['nu'] = 0.2
        
        st.session_state[sys_k].pop('k_rot', None)
        st.session_state[sys_k].pop('k_rot_super', None)

    if 'uploader_key' not in st.session_state: st.session_state.uploader_key = 0

    if st.session_state['sysA'].get('scale_manual', 0) < 0.1: st.session_state['sysA']['scale_manual'] = 2.0
    if st.session_state['sysB'].get('scale_manual', 0) < 0.1: st.session_state['sysB']['scale_manual'] = 2.0