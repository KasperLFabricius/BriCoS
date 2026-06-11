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

APP_VERSION = "0.55"
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
        # The executable may live in a read-only location (e.g. Program
        # Files), so persist user data under %APPDATA%\BriCoS instead.
        appdata = os.environ.get('APPDATA')
        base_path = os.path.dirname(sys.executable)
        if appdata:
            candidate = os.path.join(appdata, 'BriCoS')
            try:
                os.makedirs(candidate, exist_ok=True)
                base_path = candidate
            except OSError:
                pass
    else:
        # Use script directory to ensure autosaves go to the project folder
        # regardless of where the terminal command was run.
        base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_path, filename)

def get_legacy_writable_path(filename):
    """Pre-v0.45 location of persistent user data (next to the executable).

    Read-only fallback so frozen builds keep loading autosaves written
    before user data moved to %APPDATA%\\BriCoS. Returns None when not
    running as a frozen executable.
    """
    if getattr(sys, 'frozen', False):
        return os.path.join(os.path.dirname(sys.executable), filename)
    return None

# ==========================================
# DATA HELPERS & CALCULATIONS
# ==========================================

def calc_I(h_mm):
    # Legacy helper, kept for reference or potential future use.
    # Note: Solver now handles I calculation using b_eff.
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
        except (ValueError, TypeError, KeyError):
            pass
    return None

def identify_vehicle_class(loads, spacing):
    """
    Matches current load/spacing arrays against the library to find the class name.
    Returns 'Custom' if no match found.
    """
    if not loads: return "Custom"
    
    _, lib_data = get_vehicle_library()
    
    # Convert inputs to numpy arrays for comparison
    try:
        curr_l = np.array(loads, dtype=float)
        curr_s = np.array(spacing, dtype=float)
    except (ValueError, TypeError):
        return "Custom"

    for name, data in lib_data.items():
        try:
            # Parse lib strings to arrays
            lib_l = np.fromstring(data['loads'], sep=',')
            lib_s = np.fromstring(data['spacing'], sep=',')
            
            # 1. Check lengths
            if len(lib_l) != len(curr_l) or len(lib_s) != len(curr_s):
                continue
            
            # 2. Check values with tolerance
            if np.allclose(lib_l, curr_l, atol=1e-3) and np.allclose(lib_s, curr_s, atol=1e-3):
                return name
        except (ValueError, TypeError, KeyError):
            continue
            
    return "Custom"

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

    # 4. Migration: v0.49 stored the SLS stoedfaktor reduction as a boolean
    # (phi_sls_reduction). Loaded sessions are seeded with current defaults
    # before file values are applied, so phi_sls_mode always exists here;
    # migrate on the legacy value instead of key absence. v0.50+ sessions
    # never carry the legacy key, so an explicit 'Same' cannot be overridden.
    if data.get('phi_sls_reduction') is True and data.get('phi_sls_mode', 'Same') == 'Same':
        data['phi_sls_mode'] = 'Reduced'
    data.pop('phi_sls_reduction', None)

    # Migration: v0.51 stored the SLS factor for the traffic UDL as
    # udl_sls_factor; it is now part of the SLS factor set (sls_udl).
    if 'udl_sls_factor' in data:
        data['sls_udl'] = data.pop('udl_sls_factor')

    return data

# ==========================================
# VALIDATION HELPERS
# ==========================================

MIN_ENGINEERING_HEIGHT_M = 0.05
MIN_ENGINEERING_INERTIA_M4 = 1e-8
MIN_EFFECTIVE_WIDTH_M = 0.01
ACTIVE_WALL_TOLERANCE_M = 1e-4


def _is_finite_number(value):
    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _as_float(value, default=None):
    try:
        val = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(val):
        return default
    return val


def _list_value(values, index, default=None):
    if not isinstance(values, (list, tuple)) or index >= len(values):
        return default
    return values[index]


def _active_geom_indices(shape):
    try:
        shape = int(shape)
    except (TypeError, ValueError):
        return [0]
    if shape == 1:
        return [0, 2]
    if shape == 2:
        return [0, 1, 2]
    return [0]


def _validate_section_geometry(params, geom_key, fallback_key, index, label, system_label):
    errors = []
    fallback_val = _list_value(params.get(fallback_key, []), index, 0.0)
    geom = params.get(geom_key) or {'type': 1, 'shape': 0, 'vals': [fallback_val, fallback_val, fallback_val]}
    vals = geom.get('vals', [])
    if not isinstance(vals, (list, tuple)):
        vals = []

    try:
        val_mode = int(geom.get('type', 1))
    except (TypeError, ValueError):
        val_mode = 1

    if val_mode not in (0, 1):
        val_mode = 1
    shape = geom.get('shape', 0)
    active_indices = _active_geom_indices(shape)
    quantity = "inertia" if val_mode == 0 else "height"
    symbol = "I" if val_mode == 0 else "h"
    minimum = MIN_ENGINEERING_INERTIA_M4 if val_mode == 0 else MIN_ENGINEERING_HEIGHT_M
    units = "m^4" if val_mode == 0 else "m"

    for val_idx in active_indices:
        value = _as_float(_list_value(vals, val_idx, None), None)
        if value is None or value < minimum:
            if val_mode == 0:
                value_text = "non-numeric" if value is None else f"{value:.3e}"
                min_text = f"{minimum:.1e}"
            else:
                value_text = "non-numeric" if value is None else f"{value:.3f}"
                min_text = f"{minimum:.2f}"
            errors.append(
                f"{system_label}: {label} has invalid section {quantity} {symbol} = {value_text} {units}. "
                f"Minimum allowed {quantity} is {min_text} {units}."
            )
    return errors


def udl_line_load(params) -> float:
    """Traffic UDL line load [kN/m] on the analysis strip.

    Defined directly as a line load, like selfweight; the user accounts for
    the effective width in the input. Returns 0.0 when the UDL is inactive
    or invalidly defined - an invalid definition must never enter the
    analysis.
    """
    q = _as_float(params.get('udl_q', 0.0), 0.0)
    if q is None or q <= 0.0:
        return 0.0
    return q


# Selectbox presets for the vehicle-to-UDL clear distance. Single source of
# truth for the UI selectbox and for force_ui_update, which must restore the
# selector to the label matching the stored value (a stale selector silently
# overwrites a loaded custom distance with its preset on the next rerun).
UDL_GAP_PRESETS = {
    "10.0 m (DK NA Fig. A.2.2-2)": 10.0,
    "0.0 m (adjacent to vehicle)": 0.0,
}


def udl_gap_preset_label(gap) -> str:
    """The gap selectbox label matching a stored clear distance."""
    g = _as_float(gap, 10.0)
    for label, val in UDL_GAP_PRESETS.items():
        if abs(val - g) < 1e-9:
            return label
    return "Custom"


def parse_vehicle_text(load_text: str, spacing_text: str):
    """Parse vehicle text inputs without mutating UI state."""
    load_text = "" if load_text is None else str(load_text)
    spacing_text = "" if spacing_text is None else str(spacing_text)

    if not load_text.strip() and not spacing_text.strip():
        return {'loads': [], 'spacing': []}, []

    errors = []
    try:
        loads = [float(x) for x in load_text.split(',') if x.strip()]
    except ValueError:
        loads = []
        errors.append("Load input contains non-numeric values.")

    try:
        spacing = [float(x) for x in spacing_text.split(',') if x.strip()]
    except ValueError:
        spacing = []
        errors.append("Spacing input contains non-numeric values.")

    if errors:
        return {'loads': [], 'spacing': []}, errors

    if len(loads) != len(spacing):
        errors.append(f"Vehicle has {len(loads)} axle loads but {len(spacing)} spacing values.")
    if loads and spacing and spacing[0] != 0.0:
        errors.append("Vehicle spacing must start with 0.0 m.")
    if any(not np.isfinite(v) or v < 0.0 for v in loads):
        errors.append("Vehicle axle loads must be finite and non-negative.")
    if any(not np.isfinite(v) or v < 0.0 for v in spacing):
        errors.append("Vehicle spacing values must be finite and non-negative.")
    if not loads and load_text.strip():
        errors.append("Vehicle definition is empty.")

    if errors:
        return {'loads': [], 'spacing': []}, errors
    return {'loads': loads, 'spacing': spacing}, []



def format_vehicle_text(vehicle: dict) -> tuple[str, str]:
    """Return comma-separated load and incremental-spacing text for a vehicle object."""
    if not isinstance(vehicle, dict):
        return "", ""

    def _fmt(values):
        out = []
        for value in values or []:
            try:
                out.append(f"{float(value):g}")
            except (TypeError, ValueError):
                out.append(str(value))
        return ", ".join(out)

    return _fmt(vehicle.get('loads', [])), _fmt(vehicle.get('spacing', []))


def _set_vehicle_text_error(params: dict, vehicle_key: str, errors: list[str]) -> None:
    error_map = params.setdefault('_vehicle_text_errors', {})
    if errors:
        error_map[vehicle_key] = list(errors)
    else:
        error_map.pop(vehicle_key, None)
    if not error_map:
        params.pop('_vehicle_text_errors', None)


def normalize_vehicle_fields(params: dict, vehicle_key: str, loads_key: str, spacing_key: str) -> list[str]:
    """Normalize vehicle object/text fields with the calculation dict as source of truth.

    A valid existing vehicle object is never discarded just because text fields are
    missing or blank; blank text only means inactive when the vehicle object is
    already empty. Invalid text is preserved and recorded for validation.
    """
    params = params if params is not None else {}
    vehicle = params.get(vehicle_key) if isinstance(params.get(vehicle_key), dict) else {'loads': [], 'spacing': []}
    loads_text = "" if params.get(loads_key) is None else str(params.get(loads_key, ""))
    spacing_text = "" if params.get(spacing_key) is None else str(params.get(spacing_key, ""))
    has_text = bool(loads_text.strip()) or bool(spacing_text.strip())
    has_vehicle = bool(vehicle.get('loads')) or bool(vehicle.get('spacing'))

    if has_text:
        parsed_vehicle, errors = parse_vehicle_text(loads_text, spacing_text)
        params[loads_key] = loads_text
        params[spacing_key] = spacing_text
        if errors:
            # Preserve both the visible invalid text and the last valid vehicle object.
            params[vehicle_key] = vehicle
            _set_vehicle_text_error(params, vehicle_key, errors)
            return errors
        params[vehicle_key] = parsed_vehicle
        _set_vehicle_text_error(params, vehicle_key, [])
        return []

    if has_vehicle:
        # Treat missing/blank text as stale projection and regenerate from the object.
        params[loads_key], params[spacing_key] = format_vehicle_text(vehicle)
        params[vehicle_key] = vehicle
        _set_vehicle_text_error(params, vehicle_key, [])
        return []

    params[loads_key] = ""
    params[spacing_key] = ""
    params[vehicle_key] = {'loads': [], 'spacing': []}
    _set_vehicle_text_error(params, vehicle_key, [])
    return []


def vehicle_state_signature(params: dict, vehicle_key: str, loads_key: str, spacing_key: str) -> str:
    """Return a stable signature for detecting calculation-state vehicle changes."""
    payload = {
        'vehicle': params.get(vehicle_key, {}) if params else {},
        'loads_text': params.get(loads_key, "") if params else "",
        'spacing_text': params.get(spacing_key, "") if params else "",
        'errors': (params.get('_vehicle_text_errors', {}) if params else {}).get(vehicle_key, []),
    }
    return json.dumps(payload, sort_keys=True, default=str)


def sync_vehicle_widgets_from_params(sys_key: str, params: dict) -> None:
    """Project normalized vehicle state into Streamlit widget keys for one system."""
    normalize_vehicle_fields(params, 'vehicle', 'vehicle_loads', 'vehicle_space')
    normalize_vehicle_fields(params, 'vehicleB', 'vehicleB_loads', 'vehicleB_space')

    st.session_state[f"{sys_key}_A_loads_input"] = params.get('vehicle_loads', "")
    st.session_state[f"{sys_key}_A_space_input"] = params.get('vehicle_space', "")
    st.session_state[f"{sys_key}_B_loads_input"] = params.get('vehicleB_loads', "")
    st.session_state[f"{sys_key}_B_space_input"] = params.get('vehicleB_space', "")

    veh_a = params.get('vehicle', {}) if isinstance(params.get('vehicle'), dict) else {}
    veh_b = params.get('vehicleB', {}) if isinstance(params.get('vehicleB'), dict) else {}
    cls_a = identify_vehicle_class(veh_a.get('loads', []), veh_a.get('spacing', []))
    cls_b = identify_vehicle_class(veh_b.get('loads', []), veh_b.get('spacing', []))
    st.session_state[f"{sys_key}_vehA_class"] = cls_a
    st.session_state[f"{sys_key}_vehA_class_last"] = cls_a
    st.session_state[f"{sys_key}_vehB_class"] = cls_b
    st.session_state[f"{sys_key}_vehB_class_last"] = cls_b
    st.session_state[f"{sys_key}_A_vehicle_sig"] = vehicle_state_signature(params, 'vehicle', 'vehicle_loads', 'vehicle_space')
    st.session_state[f"{sys_key}_B_vehicle_sig"] = vehicle_state_signature(params, 'vehicleB', 'vehicleB_loads', 'vehicleB_space')


def apply_standard_vehicle_definition(params: dict, vehicle_name: str, vehicle_key: str, loads_key: str, spacing_key: str) -> list[str]:
    """Replace one vehicle with a named library definition."""
    _, lib_data = get_vehicle_library()
    if vehicle_name not in lib_data:
        return [f"Vehicle class '{vehicle_name}' was not found in the library."]
    params[loads_key] = lib_data[vehicle_name]['loads']
    params[spacing_key] = lib_data[vehicle_name]['spacing']
    return normalize_vehicle_fields(params, vehicle_key, loads_key, spacing_key)


def copy_system_data(source_data: dict, target_name=None) -> dict:
    """Deep-copy a system dict while preserving normalized vehicle state."""
    copied = copy.deepcopy(source_data)
    if target_name is not None:
        copied['name'] = target_name
    normalize_vehicle_fields(copied, 'vehicle', 'vehicle_loads', 'vehicle_space')
    normalize_vehicle_fields(copied, 'vehicleB', 'vehicleB_loads', 'vehicleB_space')
    return copied

def clear_vehicle_definition(params: dict, vehicle_key: str, load_text_key: str, spacing_text_key: str) -> None:
    """Clear one vehicle definition explicitly while preserving other state."""
    params[load_text_key] = ""
    params[spacing_text_key] = ""
    params[vehicle_key] = {'loads': [], 'spacing': []}
    _set_vehicle_text_error(params, vehicle_key, [])


def _validate_vehicle(vehicle, label, system_label):
    loads = vehicle.get('loads', []) if isinstance(vehicle, dict) else []
    spacing = vehicle.get('spacing', []) if isinstance(vehicle, dict) else []
    if not loads:
        return []

    errors = []
    if len(loads) != len(spacing):
        errors.append(f"{system_label}: {label} has {len(loads)} axle loads but {len(spacing)} spacing values.")
        return errors
    if spacing[0] != 0.0:
        errors.append(f"{system_label}: {label} spacing must start with 0.0 m.")
    if any(not _is_finite_number(v) or float(v) < 0.0 for v in loads):
        errors.append(f"{system_label}: {label} axle loads must be finite and non-negative.")
    if any(not _is_finite_number(v) or float(v) < 0.0 for v in spacing):
        errors.append(f"{system_label}: {label} spacing values must be finite and non-negative.")
    return errors


def validate_analysis_inputs(params: dict, system_label: str = "System") -> list[str]:
    """Return actionable engineering validation errors before analysis/reporting."""
    errors = []
    params = params or {}

    mode = params.get('mode', 'Frame')
    if mode not in ("Frame", "Beam", "Superstructure"):
        errors.append(f"{system_label}: Mode must be Frame, Beam, or Superstructure.")

    num_spans = params.get('num_spans', 1)
    if not isinstance(num_spans, int) or isinstance(num_spans, bool):
        errors.append(f"{system_label}: Number of spans must be an integer between 1 and 10.")
        num_spans_int = 0
    else:
        num_spans_int = num_spans
        if num_spans_int < 1 or num_spans_int > 10:
            errors.append(f"{system_label}: Number of spans must be between 1 and 10.")

    b_eff = _as_float(params.get('b_eff', None), None)
    if b_eff is None or b_eff < MIN_EFFECTIVE_WIDTH_M:
        errors.append(f"{system_label}: Effective width b_eff must be positive and at least {MIN_EFFECTIVE_WIDTH_M:.2f} m.")

    mesh_size = _as_float(params.get('mesh_size', None), None)
    if mesh_size is None or mesh_size <= 0.0:
        errors.append(f"{system_label}: Mesh size must be positive.")

    has_vehicle = bool(params.get('vehicle', {}).get('loads')) or bool(params.get('vehicleB', {}).get('loads'))
    step_size = _as_float(params.get('step_size', None), None)
    if has_vehicle and (step_size is None or step_size <= 0.0):
        errors.append(f"{system_label}: Vehicle step size must be positive when vehicle loads exist.")

    # Traffic UDL (fladelast). q = 0 (or blank) simply deactivates the load;
    # negative or non-finite definitions are rejected.
    udl_q = _as_float(params.get('udl_q', 0.0), None)
    if udl_q is None or udl_q < 0.0:
        errors.append(f"{system_label}: Traffic UDL line load must be a non-negative number (0 deactivates it).")
    udl_gap = _as_float(params.get('udl_gap', 10.0), None)
    if udl_gap is None or udl_gap < 0.0:
        errors.append(f"{system_label}: Traffic UDL distance to vehicle must be non-negative.")

    e_default = _as_float(params.get('E', None), None)
    active_spans = max(0, min(num_spans_int, 10))
    for i in range(active_spans):
        L = _as_float(_list_value(params.get('L_list', []), i, None), None)
        if L is None or L <= 0.0:
            errors.append(f"{system_label}: Span S{i+1} length must be positive.")

        e_span = _as_float(_list_value(params.get('E_span_list', []), i, e_default), e_default)
        if e_span is None or e_span <= 0.0:
            errors.append(f"{system_label}: Span S{i+1} E modulus must be positive.")

        errors.extend(_validate_section_geometry(params, f"span_geom_{i}", 'Is_list', i, f"Span S{i+1}", system_label))

    supports = params.get('supports', [])
    if supports and len(supports) != num_spans_int + 1:
        errors.append(f"{system_label}: Supports list must contain {num_spans_int + 1} support definitions.")
    for i, support in enumerate(supports[:num_spans_int + 1]):
        k_vec = support.get('k') if isinstance(support, dict) else support
        if not isinstance(k_vec, (list, tuple)) or len(k_vec) != 3:
            errors.append(f"{system_label}: Support {i+1} stiffness vector must contain exactly 3 numeric values.")
            continue
        if any(not _is_finite_number(v) for v in k_vec):
            errors.append(f"{system_label}: Support {i+1} stiffness values must be finite.")
        elif any(float(v) < 0.0 for v in k_vec):
            errors.append(f"{system_label}: Support {i+1} stiffness values must not be negative.")

    if mode == "Frame":
        for i in range(active_spans + 1):
            h = _as_float(_list_value(params.get('h_list', []), i, 0.0), 0.0)
            if h > ACTIVE_WALL_TOLERANCE_M:
                if h < MIN_ENGINEERING_HEIGHT_M:
                    errors.append(
                        f"{system_label}: Wall W{i+1} member height h = {h:.3f} m is invalid. "
                        f"Minimum wall member height is {MIN_ENGINEERING_HEIGHT_M:.2f} m."
                    )
                e_wall = _as_float(_list_value(params.get('E_wall_list', []), i, e_default), e_default)
                if e_wall is None or e_wall <= 0.0:
                    errors.append(f"{system_label}: Wall W{i+1} E modulus must be positive.")
                errors.extend(_validate_section_geometry(params, f"wall_geom_{i}", 'Iw_list', i, f"Wall W{i+1}", system_label))

    vehicle_error_map = params.get('_vehicle_text_errors', {})
    if isinstance(vehicle_error_map, dict):
        label_map = {'vehicle': 'Vehicle A', 'vehicleB': 'Vehicle B'}
        for vehicle_key, text_errors in vehicle_error_map.items():
            label = label_map.get(vehicle_key, vehicle_key)
            for text_error in text_errors or []:
                errors.append(f"{system_label}: {label} text input is invalid: {text_error}")

    errors.extend(_validate_vehicle(params.get('vehicle', {}), "Vehicle A", system_label))
    errors.extend(_validate_vehicle(params.get('vehicleB', {}), "Vehicle B", system_label))
    return errors

# ==========================================
# DEFAULT STATES & TEMPLATES
# ==========================================

def get_def():
    # Updated Defaults based on User Request
    # Default is now Height = 0.5m instead of calculated Inertia
    H_def = 0.5
    
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
        # Now stores Height [m] by default (was Inertia)
        'Is_list': [H_def]*10,
        'sw_list': [20.0]*10,
        'h_list': [8.0]*11,
        # Now stores Height [m] by default (was Inertia)
        'Iw_list': [H_def]*11,
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
        # L_inf methodology: 'Determinant' (EN 1991-2:2003 Tab. 6.2, combined
        # system) or 'Span' (DK NA A.2.3.5(2), L_inf = actual span).
        'phi_linf_mode': 'Determinant',
        # In 'Span' mode: 'Per member' or 'Governing' (max applied to all).
        'phi_application': 'Per member',
        # Manual phi scope: 'Global' (single value) or 'Per span'
        # (phi_span_list; walls take the max of adjacent spans).
        'phi_manual_scope': 'Global',
        'phi_span_list': [1.0]*10,
        # Phi in the Characteristic (SLS) result mode:
        # 'Same' (= ULS), 'Reduced' (1+(phi-1)/2, Vejledning 5.4.2) or
        # 'Manual' (user-defined uniform value phi_sls).
        'phi_sls_mode': 'Same',
        'phi_sls': 1.0,

        # Traffic UDL (fladelast, DK NA Anneks A.2.2/A.2.3.2). Defined as a
        # LINE load [kN/m] on the analysis strip, like selfweight: the user
        # accounts for the effective width (DK NA intensity is 2.5 kN/m2
        # times the considered width). Off by default (q = 0) so existing
        # sessions are unaffected.
        'udl_q': 0.0,            # line load [kN/m]; 0 deactivates
        'udl_gap': 10.0,         # clear distance vehicle -> UDL [m] in the
                                 # step coupling (preset: DK NA Fig. A.2.2-2)
        'udl_mode': 'Moving',    # 'Moving' (window follows the vehicle in
                                 # the step results) or 'Static' (full deck)
        'udl_footprint': False,  # True: UDL also within the vehicle window
        'gamma_udl': 0.56,       # ULS partial factor (Vejledning Fig. B3.1, LC1)

        # SLS combination factors (Vejledning Fig. B3.2, characteristic
        # combination): permanent 1.0, vehicle A 1.00, vehicle B 0.75,
        # traffic UDL 0.40. KFI is not applied in SLS.
        'sls_g': 1.0, 'sls_j': 1.0,
        'sls_veh': 1.0, 'sls_vehB': 0.75,
        'sls_udl': 0.40,
        # Independent analysis toggles. Both off -> only the unfactored
        # combination is available.
        'analyze_uls': True, 'analyze_sls': True,

        # 0.5 m mesh keeps the (undocumented before v0.48) deflection
        # interpolation error negligible; affordable since the v0.47
        # batched recovery kernel.
        'mesh_size': 0.5,
        'step_size': 0.5,
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
        'phi_linf_mode': 'Determinant',
        'phi_application': 'Per member',
        'phi_manual_scope': 'Global',
        'phi_span_list': [1.0]*10,
        'phi_sls_mode': 'Same',
        'phi_sls': 1.0,
        'udl_q': 0.0,
        'udl_gap': 10.0,
        'udl_mode': 'Moving',
        'udl_footprint': False,
        'gamma_udl': 1.0,
        'sls_g': 1.0, 'sls_j': 1.0,
        'sls_veh': 1.0, 'sls_vehB': 1.0,
        'sls_udl': 1.0,
        'analyze_uls': True, 'analyze_sls': True,

        'scale_manual': 2.0, 
        'mesh_size': 0.5, 'step_size': 0.5,
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
    normalize_vehicle_fields(data, 'vehicle', 'vehicle_loads', 'vehicle_space')
    normalize_vehicle_fields(data, 'vehicleB', 'vehicleB_loads', 'vehicleB_space')

    # 1. Main Config Keys
    st.session_state[f"{sys_key}_md_sel"] = data.get('mode', 'Frame')
    st.session_state[f"{sys_key}_emode"] = "Eurocode (f_ck)" if data.get('e_mode') == "Eurocode" else "Manual (E-Modulus)"
    st.session_state[f"{sys_key}_kfi"] = data.get('KFI', 1.0)
    st.session_state[f"{sys_key}_phim"] = data.get('phi_mode', 'Calculate')
    st.session_state[f"{sys_key}_phiv"] = data.get('phi', 1.0)
    st.session_state[f"{sys_key}_philinf"] = (
        "Per span (DK NA A.2.3.5(2))" if data.get('phi_linf_mode') == 'Span'
        else "Combined system (EN 1991-2 Tab. 6.2)"
    )
    st.session_state[f"{sys_key}_phiapp"] = (
        "Governing value for all members" if data.get('phi_application') == 'Governing'
        else "Per member"
    )
    st.session_state[f"{sys_key}_phiscope"] = (
        "Per span" if data.get('phi_manual_scope') == 'Per span' else "Global"
    )
    phi_spans = data.get('phi_span_list', [1.0]*10)
    for i in range(10):
        if i < len(phi_spans):
            st.session_state[f"{sys_key}_phiv_s{i}"] = phi_spans[i]
    sls_mode = data.get('phi_sls_mode', 'Same')
    sls_map = {'Same': "Same as ULS",
               'Reduced': "Reduced: 1+(Phi-1)/2 (Vejledning 5.4.2)",
               'Manual': "Manual SLS value"}
    st.session_state[f"{sys_key}_phisls"] = sls_map.get(sls_mode, "Same as ULS")
    st.session_state[f"{sys_key}_phislsv"] = data.get('phi_sls', 1.0)

    # Traffic UDL keys
    st.session_state[f"{sys_key}_udlq"] = data.get('udl_q', 0.0)
    st.session_state[f"{sys_key}_udlmode"] = (
        "Static (full deck)" if data.get('udl_mode') == 'Static'
        else "Moving with vehicle"
    )
    st.session_state[f"{sys_key}_udlgap_cust"] = data.get('udl_gap', 10.0)
    # Restore the preset selector too: left stale, it re-applies its old
    # preset on the next rerun and silently overwrites a custom distance.
    st.session_state[f"{sys_key}_udlgap_sel"] = udl_gap_preset_label(data.get('udl_gap', 10.0))
    st.session_state[f"{sys_key}_udlfoot"] = bool(data.get('udl_footprint', False))
    st.session_state[f"{sys_key}_gudl_cust"] = data.get('gamma_udl', 0.56)

    # SLS factor set
    st.session_state[f"{sys_key}_slsg_cust"] = data.get('sls_g', 1.0)
    st.session_state[f"{sys_key}_slsj_cust"] = data.get('sls_j', 1.0)
    st.session_state[f"{sys_key}_slsA_cust"] = data.get('sls_veh', 1.0)
    st.session_state[f"{sys_key}_slsB_cust"] = data.get('sls_vehB', 0.75)
    st.session_state[f"{sys_key}_slsudl_cust"] = data.get('sls_udl', 0.40)
    if sys_key == "sysA":
        st.session_state["uls_toggle_sidebar"] = bool(data.get('analyze_uls', True))
        st.session_state["sls_toggle_sidebar"] = bool(data.get('analyze_sls', True))
    st.session_state[f"{sys_key}_nsp"] = data.get('num_spans', 1)
    
    # 2. Shear Deformation Keys & Analysis Settings
    if sys_key == "sysA":
        st.session_state["shear_toggle_sidebar"] = data.get('use_shear_def', False)
        st.session_state["beff_input_sidebar"] = data.get('b_eff', 1.0)
        st.session_state["nu_input_sidebar"] = data.get('nu', 0.2)
        st.session_state["common_mesh_slider"] = data.get('mesh_size', 0.5)
        st.session_state["common_step_slider"] = data.get('step_size', 0.5)
    
    # 3. Factors
    st.session_state[f"{sys_key}_gg_cust"] = data.get('gamma_g', 1.0)
    st.session_state[f"{sys_key}_gj_cust"] = data.get('gamma_j', 1.0)
    st.session_state[f"{sys_key}_gamA_cust"] = data.get('gamma_veh', 1.0)
    st.session_state[f"{sys_key}_gamB_cust"] = data.get('gamma_vehB', 1.0)

    # 4. Spans & Profiler Keys
    shape_map_rev = {0: "Constant", 1: "Linear (Taper)", 2: "3-Point (Start/Mid/End)"}
    # Note: 1 is now "Height (H)" which is the new default for logic, but UI select logic handles type 0/1 mapping
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
            st.session_state[f"{sys_key}_prof_type_{el_name}"] = type_map_rev.get(g.get('type', 1), "Height (H)")
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
            st.session_state[f"{sys_key}_prof_type_{el_name}"] = type_map_rev.get(g.get('type', 1), "Height (H)")
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
    sync_vehicle_widgets_from_params(sys_key, data)

    # 7. Supports
    # Only clear the custom-spring widget keys. A broad "_k" substring match
    # would also delete unrelated keys such as f"{sys_key}_kfi".
    spring_prefixes = (f"{sys_key}_kx_", f"{sys_key}_ky_", f"{sys_key}_km_")
    supp_keys = [k for k in st.session_state.keys() if k.startswith(spring_prefixes)]
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
    """Loads session state from a DataFrame.

    Returns (loaded, skipped): loaded is False when the frame lacks any of
    the expected System/Parameter/Value columns; skipped lists
    'System/Parameter' labels of rows that could not be parsed, so callers
    can warn instead of silently dropping configuration entries.
    """
    if not {'System', 'Parameter', 'Value'}.issubset(df_load.columns):
        return False, []

    clean_transient_keys()
    skipped = []

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
        except Exception:
            skipped.append(f"{sys_n}/{row.get('Parameter', '?')}")

    if 'sysA' in st.session_state:
        st.session_state['sysA'] = sanitize_input_data(st.session_state['sysA'])
        force_ui_update('sysA', st.session_state['sysA'])
    if 'sysB' in st.session_state:
        st.session_state['sysB'] = sanitize_input_data(st.session_state['sysB'])
        force_ui_update('sysB', st.session_state['sysB'])
    return True, skipped

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
        # Frozen builds wrote autosaves next to the executable before user
        # data moved to %APPDATA%; fall back to that location for reading.
        if not os.path.exists(autosave_path):
            legacy_path = get_legacy_writable_path(AUTOSAVE_FILE)
            if legacy_path and os.path.exists(legacy_path):
                autosave_path = legacy_path
        loaded_from_autosave = False

        if os.path.exists(autosave_path):
            try:
                df_auto = pd.read_csv(autosave_path)
                st.session_state['sysA'] = get_def()
                st.session_state['sysB'] = get_def()
                loaded, skipped = load_data_from_df(df_auto)
                if loaded:
                    loaded_from_autosave = True
                    if skipped:
                        preview = ", ".join(skipped[:5]) + ("..." if len(skipped) > 5 else "")
                        st.session_state['load_status'] = (
                            'warning',
                            f"Autosave restored, but {len(skipped)} entries could not be read "
                            f"and were skipped: {preview}"
                        )
            except Exception:
                st.session_state['load_status'] = (
                    'warning',
                    "An autosave file was found but could not be read. Starting with defaults."
                )

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
