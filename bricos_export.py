"""Tabular data export for manual processing and QA.

Builds the downloadable Excel workbook / CSV offered under the Tabular
Data tab. Streamlit-free so the assembly is directly testable: the UI
passes parameter dicts and raw solver results, this module returns
DataFrames and serialized bytes.

The Total Envelope export is the full QA package: a Settings sheet with
the load factors and analysis settings, one unfactored results sheet per
validly applied load case, and one Total Envelope sheet per analyzed
result mode (ULS / SLS / Unfactored). The CSV variant carries the same
content as a settings block followed by one long-format results table
with a "Result set" column.
"""
import datetime
import io

import numpy as np
import pandas as pd
# Explicit import: pandas resolves the writer engine by name at runtime,
# which bundlers (PyInstaller) cannot trace through pd.ExcelWriter(...).
import xlsxwriter  # noqa: F401

import bricos_data as data_mod
import bricos_solver as solver

# Load cases exported as unfactored result sheets, in report order.
EXPORT_CASES = ("Selfweight", "Soil", "Surcharge", "Traffic UDL", "Vehicle Envelope")

XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def system_has_case(params, case_key):
    """Whether a load case is validly applied in the given system."""
    if case_key == "Selfweight":
        return any(v > 0 for v in params.get('sw_list', []))
    if case_key == "Soil":
        return bool(params.get('soil'))
    if case_key == "Surcharge":
        return bool(params.get('surcharge'))
    if case_key == "Traffic UDL":
        return data_mod.udl_line_load(params) > 0.0
    if case_key == "Vehicle Envelope":
        return (bool(params.get('vehicle', {}).get('loads'))
                or bool(params.get('vehicleB', {}).get('loads')))
    return True


def case_dataframe(r_dict, sys_name, step_mode=False):
    """Long-format per-position results for one load case and system.

    Mirrors the Tabular Data view: one row per result point with the
    envelope columns, or the single-step force columns in step mode.
    Returns None when the dictionary holds no result points.
    """
    if not r_dict:
        return None
    frames = []
    for eid, dat in r_dict.items():
        x_vals = np.asarray(dat.get('x', []))
        n_pts = len(x_vals)
        if n_pts == 0:
            continue

        def get_arr(key):
            arr = dat.get(key)
            if arr is None:
                return np.zeros(n_pts)
            return np.asarray(arr)

        if step_mode:
            columns = {
                "M [kNm]": get_arr('M'), "V [kN]": get_arr('V'), "N [kN]": get_arr('N'),
                "Def_X [mm]": get_arr('def_x') * 1000, "Def_Y [mm]": get_arr('def_y') * 1000,
            }
        else:
            # Static-only dictionaries (no envelope fields) fall back to
            # the single static field for both bounds.
            is_static_only = 'M_max' not in dat and 'M' in dat

            def env_arr(key):
                if is_static_only:
                    return get_arr(key.replace('_max', '').replace('_min', ''))
                return get_arr(key)

            columns = {
                "M_max [kNm]": env_arr('M_max'), "M_min [kNm]": env_arr('M_min'),
                "V_max [kN]": env_arr('V_max'), "V_min [kN]": env_arr('V_min'),
                "N_max [kN]": env_arr('N_max'), "N_min [kN]": env_arr('N_min'),
                "Def_X_max [mm]": env_arr('def_x_max') * 1000, "Def_X_min [mm]": env_arr('def_x_min') * 1000,
                "Def_Y_max [mm]": env_arr('def_y_max') * 1000, "Def_Y_min [mm]": env_arr('def_y_min') * 1000,
            }

        x_vals, columns = _merge_duplicate_sections(x_vals, columns, step_mode)

        block = {
            "System": np.repeat(sys_name, len(x_vals)),
            "Element": np.repeat(eid, len(x_vals)),
            "Location [m]": x_vals,
        }
        block.update(columns)
        frames.append(pd.DataFrame(block))
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _merge_duplicate_sections(x_vals, columns, step_mode):
    """Collapse the repeated section x-values at mesh sub-element
    boundaries: every interior boundary appears twice in the solver's
    result arrays (the end of one sub-element and the start of the next).

    Envelope columns merge conservatively - max of the maxima, min of
    the minima - since the two rows are the same section's bounds
    evaluated on either side (they differ by float noise, and in the
    coupled traffic envelope by the segment-snapped UDL window).
    Step results drop only near-identical repeats: rows that genuinely
    differ represent a physical discontinuity (e.g. the shear jump at
    an axle) and both sides are kept.
    """
    n = len(x_vals)
    if n < 2:
        return x_vals, columns
    same = np.isclose(x_vals[1:], x_vals[:-1], rtol=0.0, atol=1e-9)
    if not same.any():
        return x_vals, columns

    if step_mode:
        keep = np.ones(n, dtype=bool)
        for i in np.flatnonzero(same):
            identical = all(
                np.isclose(arr[i], arr[i + 1], rtol=1e-7, atol=1e-9)
                for arr in columns.values())
            if identical:
                keep[i + 1] = False
        return x_vals[keep], {k: arr[keep] for k, arr in columns.items()}

    # Group runs of equal x and aggregate each run to one row.
    starts = np.flatnonzero(np.concatenate(([True], ~same)))
    merged = {}
    for name, arr in columns.items():
        reduce = np.minimum.reduceat if "_min" in name else np.maximum.reduceat
        merged[name] = reduce(arr, starts)
    return x_vals[starts], merged


def _phi_text(p, raw):
    """Formatted phi value(s); a range when per-member values exist."""
    members = (raw.get('Phi Members') or {}) if raw else {}
    vals = sorted(set(round(v, 4) for v in members.values()))
    if len(vals) > 1:
        return f"Φ[{vals[0]:.2f}-{vals[-1]:.2f}]"
    if vals:
        return f"{vals[0]:.2f}"
    if p.get('phi_mode') == 'Calculate' and raw:
        return f"{raw.get('phi_calc', 1.0):.2f}"
    return f"{p.get('phi', 1.0):.2f}"


def _vehicle_text(params, struct_key):
    veh = params.get(struct_key) or {}
    loads = veh.get('loads') or []
    if not loads:
        return "none"
    spacing = veh.get('spacing') or []
    cls = data_mod.identify_vehicle_class(loads, spacing)
    loads_txt = ", ".join(f"{v:g}" for v in loads)
    spac_txt = ", ".join(f"{v:g}" for v in spacing)
    return f"{cls} | loads [t]: {loads_txt} | spacing [m]: {spac_txt}"


def _udl_application_text(params):
    if data_mod.udl_line_load(params) <= 0.0:
        return "none"
    if params.get('udl_mode') == 'Static':
        return "Static (full deck)"
    if params.get('udl_footprint', False):
        return "Moving, applied within the vehicle window"
    return f"Moving, clear distance {params.get('udl_gap', 10.0):g} m"


def settings_dataframe(params_A, params_B, raw_A, raw_B, valid_B):
    """Load factors and analysis settings, one row per parameter."""
    systems = [("System A", params_A, raw_A)]
    if valid_B:
        systems.append(("System B", params_B, raw_B))

    analyze_uls = bool(params_A.get('analyze_uls', True))
    analyze_sls = bool(params_A.get('analyze_sls', True))
    ls_txt = ("ULS and SLS" if analyze_uls and analyze_sls else
              "ULS only" if analyze_uls else
              "SLS only" if analyze_sls else
              "none (unfactored combination only)")

    def rows_for(p, raw):
        kfi = p.get('KFI', 1.0)
        shear = ("Enabled (Timoshenko, prismatic members)"
                 if p.get('use_shear_def', False) else "Disabled (Euler-Bernoulli)")
        surch_int = ("Simultaneous: vehicle traffic + surcharge"
                     if p.get('combine_surcharge_vehicle', False)
                     else "Exclusive: envelope of vehicle traffic or surcharge")
        return {
            "System name": p.get('name', ''),
            "Structure mode": p.get('mode', ''),
            "Number of spans": p.get('num_spans', 1),
            "Mesh size [m]": p.get('mesh_size', 0.5),
            "Step size [m]": p.get('step_size', 0.5),
            "Vehicle direction": p.get('vehicle_direction', 'Forward'),
            "Shear deformation": shear,
            "Effective width b_eff [m]": p.get('b_eff', 1.0),
            "Surcharge interaction": surch_int,
            "Limit states analyzed": ls_txt,
            "KFI (consequence class)": kfi,
            "γ_g - Selfweight (ULS)": p.get('gamma_g', 1.0),
            # Presets by label: the '1.0 (No KFI)' option must never leak
            # its raw stored value (1/KFI).
            "γ_j - Soil (ULS)": data_mod.soil_gamma_display(p.get('gamma_j', 1.0), kfi),
            "γ_veh,A - Vehicle A (ULS)": p.get('gamma_veh', 1.0),
            "γ_veh,B - Vehicle B (ULS)": p.get('gamma_vehB', 1.0),
            "γ_UDL - Traffic UDL (ULS)": p.get('gamma_udl', 0.56),
            "Selfweight (SLS)": p.get('sls_g', 1.0),
            "Soil (SLS)": p.get('sls_j', 1.0),
            "Vehicle A (SLS)": p.get('sls_veh', 1.0),
            "Vehicle B (SLS)": p.get('sls_vehB', 1.0),
            "Traffic UDL (SLS)": p.get('sls_udl', 0.40),
            "Dynamic factor Φ": f"{_phi_text(p, raw)} ({p.get('phi_mode', 'Calculate')})",
            "Φ SLS treatment": p.get('phi_sls_mode', 'Same'),
            "Vehicle A": _vehicle_text(p, 'vehicle'),
            "Vehicle B": _vehicle_text(p, 'vehicleB'),
            "Traffic UDL q [kN/m]": data_mod.udl_line_load(p),
            "Traffic UDL application": _udl_application_text(p),
        }

    per_system = [(label, rows_for(p, raw)) for label, p, raw in systems]
    parameters = list(per_system[0][1].keys())
    table = {"Parameter": (["BriCoS version", "Exported"] + parameters)}
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    for label, vals in per_system:
        table[label] = ([data_mod.APP_VERSION, stamp]
                        + [vals[k] for k in parameters])
    return pd.DataFrame(table)


def total_envelope_export(params_A, raw_A, params_B, raw_B, valid_B):
    """The full Total Envelope QA package.

    Returns (settings_df, sheets) where sheets maps sheet name to a
    long-format DataFrame: one unfactored sheet per validly applied load
    case, then the Total Envelope per analyzed result mode.
    """
    systems = []
    if raw_A:
        systems.append((params_A, raw_A))
    if valid_B and raw_B:
        systems.append((params_B, raw_B))

    combos = {}

    def combined(idx, mode):
        key = (idx, mode)
        if key not in combos:
            p, r = systems[idx]
            combos[key] = solver.combine_results(r, p, mode)
        return combos[key]

    sheets = {}
    for case in EXPORT_CASES:
        frames = []
        for idx, (p, _) in enumerate(systems):
            if not system_has_case(p, case):
                continue
            df = case_dataframe(combined(idx, "Unfactored").get(case, {}),
                                p.get('name', f"System {'AB'[idx]}"))
            if df is not None:
                frames.append(df)
        if frames:
            sheets[f"{case} (Unfactored)"] = pd.concat(frames, ignore_index=True)

    modes = []
    if params_A.get('analyze_uls', True):
        modes.append(("Design (ULS)", "Total Envelope (ULS)"))
    if params_A.get('analyze_sls', True):
        modes.append(("Characteristic (SLS)", "Total Envelope (SLS)"))
    modes.append(("Unfactored", "Total Envelope (Unfactored)"))
    for mode, sheet_name in modes:
        frames = []
        for idx, (p, _) in enumerate(systems):
            df = case_dataframe(combined(idx, mode).get("Total Envelope", {}),
                                p.get('name', f"System {'AB'[idx]}"))
            if df is not None:
                frames.append(df)
        if frames:
            sheets[sheet_name] = pd.concat(frames, ignore_index=True)

    settings = settings_dataframe(params_A, params_B, raw_A, raw_B, valid_B)
    return settings, sheets


def to_xlsx_bytes(settings_df, sheets):
    """Serialize the Settings sheet plus result sheets to .xlsx bytes."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        settings_df.to_excel(writer, sheet_name="Settings", index=False)
        _autosize_columns(writer.sheets["Settings"], settings_df)
        for name, df in sheets.items():
            # Excel limits sheet names to 31 characters.
            sheet = name[:31]
            df.to_excel(writer, sheet_name=sheet, index=False)
            ws = writer.sheets[sheet]
            ws.freeze_panes(1, 0)
            _autosize_columns(ws, df)
    return buf.getvalue()


def _autosize_columns(ws, df, max_width=60):
    for ci, col in enumerate(df.columns):
        content_w = 0
        if len(df):
            try:
                content_w = int(df[col].astype(str).str.len().max())
            except Exception:
                content_w = 0
        ws.set_column(ci, ci, min(max(len(str(col)), content_w) + 2, max_width))


def to_csv_bytes(settings_df, sheets):
    """Serialize the same package to one CSV: the settings block, a blank
    line, then a single long-format results table with a "Result set"
    column carrying the sheet name. UTF-8 with BOM so Excel opens the
    Greek symbols correctly; pandas reads it back transparently
    (the results table via skiprows=len(settings)+2)."""
    parts = [settings_df.to_csv(index=False), "\n"]
    frames = []
    for name, df in sheets.items():
        tagged = df.copy()
        tagged.insert(0, "Result set", name)
        frames.append(tagged)
    if frames:
        parts.append(pd.concat(frames, ignore_index=True).to_csv(index=False))
    return "".join(parts).encode("utf-8-sig")
