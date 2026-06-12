"""Session save/load (v0.66): the config download must not read
st.session_state from inside the deferred download callable - Streamlit
runs it on a thread WITHOUT script-run context, where session state is
an empty dummy, which silently produced empty save files (the bare-mode
test environment cannot reproduce this, because bare-mode session state
behaves like a normal dict - hence the source-level guard below)."""
import ast
import io
import os

import pandas as pd

import bricos_data as data


def _populated_session():
    st = data.st
    st.session_state.clear()
    pA = data.get_def()
    pA.update({"name": "Roundtrip A", "num_spans": 3, "KFI": 0.9,
               "gamma_j": 1.0 / 0.9, "udl_q": 7.5, "udl_gap": 5.0})
    pA["vehicle_loads"] = "10, 12"
    pA["vehicle_space"] = "0, 1.5"
    data.normalize_vehicle_fields(pA, "vehicle", "vehicle_loads", "vehicle_space")
    pA["span_geom_0"] = {"type": 1, "shape": 2, "vals": [0.6, 0.5, 0.6],
                         "locked": True, "align_type": 0,
                         "incline_mode": 0, "incline_val": 0.0}
    pA["supports"] = [{"type": "Custom Spring", "k": [1e5, 2e5, 0.0]},
                      {"type": "Pinned", "k": [1e14, 1e14, 0.0]},
                      {"type": "Fixed", "k": [1e14, 1e14, 1e14]},
                      {"type": "Fixed", "k": [1e14, 1e14, 1e14]}]
    st.session_state["sysA"] = data.sanitize_input_data(pA)
    st.session_state["sysB"] = data.sanitize_input_data(data.get_def())
    st.session_state["rep_pno"] = "P-123"
    st.session_state["result_mode"] = "Design (ULS)"
    return st


def test_save_load_roundtrip_preserves_the_configuration():
    st = _populated_session()
    blob = data.generate_csv_data()

    st.session_state.clear()
    loaded, skipped = data.load_data_from_df(pd.read_csv(io.BytesIO(blob)))

    assert loaded is True
    assert skipped == []
    p = st.session_state["sysA"]
    assert p["name"] == "Roundtrip A"
    assert p["num_spans"] == 3
    assert p["KFI"] == 0.9
    assert p["vehicle"] == {"loads": [10.0, 12.0], "spacing": [0.0, 1.5]}
    assert p["span_geom_0"]["vals"] == [0.6, 0.5, 0.6]
    assert p["supports"][0] == {"type": "Custom Spring", "k": [1e5, 2e5, 0.0]}
    assert st.session_state["rep_pno"] == "P-123"
    assert st.session_state["result_mode"] == "Design (ULS)"


def test_download_callable_works_without_session_state():
    # The decisive scenario: the callable created at render time must
    # produce the full payload even when session state is gone by the
    # time it runs (Streamlit's download thread has no context).
    st = _populated_session()
    builder = data.session_csv_builder()

    st.session_state.clear()  # simulate the context-less download thread
    blob = builder()

    df = pd.read_csv(io.BytesIO(blob))
    assert {"System", "Parameter", "Value"}.issubset(df.columns)
    assert (df["System"] == "sysA").sum() > 30
    assert "Roundtrip A" in df.loc[df["Parameter"] == "name", "Value"].iloc[0]
    # And it loads back into the (fresh) session.
    loaded, skipped = data.load_data_from_df(df)
    assert loaded is True and skipped == []
    assert st.session_state["sysA"]["name"] == "Roundtrip A"


def test_builder_reflects_in_place_changes_made_after_render():
    # The payload holds live references to the system dicts: edits between
    # the button render and the download click are included, matching the
    # old in-script serialization behaviour.
    st = _populated_session()
    builder = data.session_csv_builder()
    st.session_state["sysA"]["name"] = "Edited After Render"

    df = pd.read_csv(io.BytesIO(builder()))
    assert "Edited After Render" in df.loc[df["Parameter"] == "name", "Value"].iloc[0]


def test_build_session_csv_matches_generate_csv_data():
    _populated_session()
    assert data.build_session_csv(*data.session_csv_payload()) == data.generate_csv_data()


def test_main_passes_a_render_time_payload_to_the_config_download():
    # Source-level guard (bare mode cannot reproduce the missing-context
    # thread): the config download_button must receive a CALL result
    # (payload captured at render), never a bare reference to a function
    # that reads st.session_state when invoked.
    main_path = os.path.join(os.path.dirname(data.__file__), "bricos_main.py")
    tree = ast.parse(open(main_path, encoding="utf-8").read())

    found = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "download_button"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and "Download Configuration" in str(node.args[0].value)):
            found.append(node)
    assert len(found) == 1
    data_arg = found[0].args[1]
    assert isinstance(data_arg, ast.Call), (
        "the config download data must be session_csv_builder() - a bare "
        "function reference would read the empty dummy session state on "
        "the download thread")
    assert getattr(data_arg.func, "attr", "") == "session_csv_builder"
