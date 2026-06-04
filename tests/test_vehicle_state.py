import bricos_data as data


def test_parse_valid_vehicle_text():
    vehicle, errors = data.parse_vehicle_text("10, 20", "0, 1.5")

    assert errors == []
    assert vehicle == {"loads": [10.0, 20.0], "spacing": [0.0, 1.5]}


def test_parse_vehicle_text_rejects_mismatched_lengths():
    vehicle, errors = data.parse_vehicle_text("10, 20", "0")

    assert vehicle == {"loads": [], "spacing": []}
    assert any("2 axle loads but 1 spacing values" in e for e in errors)


def test_parse_vehicle_text_rejects_spacing_not_starting_with_zero():
    vehicle, errors = data.parse_vehicle_text("10", "1")

    assert vehicle == {"loads": [], "spacing": []}
    assert any("start with 0.0" in e for e in errors)


def test_parse_vehicle_text_rejects_negative_incremental_spacing():
    vehicle, errors = data.parse_vehicle_text("10, 20", "0, -1.5")

    assert vehicle == {"loads": [], "spacing": []}
    assert any("finite and non-negative" in e for e in errors)


def test_parse_vehicle_text_rejects_non_numeric_spacing():
    vehicle, errors = data.parse_vehicle_text("10, 20", "0, bad")

    assert vehicle == {"loads": [], "spacing": []}
    assert any("non-numeric" in e for e in errors)


def test_parse_vehicle_text_accepts_incremental_spacing_that_is_not_nondecreasing():
    vehicle, errors = data.parse_vehicle_text(
        "7.0, 7.0, 9.5, 9.5, 17.8, 17.8",
        "0, 1.4, 3.2, 1.4, 6.0, 1.4",
    )

    assert errors == []
    assert vehicle["loads"] == [7.0, 7.0, 9.5, 9.5, 17.8, 17.8]
    assert vehicle["spacing"] == [0.0, 1.4, 3.2, 1.4, 6.0, 1.4]


def test_parse_empty_vehicle_text_returns_inactive_vehicle_without_error():
    vehicle, errors = data.parse_vehicle_text("", "")

    assert errors == []
    assert vehicle == {"loads": [], "spacing": []}


def test_clear_vehicle_definition_requires_explicit_state_change():
    params = {
        "vehicle": {"loads": [10.0], "spacing": [0.0]},
        "vehicle_loads": "10",
        "vehicle_space": "0",
    }

    data.clear_vehicle_definition(params, "vehicle", "vehicle_loads", "vehicle_space")

    assert params["vehicle"] == {"loads": [], "spacing": []}
    assert params["vehicle_loads"] == ""
    assert params["vehicle_space"] == ""



def _class_100_params():
    params = data.get_def()
    veh = data.load_vehicle_from_csv("Class 100")
    assert veh is not None

    params["vehicle"] = {"loads": veh["loads"], "spacing": veh["spacing"]}
    params["vehicle_loads"] = veh["l_str"]
    params["vehicle_space"] = veh["s_str"]

    params["vehicleB"] = {"loads": [], "spacing": []}
    params["vehicleB_loads"] = ""
    params["vehicleB_space"] = ""

    return params


def test_force_ui_update_preserves_copied_vehicle_and_class_state():
    st = data.st
    st.session_state.clear()
    copied = _class_100_params()

    data.force_ui_update("sysB", copied)

    assert copied["vehicle"]["loads"]
    assert copied["vehicle_loads"]
    assert copied["vehicle_space"]
    assert st.session_state["sysB_A_loads_input"] == copied["vehicle_loads"]
    assert st.session_state["sysB_A_space_input"] == copied["vehicle_space"]
    assert st.session_state["sysB_vehA_class"] == "Class 100"
    assert st.session_state["sysB_vehA_class_last"] == "Class 100"


def test_force_ui_update_ignores_stale_blank_widget_keys_for_non_empty_vehicle():
    st = data.st
    st.session_state.clear()
    copied = _class_100_params()
    st.session_state["sysB_A_loads_input"] = ""
    st.session_state["sysB_A_space_input"] = ""

    data.force_ui_update("sysB", copied)

    assert copied["vehicle"]["loads"]
    assert copied["vehicle_loads"]
    assert copied["vehicle_space"]
    assert st.session_state["sysB_A_loads_input"] == copied["vehicle_loads"]
    assert st.session_state["sysB_A_space_input"] == copied["vehicle_space"]


def test_signature_refresh_overwrites_stale_widget_values_after_state_change():
    st = data.st
    st.session_state.clear()
    params = _class_100_params()
    old_sig = data.vehicle_state_signature(params, "vehicle", "vehicle_loads", "vehicle_space")
    st.session_state["sysB_A_vehicle_sig"] = old_sig
    st.session_state["sysB_A_loads_input"] = "1"
    st.session_state["sysB_A_space_input"] = "0"

    data.apply_standard_vehicle_definition(params, "Class 150", "vehicle", "vehicle_loads", "vehicle_space")
    data.sync_vehicle_widgets_from_params("sysB", params)

    assert st.session_state["sysB_A_loads_input"] == params["vehicle_loads"]
    assert st.session_state["sysB_A_space_input"] == params["vehicle_space"]
    assert st.session_state["sysB_A_vehicle_sig"] == data.vehicle_state_signature(
        params, "vehicle", "vehicle_loads", "vehicle_space"
    )
    assert st.session_state["sysB_A_loads_input"] != "1"


def test_normal_user_vehicle_text_edit_updates_calculation_dictionary():
    params = _class_100_params()
    params["vehicle_loads"] = "10, 20"
    params["vehicle_space"] = "0, 1.5"

    errors = data.normalize_vehicle_fields(params, "vehicle", "vehicle_loads", "vehicle_space")

    assert errors == []
    assert params["vehicle"] == {"loads": [10.0, 20.0], "spacing": [0.0, 1.5]}
    assert data.vehicle_state_signature(params, "vehicle", "vehicle_loads", "vehicle_space")


def test_copy_a_to_b_helper_preserves_vehicle_text_object_and_class_widgets():
    st = data.st
    st.session_state.clear()
    source = _class_100_params()

    copied = data.copy_system_data(source, "System B")
    data.force_ui_update("sysB", copied)

    assert copied["vehicle"] == source["vehicle"]
    assert copied["vehicle_loads"] == source["vehicle_loads"]
    assert copied["vehicle_space"] == source["vehicle_space"]
    assert st.session_state["sysB_vehA_class"] == "Class 100"


def test_copy_b_to_a_helper_preserves_vehicle_text_object_and_class_widgets():
    st = data.st
    st.session_state.clear()
    source = _class_100_params()

    copied = data.copy_system_data(source, "System A")
    data.force_ui_update("sysA", copied)

    assert copied["vehicle"] == source["vehicle"]
    assert copied["vehicle_loads"] == source["vehicle_loads"]
    assert copied["vehicle_space"] == source["vehicle_space"]
    assert st.session_state["sysA_vehA_class"] == "Class 100"


def test_toggling_use_shear_def_after_copy_does_not_change_vehicle_fields():
    copied_a = data.copy_system_data(_class_100_params(), "System A")
    copied_b = data.copy_system_data(_class_100_params(), "System B")
    before_a = (copied_a["vehicle"].copy(), copied_a["vehicle_loads"], copied_a["vehicle_space"])
    before_b = (copied_b["vehicle"].copy(), copied_b["vehicle_loads"], copied_b["vehicle_space"])

    copied_a["use_shear_def"] = not copied_a.get("use_shear_def", False)
    copied_b["use_shear_def"] = copied_a["use_shear_def"]
    data.normalize_vehicle_fields(copied_a, "vehicle", "vehicle_loads", "vehicle_space")
    data.normalize_vehicle_fields(copied_b, "vehicle", "vehicle_loads", "vehicle_space")

    assert (copied_a["vehicle"], copied_a["vehicle_loads"], copied_a["vehicle_space"]) == before_a
    assert (copied_b["vehicle"], copied_b["vehicle_loads"], copied_b["vehicle_space"]) == before_b


def test_selecting_custom_preserves_current_vehicle_text_and_object():
    params = _class_100_params()
    before = params.copy()
    # The UI Custom branch intentionally performs no mutation; normalizing must preserve state.
    errors = data.normalize_vehicle_fields(params, "vehicle", "vehicle_loads", "vehicle_space")

    assert errors == []
    assert params["vehicle"] == before["vehicle"]
    assert params["vehicle_loads"] == before["vehicle_loads"]
    assert params["vehicle_space"] == before["vehicle_space"]


def test_selecting_standard_vehicle_replaces_text_and_object_with_library_definition():
    params = {"vehicle": {"loads": [1.0], "spacing": [0.0]}, "vehicle_loads": "1", "vehicle_space": "0"}

    errors = data.apply_standard_vehicle_definition(params, "Class 100", "vehicle", "vehicle_loads", "vehicle_space")
    expected = _class_100_params()

    assert errors == []
    assert params["vehicle"] == expected["vehicle"]
    assert params["vehicle_loads"] == expected["vehicle_loads"]
    assert params["vehicle_space"] == expected["vehicle_space"]


def test_invalid_vehicle_text_is_preserved_and_blocks_validation():
    params = {"vehicle": {"loads": [10.0], "spacing": [0.0]}, "vehicle_loads": "10, bad", "vehicle_space": "0, 1"}

    errors = data.normalize_vehicle_fields(params, "vehicle", "vehicle_loads", "vehicle_space")
    validation_errors = data.validate_analysis_inputs(params, "System A")

    assert errors
    assert params["vehicle_loads"] == "10, bad"
    assert params["vehicle_space"] == "0, 1"
    assert params["vehicle"] == {"loads": [10.0], "spacing": [0.0]}
    assert any("Vehicle A text input is invalid" in e for e in validation_errors)


def test_empty_vehicle_text_is_inactive_when_object_is_empty():
    params = {"vehicle": {"loads": [], "spacing": []}, "vehicle_loads": "", "vehicle_space": ""}

    errors = data.normalize_vehicle_fields(params, "vehicle", "vehicle_loads", "vehicle_space")

    assert errors == []
    assert params["vehicle"] == {"loads": [], "spacing": []}
    assert params["vehicle_loads"] == ""
    assert params["vehicle_space"] == ""
    assert "_vehicle_text_errors" not in params
