import bricos_data as data


def test_valid_default_model_returns_no_validation_errors():
    assert data.validate_analysis_inputs(data.get_def(), "System A") == []


def test_invalid_frame_wall_section_height_is_blocked():
    params = data.get_def()
    params["mode"] = "Frame"
    params["num_spans"] = 1
    params["h_list"][0] = 5.0
    params["wall_geom_0"] = {"type": 1, "shape": 0, "vals": [0.0, 0.0, 0.0]}

    errors = data.validate_analysis_inputs(params, "System A")

    assert errors
    assert any("Wall W1" in e and "height" in e for e in errors)


def test_invalid_beam_span_section_height_is_blocked():
    params = data.get_def()
    params["mode"] = "Beam"
    params["num_spans"] = 1
    params["span_geom_0"] = {"type": 1, "shape": 0, "vals": [0.0, 0.0, 0.0]}

    errors = data.validate_analysis_inputs(params, "System A")

    assert errors
    assert any("Span S1" in e and "height" in e for e in errors)


def test_inertia_based_span_with_zero_inertia_is_blocked():
    params = data.get_def()
    params["mode"] = "Beam"
    params["num_spans"] = 1
    params["span_geom_0"] = {"type": 0, "shape": 0, "vals": [0.0, 0.0, 0.0]}

    errors = data.validate_analysis_inputs(params, "System B")

    assert errors
    assert any("Span S1" in e and "inertia" in e for e in errors)


def test_nonpositive_effective_width_is_blocked():
    params = data.get_def()
    params["b_eff"] = 0.0

    errors = data.validate_analysis_inputs(params, "System A")

    assert any("b_eff" in e for e in errors)


def test_nonpositive_span_e_modulus_is_blocked():
    params = data.get_def()
    params["num_spans"] = 1
    params["E_span_list"][0] = 0.0

    errors = data.validate_analysis_inputs(params, "System A")

    assert any("Span S1" in e and "E modulus" in e for e in errors)


def test_vehicle_mismatched_loads_and_spacing_is_blocked():
    params = data.get_def()
    params["vehicle"] = {"loads": [1.0, 2.0, 3.0], "spacing": [0.0, 1.0]}

    errors = data.validate_analysis_inputs(params, "System A")

    assert any("Vehicle A has 3 axle loads but 2 spacing values" in e for e in errors)


def test_vehicle_spacing_must_start_with_zero():
    params = data.get_def()
    params["vehicle"] = {"loads": [1.0], "spacing": [1.0]}

    errors = data.validate_analysis_inputs(params, "System A")

    assert any("Vehicle A spacing must start" in e for e in errors)


def test_empty_vehicle_is_accepted():
    params = data.get_def()
    params["vehicle"] = {"loads": [], "spacing": []}
    params["vehicleB"] = {"loads": [], "spacing": []}
    params["step_size"] = 0.0

    errors = data.validate_analysis_inputs(params, "System A")

    assert not any("Vehicle" in e for e in errors)
    assert not any("step size" in e.lower() for e in errors)


def test_linear_taper_validates_start_and_end_not_mid():
    params = data.get_def()
    params["mode"] = "Beam"
    params["num_spans"] = 1
    params["span_geom_0"] = {"type": 1, "shape": 1, "vals": [0.50, 0.0, 0.60]}

    errors = data.validate_analysis_inputs(params, "System A")

    assert not any("Span S1" in e and "height" in e for e in errors)


def _geometry_errors(errors):
    return [e for e in errors if "section" in e]


def test_inertia_based_span_below_minimum_inertia_is_blocked_as_inertia():
    params = data.get_def()
    params["mode"] = "Beam"
    params["num_spans"] = 1
    params["span_geom_0"] = {"type": 0, "shape": 0, "vals": [9.0e-9, 9.0e-9, 9.0e-9]}

    errors = data.validate_analysis_inputs(params, "System A")

    assert any("Span S1" in e and "inertia" in e for e in errors)
    assert not any("Span S1" in e and "height" in e for e in errors)


def test_inertia_based_span_above_minimum_inertia_has_no_section_geometry_error():
    params = data.get_def()
    params["mode"] = "Beam"
    params["num_spans"] = 1
    params["span_geom_0"] = {"type": 0, "shape": 0, "vals": [1.1e-8, 1.1e-8, 1.1e-8]}

    errors = data.validate_analysis_inputs(params, "System A")

    assert not _geometry_errors(errors)


def test_height_based_span_below_minimum_height_is_still_blocked_as_height():
    params = data.get_def()
    params["mode"] = "Beam"
    params["num_spans"] = 1
    params["span_geom_0"] = {"type": 1, "shape": 0, "vals": [0.049, 0.049, 0.049]}

    errors = data.validate_analysis_inputs(params, "System A")

    assert any("Span S1" in e and "height" in e for e in errors)
    assert not any("Span S1" in e and "inertia" in e for e in errors)
