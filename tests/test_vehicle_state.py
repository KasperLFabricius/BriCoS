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
