import ast
from pathlib import Path


ROBOT_SOURCE = Path(__file__).resolve().parents[1] / "3_robot.py"


def load_functions():
    tree = ast.parse(ROBOT_SOURCE.read_text(encoding="utf-8"))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def test_tracking_helpers_are_defined_once():
    tree = ast.parse(ROBOT_SOURCE.read_text(encoding="utf-8"))
    names = [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    ]

    assert names.count("face_center") == 1
    assert names.count("stable_track_name") == 1


def test_stable_track_name_keeps_vote_based_recognition():
    functions = load_functions()
    source = ast.unparse(functions["stable_track_name"])

    assert "Counter" in source
    assert "MIN_VOTES" in source
    assert '"Unknown"' in source


def test_greeting_for_keeps_role_and_name_in_message():
    functions = load_functions()
    source = ast.unparse(functions["greeting_for"])

    assert "role" in source
    assert "name" in source
    assert "Good morning" in source
    assert "Good afternoon" in source
    assert "Good evening" in source
    assert '"Hello"' in source


def test_geometry_features_guards_against_degenerate_face_scale():
    functions = load_functions()
    source = ast.unparse(functions["geometry_features"])

    assert "fw" in source
    assert "ed" in source
    assert "fh" in source
    assert "min(fw, ed, fh)" in source
    assert "return None" in source
