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
