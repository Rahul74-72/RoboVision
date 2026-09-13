import ast
from pathlib import Path


ROBOT_SOURCE = Path(__file__).resolve().parents[1] / "3_robot.py"


def test_recognize_passes_geometry_statistics_to_shared_normalizer():
    tree = ast.parse(ROBOT_SOURCE.read_text(encoding="utf-8"))
    recognize = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "recognize"
    )

    call = next(
        node
        for node in ast.walk(recognize)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "normalize_geometry"
    )

    positional = [ast.unparse(arg) for arg in call.args]
    assert positional == ["geometry", "geometry_mean", "geometry_std"]
