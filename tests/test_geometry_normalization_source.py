import ast
from pathlib import Path


ROBOT_SOURCE = Path(__file__).resolve().parents[1] / "3_robot.py"


def test_recognize_masks_zero_geometry_std_before_division():
    tree = ast.parse(ROBOT_SOURCE.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "recognize"
    )

    source = ast.unparse(function)

    assert "valid_std" in source
    assert "np.divide" in source
    assert "where=valid_std" in source
    assert "z = l2(z)" in source
