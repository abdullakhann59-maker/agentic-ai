"""Small utility tools."""
from __future__ import annotations

import ast
import math
import operator as op

from langchain_core.tools import tool

OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow,
       ast.Mod: op.mod, ast.FloorDiv: op.floordiv, ast.USub: op.neg, ast.UAdd: op.pos}
FUNCS = {"sqrt": math.sqrt, "round": round, "abs": abs, "min": min, "max": max, "log": math.log}


def safe_eval(expr: str) -> float:
    """Evaluate arithmetic without eval(): only numbers, + - * / ** % // and a few functions."""
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
            return OPS[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCS:
            return FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise ValueError("Only arithmetic is allowed")
    return ev(ast.parse(expr.replace("^", "**"), mode="eval"))


@tool
def calculator(expression: str) -> str:
    """Calculate a math expression exactly, e.g. "18 * 1.18" or "sqrt(144) + 5". Use it for any arithmetic."""
    try:
        result = safe_eval(expression)
    except Exception as e:
        return f"Error: {e}"
    return f"{expression} = {round(result, 6)}"
