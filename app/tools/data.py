"""Data analyst tools: safe, predefined pandas operations (no arbitrary code execution)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

import pandas as pd
from langchain_core.tools import tool

from app.config import get_settings
from app.context import ctx
from app.tools.files import resolve


def _load(file_path: str) -> pd.DataFrame:
    p = resolve(file_path)
    if p.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(p)
    if p.suffix.lower() == ".json":
        return pd.read_json(p)
    return pd.read_csv(p)


@tool
def data_describe(file_path: str) -> str:
    """Show the columns, data types, row count and first rows of a CSV/Excel/JSON file.
    Call this before data_analyze or chart_make so you know the exact column names."""
    try:
        df = _load(file_path)
    except Exception as e:
        return f"Error: {e}"
    dtypes = ", ".join(f"{c} ({t})" for c, t in df.dtypes.astype(str).items())
    return f"Rows: {len(df)}\nColumns: {dtypes}\n\nFirst rows:\n{df.head(5).to_string(index=False)}"


@tool
def data_analyze(file_path: str,
                 operation: Literal["summary", "sum", "mean", "count", "top_n", "group_sum", "group_mean",
                                    "group_count", "filter_equals", "filter_greater"],
                 column: str | None = None, by: str | None = None, n: int = 5,
                 value: str | None = None) -> str:
    """Answer a question about a CSV/Excel file with one safe operation.
    summary: stats for all columns | sum/mean/count: of `column` | top_n: rows with largest `column`
    group_sum/group_mean/group_count: `column` grouped by `by` | filter_equals: rows where `column` == value
    filter_greater: rows where `column` > value."""
    try:
        df = _load(file_path)
        if operation == "summary":
            res = df.describe(include="all").round(2).to_string()
        elif operation in ("sum", "mean", "count"):
            res = str(getattr(df[column], operation)())
        elif operation == "top_n":
            res = df.nlargest(n, column).to_string(index=False)
        elif operation.startswith("group_"):
            agg = operation.split("_")[1]
            res = df.groupby(by)[column].agg(agg).sort_values(ascending=False).round(2).to_string()
        elif operation == "filter_equals":
            res = df[df[column].astype(str).str.lower() == str(value).lower()].to_string(index=False)
        elif operation == "filter_greater":
            res = df[df[column] > float(value)].to_string(index=False)
        else:
            return f"Error: unknown operation {operation}"
    except KeyError as e:
        return f"Error: column {e} not found. Call data_describe to see the column names."
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
    return res[: get_settings().tool_output_chars]


@tool
def chart_make(file_path: str, chart_type: Literal["bar", "line", "pie", "scatter"], x: str, y: str,
               aggregate: Literal["sum", "mean", "count", "none"] = "sum", title: str = "") -> str:
    """Create a chart (PNG) from a CSV/Excel file. For bar/pie/line, rows are grouped by `x` and `y` is
    aggregated. Returns the image path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401

    try:
        df = _load(file_path)
        data = df.groupby(x)[y].agg(aggregate) if aggregate != "none" and chart_type != "scatter" else df
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if chart_type == "scatter":
            ax.scatter(df[x], df[y], color="#2563eb")
            ax.set_xlabel(x)
            ax.set_ylabel(y)
        elif chart_type == "pie":
            ax.pie(data.values, labels=data.index, autopct="%1.0f%%")
        elif chart_type == "line":
            (data if aggregate != "none" else df.set_index(x)[y]).plot(ax=ax, marker="o", color="#2563eb")
        else:
            (data if aggregate != "none" else df.set_index(x)[y]).sort_values(ascending=False).plot.bar(
                ax=ax, color="#2563eb")
        ax.set_title(title or f"{aggregate} of {y} by {x}")
        if chart_type != "pie":
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
            ax.tick_params(axis="x", labelrotation=0)
        fig.tight_layout()
        folder = Path(get_settings().workspace_dir) / "charts"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"chart_{int(time.time() * 1000)}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
    except KeyError as e:
        return f"Error: column {e} not found. Call data_describe first."
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
    rel = str(path.relative_to(get_settings().workspace_dir))
    ctx().add_file(rel)
    return f"Chart saved: {rel}"
