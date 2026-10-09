"""Document tools: read documents, OCR images, extract invoice fields."""
from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import tool

from app.config import get_settings
from app.context import ctx
from app.doc_text import read_document
from app.llm import get_chat_model, token_usage
from app.tools.files import resolve


@tool
def doc_read(file_path: str) -> str:
    """Read a PDF, DOCX, TXT or MD file from the workspace and return its text (trimmed).
    For questions about documents already added to the knowledge base, the relevant parts are
    given to you automatically, so you usually do not need this tool for them."""
    try:
        pages = read_document(resolve(file_path))
    except Exception as e:
        return f"Error: {e}"
    text = "\n".join(f"--- page {n} ---\n{t}" for n, t in pages)
    return text[: get_settings().tool_output_chars * 2]


def _ocr(path: Path) -> str:
    try:
        import easyocr  # optional, heavy dependency (PyTorch)
    except ImportError:
        raise RuntimeError("OCR needs the optional package easyocr: pip install -r requirements-ocr.txt")
    import cv2

    img = cv2.imread(str(path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    results = reader.readtext(gray)
    # rebuild reading order: sort by line (y) then x
    results.sort(key=lambda r: (round(r[0][0][1] / 15), r[0][0][0]))
    return "\n".join(r[1] for r in results)


@tool
def ocr_image(file_path: str) -> str:
    """Read the text from an image or scanned document (PNG, JPG, TIFF) with OCR."""
    try:
        return _ocr(resolve(file_path))[: get_settings().tool_output_chars]
    except Exception as e:
        return f"Error: {e}"


INVOICE_PROMPT = """You extract invoice data. Use ONLY the invoice text below. Never invent values.
Return JSON with these keys (null if missing): invoice_number, invoice_date (YYYY-MM-DD), due_date,
vendor_name, vendor_gstin, buyer_name, buyer_gstin, currency, line_items (list of
{{description, quantity, unit_price, amount}}), subtotal, cgst, sgst, igst, total_amount.
Numbers without currency symbols or commas.

INVOICE TEXT:
{text}"""


def validate_invoice(inv: dict) -> list[str]:
    warnings = []
    def num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None
    items = inv.get("line_items") or []
    sub = num(inv.get("subtotal"))
    if items and sub is not None:
        s = sum(num(i.get("amount")) or 0 for i in items)
        if abs(s - sub) > 1:
            warnings.append(f"Line items add up to {s:.2f} but subtotal is {sub:.2f}")
    for i in items:
        q, p, a = num(i.get("quantity")), num(i.get("unit_price")), num(i.get("amount"))
        if None not in (q, p, a) and abs(q * p - a) > 1:
            warnings.append(f"'{i.get('description')}': {q} x {p} != {a}")
    taxes = sum(num(inv.get(k)) or 0 for k in ("cgst", "sgst", "igst"))
    total = num(inv.get("total_amount"))
    if sub is not None and total is not None and abs(sub + taxes - total) > 1:
        warnings.append(f"Subtotal {sub} + tax {taxes} != total {total}")
    if num(inv.get("cgst")) and num(inv.get("sgst")) and abs(num(inv["cgst"]) - num(inv["sgst"])) > 0.5:
        warnings.append("CGST and SGST should be equal")
    for k in ("invoice_number", "invoice_date", "vendor_name", "total_amount"):
        if not inv.get(k):
            warnings.append(f"Missing required field: {k}")
    return warnings


@tool
def invoice_extract(file_path: str) -> str:
    """Extract structured fields (invoice number, dates, vendor, GSTIN, line items, taxes, total) from an
    invoice PDF or image, check the math, and save the result as JSON in the workspace."""
    try:
        path = resolve(file_path)
        if path.suffix.lower() == ".pdf":
            text = "\n".join(t for _, t in read_document(path))
            if len(text.strip()) < 50:
                return "Error: this PDF has no text layer (scanned). Convert a page to an image and use OCR."
        elif path.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            text = _ocr(path)
        else:
            text = "\n".join(t for _, t in read_document(path))
    except Exception as e:
        return f"Error: {e}"
    llm = get_chat_model(json_mode=True)
    msg = llm.invoke(INVOICE_PROMPT.format(text=text[:6000]))
    ctx().add_tokens(*token_usage(msg))
    try:
        data = json.loads(msg.content)
    except (ValueError, TypeError):
        return f"Error: the model did not return valid JSON. Raw output: {str(msg.content)[:300]}"
    warnings = validate_invoice(data)
    out_dir = Path(get_settings().workspace_dir) / "invoices"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{path.stem}.json"
    out.write_text(json.dumps({"data": data, "validation_warnings": warnings}, indent=2))
    rel = str(out.relative_to(get_settings().workspace_dir))
    ctx().add_file(rel)
    return json.dumps({"saved_to": rel, "data": data, "validation_warnings": warnings}, default=str)
