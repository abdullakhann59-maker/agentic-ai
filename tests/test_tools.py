"""Tool unit tests (no LLM needed except the scripted one for invoices)."""
from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage

from app.scraper import map_fields, structure_records
from app.tools.data import data_analyze, data_describe
from app.tools.documents import invoice_extract, validate_invoice
from app.tools.files import resolve
from app.tools.memory_tools import memory_save
from app.tools.utility import calculator, safe_eval


def test_calculator_is_safe():
    assert safe_eval("18 * 1.18") == pytest.approx(21.24)
    assert "Error" in calculator.invoke({"expression": "__import__('os').system('ls')"})


def test_workspace_sandbox_blocks_path_traversal():
    with pytest.raises(PermissionError):
        resolve("../../etc/passwd")


def test_data_tools():
    assert "region" in data_describe.invoke({"file_path": "sales_data.csv"})
    out = data_analyze.invoke({"file_path": "uploads/sales_data.csv", "operation": "group_sum",
                               "column": "revenue", "by": "region"})
    assert "North" in out and "South" in out
    assert "not found" in data_analyze.invoke({"file_path": "sales_data.csv", "operation": "sum",
                                               "column": "nope"})


def test_field_mapping_and_cleaning():
    assert map_fields(["title", "price_inr", "rating"], ["name", "price"]) == {"name": "title", "price": "price_inr"}
    rows, _ = structure_records([{"Product": "A", "Price": "₹ 1,299"}, {"Product": "A", "Price": "₹ 1,299"}],
                                ["name", "price"])
    assert rows == [{"name": "A", "price": 1299}]          # cleaned + de-duplicated


def test_invoice_validation_math():
    inv = {"invoice_number": "1", "invoice_date": "2026-01-01", "vendor_name": "X", "subtotal": 100,
           "cgst": 9, "sgst": 9, "total_amount": 118,
           "line_items": [{"description": "a", "quantity": 2, "unit_price": 50, "amount": 100}]}
    assert validate_invoice(inv) == []
    inv["total_amount"] = 200
    assert any("total" in w for w in validate_invoice(inv))


def test_invoice_extract_with_scripted_llm(fake_llm):
    data = {"invoice_number": "INV-2026-0147", "invoice_date": "2026-09-15", "vendor_name": "Sharma Electronics",
            "subtotal": 14000, "cgst": 1260, "sgst": 1260, "total_amount": 16520,
            "line_items": [{"description": "Wireless Mouse", "quantity": 10, "unit_price": 650, "amount": 6500},
                           {"description": "USB Keyboard", "quantity": 5, "unit_price": 900, "amount": 4500},
                           {"description": "HDMI Cable", "quantity": 20, "unit_price": 150, "amount": 3000}]}
    fake_llm.replies = [AIMessage(content=json.dumps(data))]
    out = json.loads(invoice_extract.invoke({"file_path": "sample_invoice.pdf"}))
    assert out["validation_warnings"] == []
    assert out["saved_to"].startswith("invoices/")
    # the real PDF text was sent to the model
    assert "INV-2026-0147" in str(fake_llm.calls[0])


def test_memory_refuses_secrets(fake_llm):
    assert "Refused" in memory_save.invoke({"fact": "my password is hunter2"})
    assert "Saved" in memory_save.invoke({"fact": "My manager is Priya"})
