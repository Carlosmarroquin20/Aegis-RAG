"""
Unit tests for the shared HTML tag stripper.

Two failure directions matter equally: markup that survives (an XSS vector when
LLM output is rendered) and prose that is deleted because it merely contains
angle brackets (silent corruption of queries and answers).
"""

from __future__ import annotations

import time

import pytest

from aegis.domain.models.query import RawQuery
from aegis.infrastructure.security.markup import strip_html_tags
from aegis.infrastructure.security.output_sanitizer import OutputSanitizer
from aegis.infrastructure.security.security_gateway import SecurityGateway


class TestMarkupIsRemoved:
    @pytest.mark.parametrize(
        ("markup", "expected"),
        [
            ("<b>bold</b>", "bold"),
            ("a<br/>b", "ab"),
            ('<a href="https://x.test" onclick="steal()">link</a>', "link"),
            ("<script>alert(1)</script>", "alert(1)"),
            ("<SCRIPT SRC=//x.test/a.js></SCRIPT>", ""),
            ("<img src=x onerror=alert(1)>", ""),
            ("before<!-- hidden -->after", "beforeafter"),
            ("<!DOCTYPE html>text", "text"),
            ("<?xml version='1.0'?>text", "text"),
            ("<svg/onload=alert(1)>", ""),
        ],
    )
    def test_strips_tags(self, markup: str, expected: str) -> None:
        assert strip_html_tags(markup) == expected


class TestProseIsPreserved:
    @pytest.mark.parametrize(
        "text",
        [
            "revenue < 5M and margin > 20%",
            "is 3 < 5 and 7 > 2?",
            "if x <= 10 and y >= 2",
            "a <- b -> c",
            "5 <> 6",
            "use the << and >> operators",
            "temperature < 0 means frozen",
        ],
    )
    def test_comparison_operators_are_not_markup(self, text: str) -> None:
        assert strip_html_tags(text) == text

    def test_gateway_keeps_comparisons_in_the_forwarded_query(self) -> None:
        query = "Which products have a price < 50 and a rating > 4?"
        result = SecurityGateway().evaluate(RawQuery(text=query))
        assert result.sanitized_query is not None
        assert result.sanitized_query.text == query

    def test_output_sanitizer_keeps_comparisons_in_the_answer(self) -> None:
        answer = "The threshold applies when latency < 200 ms and uptime > 99.9%."
        assert OutputSanitizer().sanitize(answer).text == answer


class TestCostIsBounded:
    def test_unterminated_openers_do_not_backtrack_quadratically(self) -> None:
        payload = "<a " * 2700  # ~8 KB of openers that never close
        started = time.perf_counter()
        strip_html_tags(payload)
        assert time.perf_counter() - started < 0.25
