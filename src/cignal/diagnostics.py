"""Structured diagnostics.

Copied from multigen (`src/multigen/frontend/diagnostics.py`); the rule ids are cignal's own.
"""

import ast
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class DiagnosticSeverity(Enum):
    """How a diagnostic affects the validity of the analysed source."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class RuleId:
    """Stable diagnostic identifiers. Append-only: callers filter and test on them."""

    SYNTAX_ERROR = "CIGNAL.SYNTAX_ERROR"
    IMPORT_ERROR = "CIGNAL.IMPORT_ERROR"
    KERNEL_NOT_FOUND = "CIGNAL.KERNEL_NOT_FOUND"
    UNSUPPORTED_SYNTAX = "CIGNAL.UNSUPPORTED_SYNTAX"
    UNSUPPORTED_CALL = "CIGNAL.UNSUPPORTED_CALL"
    IMPLICIT_PRECISION = "CIGNAL.IMPLICIT_PRECISION"
    UNKNOWN_TYPE = "CIGNAL.UNKNOWN_TYPE"
    UNKNOWN_NAME = "CIGNAL.UNKNOWN_NAME"
    UNDECLARED_STATE = "CIGNAL.UNDECLARED_STATE"
    MISSING_DEFAULT = "CIGNAL.MISSING_DEFAULT"
    MISSING_TICK = "CIGNAL.MISSING_TICK"
    BAD_SIGNATURE = "CIGNAL.BAD_SIGNATURE"
    MIXED_NUMERIC = "CIGNAL.MIXED_NUMERIC"
    TYPE_MISMATCH = "CIGNAL.TYPE_MISMATCH"
    CONSTANT_RANGE = "CIGNAL.CONSTANT_RANGE"
    RECURSION = "CIGNAL.RECURSION"
    BAD_PARAM = "CIGNAL.BAD_PARAM"
    UNPROVEN_INDEX = "CIGNAL.UNPROVEN_INDEX"
    BAD_TABLE = "CIGNAL.BAD_TABLE"
    UNBOUNDED_LOOP = "CIGNAL.UNBOUNDED_LOOP"  # a warning
    BLOCK_DEPENDENT = "CIGNAL.BLOCK_DEPENDENT"  # a warning: `control` reads or writes what `tick` writes


@dataclass(frozen=True)
class SourceSpan:
    """A region of source, one-indexed by line and zero-indexed by column."""

    line: int
    column: int = 0
    end_line: int | None = None
    end_column: int | None = None

    @classmethod
    def from_node(cls, node: ast.AST) -> Optional["SourceSpan"]:
        """Build a span from an AST node, or None if it carries no position."""
        line = getattr(node, "lineno", None)
        if line is None:
            return None
        return cls(
            line=line,
            column=getattr(node, "col_offset", 0),
            end_line=getattr(node, "end_lineno", None),
            end_column=getattr(node, "end_col_offset", None),
        )

    def __str__(self) -> str:
        """Render as `line:column`, the form editors and tooling expect."""
        return f"{self.line}:{self.column}"


@dataclass(frozen=True)
class Diagnostic:
    """A single validation finding."""

    rule_id: str
    severity: DiagnosticSeverity
    message: str
    span: SourceSpan | None = None
    node_type: str | None = None
    feature: str | None = None
    remediation: str | None = None
    # Supporting detail such as an inferred type or the offending identifier.
    evidence: dict[str, str] = field(default_factory=dict)

    @property
    def is_error(self) -> bool:
        """Whether this diagnostic invalidates the source."""
        return self.severity is DiagnosticSeverity.ERROR

    def format(self, path: str | None = None) -> str:
        """Render as a conventional compiler diagnostic line."""
        location = path or "<source>"
        if self.span is not None:
            location = f"{location}:{self.span}"
        return f"{location}: {self.severity.value}: [{self.rule_id}] {self.message}"

    def to_dict(self) -> dict[str, Any]:
        """Render as a JSON-serialisable mapping."""
        payload: dict[str, Any] = {
            "rule_id": self.rule_id,
            "severity": self.severity.value,
            "message": self.message,
        }
        if self.span is not None:
            payload["span"] = {
                "line": self.span.line,
                "column": self.span.column,
                "end_line": self.span.end_line,
                "end_column": self.span.end_column,
            }
        for name, value in (
            ("node_type", self.node_type),
            ("feature", self.feature),
            ("remediation", self.remediation),
        ):
            if value is not None:
                payload[name] = value
        if self.evidence:
            payload["evidence"] = dict(self.evidence)
        return payload


def diagnostics_to_json(diagnostics: list[Diagnostic], indent: int | None = 2) -> str:
    """Serialise diagnostics, preserving the order they were reported in."""
    return json.dumps([d.to_dict() for d in diagnostics], indent=indent)


class CompileError(Exception):
    """Raised when compilation stops on one or more error diagnostics."""

    def __init__(self, diagnostics: list[Diagnostic], path: str | None = None) -> None:
        self.diagnostics = diagnostics
        self.path = path
        super().__init__("\n".join(d.format(path) for d in diagnostics))
