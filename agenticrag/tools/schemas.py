from __future__ import annotations

from typing import Any


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search indexed knowledge-base chunks for one or more queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "minItems": 1,
                        "maxItems": 5,
                        "description": "Search queries to run against the retriever.",
                    }
                },
                "required": ["queries"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find",
            "description": "Find case-insensitive substring matches in a referenced source document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reference_id": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Reference id returned by search.",
                    },
                    "patterns": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "minItems": 1,
                        "maxItems": 10,
                        "description": "Literal substrings to search for.",
                    },
                },
                "required": ["reference_id", "patterns"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open",
            "description": "Open numbered source lines around a reference.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reference_id": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Reference id returned by search.",
                    },
                    "line_number": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 0,
                        "description": "One-based line number to center, or 0 to use the reference line.",
                    },
                },
                "required": ["reference_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize",
            "description": "Compress prior tool results unrelated to retained references.",
            "parameters": {
                "type": "object",
                "properties": {
                    "candidate_reference_ids": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "maxItems": 20,
                        "description": "Reference ids that should remain available in full detail.",
                    }
                },
                "required": ["candidate_reference_ids"],
                "additionalProperties": False,
            },
        },
    },
]
