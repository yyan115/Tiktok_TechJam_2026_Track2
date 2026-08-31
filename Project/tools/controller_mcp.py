#!/usr/bin/env python3
"""Minimal stdio MCP bridge to the narrow Track 2 controller capability.

The researcher gets this bridge instead of a Bash tool.  It exposes four
controller operations plus one local, read-only hash helper for binding a
staged candidate into its attempt card.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


MAX_MESSAGE_BYTES = 4 * 1024 * 1024
SERVER_NAME = "track2-controller"
SERVER_VERSION = "2.0.0"
MAX_JSONRPC_ID_TEXT = 128


class McpError(RuntimeError):
    pass


def _strict_object(payload: bytes, label: str) -> dict[str, Any]:
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise McpError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def no_constants(value):
        raise ValueError(f"non-finite constant {value}")

    try:
        value = json.loads(
            payload, object_pairs_hook=no_duplicates, parse_constant=no_constants
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise McpError(f"{label} is not strict JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise McpError(f"{label} must be one object")
    return value


def _canonical(value: Any) -> bytes:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise McpError(f"MCP output is not finite canonical JSON: {exc}") from exc
    if len(payload) > MAX_MESSAGE_BYTES:
        raise McpError("MCP output exceeds its byte limit")
    return payload


def _load_control():
    installed = Path("/control/control.py")
    path = installed if installed.is_file() else Path(__file__).with_name("control.py")
    if path.is_symlink() or not path.is_file():
        raise McpError("fixed controller client is unavailable")
    spec = importlib.util.spec_from_file_location("track2_control_client", path)
    if spec is None or spec.loader is None:
        raise McpError("fixed controller client cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tool_definitions() -> list[dict[str, Any]]:
    empty = {"type": "object", "properties": {}, "additionalProperties": False}
    return [
        {
            "name": "hash_solution",
            "title": "Hash staged solution",
            "description": (
                "Safely read one staged Project/solutions/*.py file and return "
                "its exact SHA-256 and byte size. This is local and consumes no "
                "official attempt."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "solution": {
                        "type": "string",
                        "pattern": r"^Project/solutions/[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.py$",
                    },
                },
                "required": ["solution"],
                "additionalProperties": False,
            },
            "annotations": {
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": False,
            },
        },
        {
            "name": "log",
            "title": "Read official run state",
            "description": (
                "Read the controller's bounded state summary. This does not consume "
                "an attempt and returns no validation labels or hidden predictions."
            ),
            "inputSchema": empty,
            "annotations": {
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": False,
            },
        },
        {
            "name": "run",
            "title": "Submit one official attempt",
            "description": (
                "Commit one candidate/card pair and ask the frozen controller to "
                "consume exactly one official attempt. Use only after both files are "
                "complete. Terminal and policy decisions cannot be overridden."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "solution": {
                        "type": "string",
                        "pattern": r"^Project/solutions/[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.py$",
                    },
                    "card": {
                        "type": "string",
                        "pattern": r"^Project/research/attempts/[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.json$",
                    },
                },
                "required": ["solution", "card"],
                "additionalProperties": False,
            },
            "annotations": {
                "readOnlyHint": False,
                "destructiveHint": True,
                "idempotentHint": False,
                "openWorldHint": False,
            },
        },
        {
            "name": "retry",
            "title": "Retry pending exact request",
            "description": (
                "Replay only the exact durable pending request and request ID after "
                "an ambiguous transport failure. Never creates a new request."
            ),
            "inputSchema": empty,
            "annotations": {
                "readOnlyHint": False,
                "destructiveHint": True,
                "idempotentHint": True,
                "openWorldHint": False,
            },
        },
        {
            "name": "recover",
            "title": "Read durable completed response",
            "description": (
                "Return the locally persisted response for the last completed request. "
                "This operation never sends a completed run to the server again."
            ),
            "inputSchema": empty,
            "annotations": {
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": False,
            },
        },
    ]


def _exact_arguments(value: Any, expected: set[str]) -> dict[str, Any]:
    if value is None:
        value = {}
    if not isinstance(value, dict) or set(value) != expected:
        raise McpError("tool arguments have missing or extra fields")
    return value


def call_tool(name: Any, arguments: Any, control=None) -> dict[str, Any]:
    if not isinstance(name, str):
        raise McpError("tool name must be text")
    control = _load_control() if control is None else control
    if name == "hash_solution":
        args = _exact_arguments(arguments, {"solution"})
        if not isinstance(args["solution"], str):
            raise McpError("solution path must be text")
        response = control.hash_solution(args["solution"])
    else:
        socket_value = os.environ.get(control.SOCKET_ENV)
        if not socket_value:
            raise McpError("controller socket capability is not configured")
        socket_path = Path(socket_value)
        if name == "log":
            _exact_arguments(arguments, set())
            request = control.build_request("log")
            response = control.issue_request(socket_path, request)
        elif name == "run":
            args = _exact_arguments(arguments, {"solution", "card"})
            if not all(isinstance(args[key], str) for key in ("solution", "card")):
                raise McpError("run paths must be text")
            request = control.build_request(
                "run", solution=args["solution"], card=args["card"]
            )
            response = control.issue_request(socket_path, request)
        elif name == "retry":
            _exact_arguments(arguments, set())
            response = control.retry_persisted(
                socket_path, recover_completed=False
            )
        elif name == "recover":
            _exact_arguments(arguments, set())
            response = control.retry_persisted(
                socket_path, recover_completed=True
            )
        else:
            raise McpError("unknown controller tool")
    serialized = _canonical(response).decode("utf-8")
    return {
        "content": [{"type": "text", "text": serialized}],
        "isError": (
            response.get("ok") is not True if name != "hash_solution" else False
        ),
    }


def handle_message(message: dict[str, Any], control=None) -> dict[str, Any] | None:
    if message.get("jsonrpc") != "2.0":
        raise McpError("unsupported JSON-RPC version")
    method = message.get("method")
    request_id = message.get("id")
    if request_id is not None and (
        isinstance(request_id, bool)
        or not isinstance(request_id, (str, int))
        or (isinstance(request_id, str) and len(request_id) > MAX_JSONRPC_ID_TEXT)
    ):
        raise McpError("JSON-RPC id must be text or integer")
    if request_id is None:
        if not isinstance(method, str) or not method.startswith("notifications/"):
            raise McpError("only notifications may omit an id")
        return None
    if method == "initialize":
        params = message.get("params")
        version = params.get("protocolVersion") if isinstance(params, dict) else None
        if not isinstance(version, str) or re.fullmatch(r"20[0-9]{2}-[0-9]{2}-[0-9]{2}", version) is None:
            raise McpError("initialize has an invalid protocol version")
        result = {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            "instructions": (
                "This is the only execution capability. Use hash_solution to bind "
                "staged bytes, log for state, run once "
                "per intended official attempt, retry only after ambiguous transport "
                "failure, and recover only to read a durable completed response."
            ),
        }
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        params = message.get("params", {})
        if not isinstance(params, dict) or any(key != "cursor" for key in params):
            raise McpError("tools/list has invalid parameters")
        if params.get("cursor") not in {None, ""}:
            raise McpError("tools/list cursor is unsupported")
        result = {"tools": tool_definitions()}
    elif method == "tools/call":
        params = message.get("params")
        if not isinstance(params, dict) or not set(params).issubset(
            {"name", "arguments", "_meta"}
        ) or "name" not in params:
            raise McpError("tools/call has invalid parameters")
        try:
            result = call_tool(params["name"], params.get("arguments", {}), control)
        except Exception as exc:
            result = {
                "content": [{
                    "type": "text",
                    "text": f"controller tool refused: {type(exc).__name__}: {exc}"[:16000],
                }],
                "isError": True,
            }
    else:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": "method not found"},
        }
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def main() -> int:
    while True:
        raw = sys.stdin.buffer.readline(MAX_MESSAGE_BYTES + 1)
        if not raw:
            return 0
        if len(raw) > MAX_MESSAGE_BYTES or not raw.endswith(b"\n"):
            raise McpError("MCP input is oversized or not newline terminated")
        message = None
        try:
            message = _strict_object(raw[:-1], "MCP input")
            response = handle_message(message)
        except Exception as exc:
            request_id = None
            try:
                request_id = message.get("id")
            except (NameError, AttributeError):
                pass
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32600,
                    "message": f"invalid request: {type(exc).__name__}: {exc}"[:1000],
                },
            }
        if response is not None:
            sys.stdout.buffer.write(_canonical(response) + b"\n")
            sys.stdout.buffer.flush()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except McpError as exc:
        raise SystemExit(f"REFUSED: {exc}") from exc
