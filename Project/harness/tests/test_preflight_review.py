from __future__ import annotations

import copy
import json
import math
import os
import stat
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

SOURCE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE_ROOT / "Project" / "tools"))
import preflight_review


class QueueTransport:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[tuple[dict, str, float]] = []

    def __call__(self, payload: dict, api_key: str, timeout: float):
        self.calls.append((copy.deepcopy(payload), api_key, timeout))
        if not self.outcomes:
            raise AssertionError("reviewer exceeded the mocked call budget")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class PreflightReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="track2-preflight-", dir="/tmp")
        self.base = Path(self.temp.name)
        self.root = self.base / "repo"
        schema_dir = self.root / "Project" / "audits"
        schema_dir.mkdir(parents=True)
        self.schema_bytes = (
            SOURCE_ROOT / "Project" / "audits" / "preflight_schema.json"
        ).read_bytes()
        (schema_dir / "preflight_schema.json").write_bytes(self.schema_bytes)
        self.cache = self.base / "private-review-cache"

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def request_id(number: int) -> str:
        return f"{number:064x}"

    def packet(self, number: int, *, kind: str = "attempt", value: str = "x") -> dict:
        return {
            "request_id": self.request_id(number),
            "kind": kind,
            "candidate": {"value": value},
        }

    @staticmethod
    def finding(number: int, severity: str = "major") -> dict:
        return {
            "code": f"RULE_{number}",
            "severity": severity,
            "evidence": f"packet.candidate.value at item {number}",
            "issue": f"Concrete policy violation number {number} is present in the packet.",
        }

    def response(
        self,
        verdict: str,
        number: int,
        *,
        findings: list[dict] | None = None,
        summary: str | None = None,
    ) -> tuple[dict, dict]:
        if findings is None:
            findings = [self.finding(number)] if verdict == "REJECT" else []
        review = {
            "verdict": verdict,
            "findings": findings,
            "summary": summary or f"review summary {number}",
        }
        return (
            {
                "id": f"resp_{number}",
                "status": "completed",
                "error": None,
                "model": preflight_review.MODEL,
                "output": [
                    {"type": "reasoning", "summary": []},
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps(
                                    review,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ),
                            }
                        ],
                    },
                ],
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "total_tokens": 120,
                },
            },
            {"x-request-id": f"req_{number}"},
        )

    def run_review(
        self,
        number: int,
        outcomes,
        *,
        packet: dict | None = None,
        kind: str = "attempt",
        cache_dir: Path | None = None,
        timeout: float = 30.0,
    ) -> tuple[dict, QueueTransport]:
        packet = packet if packet is not None else self.packet(number, kind=kind)
        transport = QueueTransport(outcomes)
        result = preflight_review.run_review(
            root=self.root,
            kind=kind,
            request_id=self.request_id(number),
            packet=packet,
            timeout_seconds=timeout,
            cache_dir=cache_dir or self.cache,
            transport=transport,
        )
        return result, transport

    def cache_path(self, number: int) -> Path:
        return self.cache / f"{self.request_id(number)}.json"

    @staticmethod
    def rewrite_cache(path: Path, value: dict) -> None:
        path.write_bytes(preflight_review.canonical_bytes(value) + b"\n")
        os.chmod(path, 0o600)

    @staticmethod
    def resign_cache(value: dict) -> None:
        value["result_sha256"] = preflight_review.sha256_bytes(
            preflight_review.canonical_bytes(value["result"])
        )
        unsigned = {key: item for key, item in value.items() if key != "entry_sha256"}
        value["entry_sha256"] = preflight_review.sha256_bytes(
            preflight_review.canonical_bytes(unsigned)
        )

    def test_exact_no_tools_responses_payload_and_two_call_approval(self):
        packet = self.packet(1)
        result, transport = self.run_review(
            1,
            [self.response("APPROVE", 1), self.response("APPROVE", 2)],
            packet=packet,
        )

        expected = {
            "model": "gpt-5.6-sol",
            "instructions": (
                preflight_review.PROMPTS["attempt"]
                + preflight_review.COMMON_INSTRUCTIONS
            ),
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": preflight_review.canonical_bytes(packet).decode("utf-8"),
                        }
                    ],
                }
            ],
            "reasoning": {"effort": "high"},
            "tools": [],
            "tool_choice": "none",
            "parallel_tool_calls": False,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "track2_preflight",
                    "strict": True,
                    "schema": json.loads(self.schema_bytes),
                }
            },
            "store": False,
            "max_output_tokens": 5000,
        }
        self.assertEqual(len(transport.calls), 2)
        for payload, _, timeout in transport.calls:
            self.assertEqual(payload, expected)
            self.assertTrue(math.isfinite(timeout))
            self.assertGreater(timeout, 0)
            self.assertLessEqual(timeout, 30.0)
            self.assertNotIn("OPENAI_API_KEY", preflight_review.canonical_bytes(payload).decode())
        self.assertEqual(result["verdict"], "APPROVE")
        self.assertTrue(result["accepted"])
        self.assertEqual(len(result["calls"]), 2)

    def test_two_unanimous_rejections_form_reject_consensus(self):
        result, transport = self.run_review(
            2,
            [self.response("REJECT", 1), self.response("REJECT", 2)],
        )
        self.assertEqual(len(transport.calls), 2)
        self.assertEqual(result["verdict"], "REJECT")
        self.assertFalse(result["accepted"])
        self.assertEqual(
            [finding["code"] for finding in result["findings"]],
            ["RULE_1", "RULE_2"],
        )

    def test_accept_and_approve_with_notes_are_one_consensus_class(self):
        note = self.finding(7, severity="note")
        result, transport = self.run_review(
            3,
            [
                self.response("APPROVE", 1),
                self.response("APPROVE_WITH_NOTES", 2, findings=[note]),
            ],
        )
        self.assertEqual(len(transport.calls), 2)
        self.assertEqual(result["verdict"], "APPROVE_WITH_NOTES")
        self.assertEqual(result["findings"], [note])

    def test_disagreement_uses_exactly_one_tiebreaker_and_majority(self):
        cases = [
            (
                4,
                ["REJECT", "APPROVE", "APPROVE", "REJECT"],
                "APPROVE",
            ),
            (
                5,
                ["APPROVE", "REJECT", "REJECT", "APPROVE"],
                "REJECT",
            ),
        ]
        for number, verdicts, expected in cases:
            with self.subTest(expected=expected):
                outcomes = [
                    self.response(verdict, 20 * number + index)
                    for index, verdict in enumerate(verdicts)
                ]
                result, transport = self.run_review(number, outcomes)
                self.assertEqual(len(transport.calls), preflight_review.MAX_API_CALLS)
                self.assertEqual(len(result["calls"]), 3)
                self.assertEqual(result["verdict"], expected)
                self.assertEqual(len(transport.outcomes), 1)

    def test_malformed_or_tool_bearing_response_fails_closed_without_retry(self):
        malformed_cases = []
        bad_json, headers = self.response("APPROVE", 1)
        bad_json["output"][1]["content"][0]["text"] = "{not-json"
        malformed_cases.append((bad_json, headers))

        tool_output, headers = self.response("APPROVE", 2)
        tool_output["output"].append(
            {"type": "function_call", "name": "forbidden", "arguments": "{}"}
        )
        malformed_cases.append((tool_output, headers))

        invalid_reject = self.response("REJECT", 3, findings=[])
        malformed_cases.append(invalid_reject)

        wrong_model, headers = self.response("APPROVE", 4)
        wrong_model["model"] = "gpt-5.6-solar"
        malformed_cases.append((wrong_model, headers))

        unicode_code = self.response(
            "REJECT",
            5,
            findings=[
                {
                    **self.finding(5),
                    "code": "RÜLE_5",
                }
            ],
        )
        malformed_cases.append(unicode_code)

        for offset, malformed in enumerate(malformed_cases, 10):
            with self.subTest(case=offset):
                transport = QueueTransport([malformed, self.response("APPROVE", 99)])
                with self.assertRaises(preflight_review.ReviewError):
                    preflight_review.run_review(
                        root=self.root,
                        kind="attempt",
                        request_id=self.request_id(offset),
                        packet=self.packet(offset),
                        timeout_seconds=30,
                        cache_dir=self.cache,
                        transport=transport,
                    )
                self.assertEqual(len(transport.calls), 1)
                self.assertFalse(self.cache_path(offset).exists())

    def test_transport_retries_are_bounded_per_reviewer_call(self):
        always_fails = QueueTransport(
            [urllib.error.URLError("offline"), urllib.error.URLError("offline")]
        )
        with mock.patch.object(preflight_review.time, "sleep", return_value=None):
            with self.assertRaisesRegex(preflight_review.ReviewError, "transport failed"):
                preflight_review.run_review(
                    root=self.root,
                    kind="attempt",
                    request_id=self.request_id(20),
                    packet=self.packet(20),
                    timeout_seconds=30,
                    cache_dir=self.cache,
                    transport=always_fails,
                )
        self.assertEqual(len(always_fails.calls), preflight_review.MAX_TRANSPORT_ATTEMPTS)
        self.assertTrue(all(math.isfinite(call[2]) and call[2] > 0 for call in always_fails.calls))

        retry_then_consensus = QueueTransport(
            [
                urllib.error.URLError("transient"),
                self.response("APPROVE", 21),
                self.response("APPROVE", 22),
            ]
        )
        with mock.patch.object(preflight_review.time, "sleep", return_value=None):
            result = preflight_review.run_review(
                root=self.root,
                kind="attempt",
                request_id=self.request_id(21),
                packet=self.packet(21),
                timeout_seconds=30,
                cache_dir=self.cache,
                transport=retry_then_consensus,
            )
        self.assertEqual(len(retry_then_consensus.calls), 3)
        self.assertEqual(result["calls"][0]["transport_attempts"], 2)
        self.assertEqual(result["calls"][1]["transport_attempts"], 1)

    def test_request_id_packet_and_review_kind_are_strictly_bound(self):
        no_calls = QueueTransport([])
        cases = [
            (
                self.request_id(30),
                {"request_id": self.request_id(31), "kind": "attempt"},
                "attempt",
            ),
            (
                self.request_id(32),
                {"request_id": self.request_id(32), "kind": "final"},
                "attempt",
            ),
            (
                "a" * 63 + "/",
                {"request_id": "a" * 63 + "/", "kind": "attempt"},
                "attempt",
            ),
        ]
        unbound_packet = {"kind": "attempt", "candidate": "unbound"}
        cases.append(
            (
                preflight_review.sha256_bytes(
                    preflight_review.canonical_bytes(unbound_packet)
                ),
                unbound_packet,
                "attempt",
            )
        )
        for request_id, packet, kind in cases:
            with self.subTest(request_id=request_id):
                with self.assertRaises(preflight_review.ReviewError):
                    preflight_review.run_review(
                        root=self.root,
                        kind=kind,
                        request_id=request_id,
                        packet=packet,
                        timeout_seconds=30,
                        cache_dir=self.cache,
                        transport=no_calls,
                    )
        self.assertEqual(no_calls.calls, [])

    def test_timeout_must_be_strictly_positive_and_finite(self):
        for timeout in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(timeout=timeout):
                transport = QueueTransport([])
                with self.assertRaisesRegex(preflight_review.ReviewError, "finite positive"):
                    preflight_review.run_review(
                        root=self.root,
                        kind="attempt",
                        request_id=self.request_id(40),
                        packet=self.packet(40),
                        timeout_seconds=timeout,
                        cache_dir=self.cache,
                        transport=transport,
                    )
                self.assertEqual(transport.calls, [])

    def test_private_external_cache_reuses_the_exact_result_without_transport(self):
        result, transport = self.run_review(
            50,
            [self.response("APPROVE", 1), self.response("APPROVE", 2)],
        )
        self.assertEqual(len(transport.calls), 2)
        cache_path = self.cache_path(50)
        self.assertTrue(cache_path.is_file())
        self.assertFalse(self.cache.resolve().is_relative_to(self.root.resolve()))
        self.assertEqual(stat.S_IMODE(self.cache.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(cache_path.stat().st_mode), 0o600)
        self.assertEqual(cache_path.stat().st_uid, os.geteuid())

        def forbidden_transport(*_args):
            raise AssertionError("exact sticky cache reuse must not call transport")

        reused = preflight_review.run_review(
            root=self.root,
            kind="attempt",
            request_id=self.request_id(50),
            packet=self.packet(50),
            timeout_seconds=30,
            cache_dir=self.cache,
            transport=forbidden_transport,
        )
        self.assertEqual(reused, result)

    def test_cache_rejects_changed_packet_and_root_bindings(self):
        self.run_review(51, [self.response("APPROVE", 1), self.response("APPROVE", 2)])
        changed = self.packet(51, value="changed")
        with self.assertRaisesRegex(preflight_review.ReviewError, "identity mismatch"):
            self.run_review(51, [], packet=changed)

        other_root = self.base / "other-repo"
        other_schema = other_root / "Project" / "audits"
        other_schema.mkdir(parents=True)
        (other_schema / "preflight_schema.json").write_bytes(self.schema_bytes)
        with self.assertRaisesRegex(preflight_review.ReviewError, "identity mismatch"):
            preflight_review.run_review(
                root=other_root,
                kind="attempt",
                request_id=self.request_id(51),
                packet=self.packet(51),
                timeout_seconds=30,
                cache_dir=self.cache,
                transport=QueueTransport([]),
            )

    def test_cache_rejects_tampering_even_when_json_remains_canonical(self):
        self.run_review(52, [self.response("APPROVE", 1), self.response("APPROVE", 2)])
        path = self.cache_path(52)
        value = json.loads(path.read_bytes())
        value["result"]["summary"] = "tampered summary"
        self.rewrite_cache(path, value)
        with self.assertRaisesRegex(preflight_review.ReviewError, "integrity digest"):
            self.run_review(52, [])

    def test_cache_rejects_resigned_noncanonical_evidence_path(self):
        self.run_review(53, [self.response("APPROVE", 1), self.response("APPROVE", 2)])
        path = self.cache_path(53)
        value = json.loads(path.read_bytes())
        value["result"]["packet_path"] = "../outside.json"
        self.resign_cache(value)
        self.rewrite_cache(path, value)
        with self.assertRaisesRegex(preflight_review.ReviewError, "evidence path"):
            self.run_review(53, [])

    def test_cache_rejects_unsafe_file_modes_links_and_owner(self):
        self.run_review(54, [self.response("APPROVE", 1), self.response("APPROVE", 2)])
        path = self.cache_path(54)
        os.chmod(path, 0o644)
        with self.assertRaisesRegex(preflight_review.ReviewError, "private bounded"):
            self.run_review(54, [])

        os.chmod(path, 0o600)
        hardlink = self.base / "cache-hardlink.json"
        os.link(path, hardlink)
        with self.assertRaisesRegex(preflight_review.ReviewError, "private bounded"):
            self.run_review(54, [])
        hardlink.unlink()

        with mock.patch.object(preflight_review.os, "geteuid", return_value=os.geteuid() + 1):
            with self.assertRaisesRegex(preflight_review.ReviewError, "owner-controlled"):
                self.run_review(54, [])

        path.unlink()
        os.mkfifo(path, mode=0o600)
        with self.assertRaisesRegex(preflight_review.ReviewError, "private bounded"):
            self.run_review(54, [])
        path.unlink()

        target = self.base / "symlink-target.json"
        target.write_text("{}\n")
        os.chmod(target, 0o600)
        path.symlink_to(target)
        with self.assertRaisesRegex(preflight_review.ReviewError, "unsafe"):
            self.run_review(54, [])

    def test_cache_rejects_directory_mode_and_symlink_paths(self):
        self.cache.mkdir(mode=0o700)
        os.chmod(self.cache, 0o755)
        with self.assertRaisesRegex(preflight_review.ReviewError, "mode 0700"):
            self.run_review(55, [])

        real_cache = self.base / "real-cache"
        real_cache.mkdir(mode=0o700)
        symlink_cache = self.base / "symlink-cache"
        symlink_cache.symlink_to(real_cache, target_is_directory=True)
        with self.assertRaisesRegex(preflight_review.ReviewError, "symlink"):
            self.run_review(56, [], cache_dir=symlink_cache)

    def test_cache_must_be_absolute_and_outside_repository(self):
        for cache_dir in (Path("relative-cache"), self.root / "private-cache"):
            with self.subTest(cache_dir=cache_dir):
                with self.assertRaises(preflight_review.ReviewError):
                    self.run_review(57, [], cache_dir=cache_dir)


if __name__ == "__main__":
    unittest.main()
