# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for pending review validation; no GitHub writes are made."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from validate_pending_review import main, parse_hunks_by_path, validate_review_draft


class PendingReviewValidationTest(unittest.TestCase):
    """Exercise anchors and invalid review drafts without a GitHub connection."""

    def setUp(self):
        self.hunks = parse_hunks_by_path(
            "diff --git a/example.py b/example.py\n"
            "--- a/example.py\n+++ b/example.py\n"
            "@@ -1,3 +1,3 @@\n context\n--- old text\n+++ new text\n context\n"
            "@@ -10 +10 @@\n-old\n+new\n"
        )

    def make_review_draft(self, **changes):
        comment = {
            "path": "example.py",
            "line": 2,
            "side": "RIGHT",
            "body": "_AI-drafted (Codex), reviewed by me._\n\nCheck this change.",
        }
        comment.update(changes)
        return {"commit_id": "head", "comments": [comment]}

    def test_header_like_code_is_a_commentable_line(self):
        for side in ("LEFT", "RIGHT"):
            with self.subTest(side=side):
                self.assertEqual(validate_review_draft(self.make_review_draft(side=side), "head", self.hunks), [])

    def test_range_within_hunk(self):
        review_draft = self.make_review_draft(start_line=1, start_side="RIGHT", line=3)
        self.assertEqual(validate_review_draft(review_draft, "head", self.hunks), [])

    def test_invalid_anchors(self):
        for changes in (
            {"side": "INVALID"},
            {"side": None},
            {"line": True},
            {"line": "2"},
            {"line": 0},
            {"line": 5},
            {"path": []},
            {"path": "missing.py"},
            {"start_line": 1},
            {"start_line": 3, "start_side": "RIGHT", "line": 2},
            {"start_line": 2, "start_side": "RIGHT", "line": 10},
            {"start_line": 1, "start_side": "LEFT"},
        ):
            with self.subTest(changes=changes):
                self.assertTrue(validate_review_draft(self.make_review_draft(**changes), "head", self.hunks))

    def test_malformed_review_drafts(self):
        for review_draft in (
            None,
            [],
            {"comments": None},
            {"comments": [None]},
            {"commit_id": "head", "body": 42},
            {"commit_id": "head"},
        ):
            with self.subTest(review_draft=review_draft):
                self.assertTrue(validate_review_draft(review_draft, "head", self.hunks))

    def test_event_and_stale_head_rejected(self):
        review_draft = self.make_review_draft()
        review_draft.update(event="COMMENT", commit_id="old")
        self.assertEqual(len(validate_review_draft(review_draft, "head", self.hunks)), 2)

    def test_summary_only_review(self):
        self.assertEqual(
            validate_review_draft(
                {"commit_id": "head", "body": "_AI-drafted (Codex), reviewed by me._\n\nNo findings."}, "head", {}
            ),
            [],
        )

    def test_attribution_required_in_summary_and_comments(self):
        for body in (
            "Missing attribution.",
            "_AI-drafted (Codex)._\n\nMissing reviewer acknowledgement.",
            "_AI-drafted (Codex), reviewed by me._\nMissing blank line.",
            "Finding first.\n\n_AI-drafted (Codex), reviewed by me._\n\n",
        ):
            with self.subTest(body=body):
                self.assertTrue(validate_review_draft(self.make_review_draft(body=body), "head", self.hunks))
                self.assertTrue(validate_review_draft({"commit_id": "head", "body": body}, "head", {}))

    def test_attribution_accepts_different_agent_names(self):
        for agent_name in ("Codex", "Claude Code"):
            with self.subTest(agent_name=agent_name):
                body = f"_AI-drafted ({agent_name}), reviewed by me._\n\nCheck this change."
                review_draft = self.make_review_draft(body=body)
                review_draft["body"] = body
                self.assertEqual(validate_review_draft(review_draft, "head", self.hunks), [])

    def test_head_change_during_validation(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as review_draft_file:
            json.dump(
                {"commit_id": "head", "body": "_AI-drafted (Codex), reviewed by me._\n\nNo findings."},
                review_draft_file,
            )
            review_draft_file.flush()
            with (
                patch("sys.argv", ["validate", "owner/repo", "1", review_draft_file.name]),
                patch("validate_pending_review.run_gh", side_effect=["head", "", "new-head"]),
                redirect_stdout(io.StringIO()) as output,
            ):
                self.assertEqual(main(), 1)
            self.assertTrue("PR head changed" in output.getvalue())

    def test_quoted_and_deleted_paths(self):
        hunks = parse_hunks_by_path(
            'diff --git "a/file\\303\\251.py" "b/file\\303\\251.py"\n'
            '--- "a/file\\303\\251.py"\n+++ /dev/null\n@@ -1 +0,0 @@\n-deleted\n'
        )
        review_draft = self.make_review_draft(path="fileé.py", side="LEFT", line=1)
        self.assertEqual(validate_review_draft(review_draft, "head", hunks), [])


if __name__ == "__main__":
    unittest.main()
