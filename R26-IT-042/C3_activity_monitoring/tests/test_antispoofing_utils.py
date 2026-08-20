"""
R26-IT-042 — C3: Tests
C3_activity_monitoring/tests/test_antispoofing_utils.py

Unit tests for automatic camera anti-spoofing checks.
Run with: python -m unittest C3_activity_monitoring.tests.test_antispoofing_utils -v
"""

import unittest
from unittest.mock import MagicMock, patch

from common.antispoofing_utils import run_camera_antispoofing_check


class _MockCollection:
    def __init__(self):
        self.docs = []

    def insert_one(self, doc):
        self.docs.append(doc)
        return type("InsertResult", (), {"inserted_id": "mock-id"})()


class _MockDB:
    is_connected = True

    def __init__(self):
        self.collection = _MockCollection()

    def get_collection(self, name):
        if name == "antispoofing_checks":
            return self.collection
        return None


class TestAntispoofingUtils(unittest.TestCase):

    @patch("C2_Anti_Spoofing_Detection.src.antispoofing_detector.AntiSpoofingDetector")
    def test_run_camera_antispoofing_check_persists_result(self, mock_detector_cls):
        """Automatic camera check should store a result for admin display."""
        detector = MagicMock()
        detector.load_model.return_value = True
        detector.predict_from_camera.return_value = (True, 0.92, "Anti-spoofing: Real (avg conf: 0.92)")
        mock_detector_cls.return_value = detector

        db = _MockDB()
        ok = run_camera_antispoofing_check(
            db_client=db,
            user_id="EMP001",
            timeout_sec=0.1,
            windows=3,
            source="break_overrun:short_1",
        )

        self.assertTrue(ok)
        self.assertEqual(len(db.collection.docs), 1)
        doc = db.collection.docs[0]
        self.assertEqual(doc["user_id"], "EMP001")
        self.assertIn(doc["verdict"], {"REAL", "REAL_UNKNOWN", "NO_FACE"})
        self.assertEqual(doc["check_source"], "break_overrun:short_1")
        self.assertIn("check_reason", doc)
        self.assertIn(doc["identity_status"], {"UNKNOWN", "REAL_UNKNOWN", "NO_FACE_DETECTED", "NO_TEMPLATE", "VERIFIER_UNAVAILABLE", "SAME_PERSON", "DIFFERENT_PERSON"})
        detector.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
