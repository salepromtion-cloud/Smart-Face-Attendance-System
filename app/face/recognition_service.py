import cv2
import numpy as np
import face_recognition
from typing import Any, Dict, Iterable, List, Optional, Tuple


class RecognitionServiceError(Exception):
    """Raised when face encoding or matching cannot be performed safely."""


class RecognitionService:
    """Face encoding and matching service using face_recognition.

    This service is intentionally limited to face encoding and comparison only.
    It does not handle cameras, liveness checks, Excel storage, employee
    registration, or attendance logic.
    """

    def __init__(self, tolerance: float = 0.50) -> None:
        self.tolerance = float(tolerance)

    @staticmethod
    def _ensure_valid_image(face_image: np.ndarray) -> np.ndarray:
        if not isinstance(face_image, np.ndarray):
            raise RecognitionServiceError("face_image must be a NumPy array.")
        if face_image.size == 0:
            raise RecognitionServiceError("face_image is empty.")
        if face_image.ndim != 3:
            raise RecognitionServiceError("face_image must have shape (H, W, C).")
        if face_image.shape[2] != 3:
            raise RecognitionServiceError("face_image must have exactly 3 channels.")
        if not np.isfinite(face_image).all():
            raise RecognitionServiceError("face_image contains invalid numeric pixel data.")
        return face_image

    @staticmethod
    def _to_rgb(face_image: np.ndarray) -> np.ndarray:
        # face_recognition expects RGB order. OpenCV images are BGR by default.
        return cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _validate_encoding(encoding: np.ndarray) -> np.ndarray:
        if not isinstance(encoding, np.ndarray):
            raise RecognitionServiceError("face encoding must be a NumPy array.")
        if encoding.size != 128:
            raise RecognitionServiceError("face encoding must contain exactly 128 values.")
        if not np.isfinite(encoding).all():
            raise RecognitionServiceError("face encoding contains non-finite values.")
        return encoding.astype(np.float64, copy=False)

    def encode_face(self, face_image: np.ndarray) -> np.ndarray:
        """Return a 128-d face encoding for a single detected face.

        The caller provides a cropped face image. The image is expected to be in
        OpenCV BGR order; it is converted to RGB before face_recognition processing.
        """
        image = self._ensure_valid_image(face_image)
        rgb_image = self._to_rgb(image)

        face_locations = face_recognition.face_locations(rgb_image)
        if len(face_locations) == 0:
            raise RecognitionServiceError("No face detected in the supplied image.")
        if len(face_locations) > 1:
            raise RecognitionServiceError(
                "Multiple faces detected. Exactly one face is required for encoding."
            )

        encodings = face_recognition.face_encodings(rgb_image, known_face_locations=face_locations)
        if not encodings:
            raise RecognitionServiceError("Face encoding generation failed for the detected face.")

        return self._validate_encoding(encodings[0])

    def compare_face(
        self,
        face_encoding: np.ndarray,
        known_encoding: np.ndarray,
        tolerance: float = 0.50,
    ) -> Dict[str, Any]:
        """Compare a candidate encoding against a known encoding.

        Returns a dictionary with match state and distance. The caller must still
        decide whether to accept the result based on the configured tolerance.
        """
        candidate = self._validate_encoding(face_encoding)
        known = self._validate_encoding(known_encoding)
        tolerance_value = float(tolerance)

        distances = face_recognition.face_distance([known], candidate)
        distance = float(distances[0]) if len(distances) > 0 else float("inf")

        return {
            "is_match": bool(distance <= tolerance_value),
            "distance": distance,
            "tolerance": tolerance_value,
        }

    def find_best_match(
        self,
        face_encoding: np.ndarray,
        known_encodings: Dict[str, np.ndarray],
        tolerance: float = 0.50,
    ) -> Dict[str, Any]:
        """Match a face against multiple known encodings.

        The keys are expected to be Employee_ID values or similar identifiers.
        This service does not read or write persistence; the caller provides the
        known encoding mapping.
        """
        candidate = self._validate_encoding(face_encoding)
        tolerance_value = float(tolerance)

        if not known_encodings:
            return {
                "employee_id": None,
                "is_match": False,
                "best_distance": None,
                "tolerance": tolerance_value,
            }

        best_employee_id: Optional[str] = None
        best_distance: Optional[float] = None

        for employee_id, known_encoding in known_encodings.items():
            if not isinstance(employee_id, str):
                raise RecognitionServiceError(
                    "known_encodings keys must be strings representing Employee_ID values."
                )
            if not isinstance(known_encoding, np.ndarray):
                raise RecognitionServiceError(
                    f"Known encoding for employee '{employee_id}' is invalid; expected NumPy array."
                )

            validated_known = self._validate_encoding(known_encoding)
            distances = face_recognition.face_distance([validated_known], candidate)
            current_distance = float(distances[0]) if len(distances) > 0 else float("inf")

            if best_distance is None or current_distance < best_distance:
                best_distance = current_distance
                best_employee_id = employee_id

        if best_distance is None:
            return {
                "employee_id": None,
                "is_match": False,
                "best_distance": None,
                "tolerance": tolerance_value,
            }

        return {
            "employee_id": best_employee_id if best_distance <= tolerance_value else None,
            "is_match": bool(best_distance <= tolerance_value),
            "best_distance": float(best_distance),
            "tolerance": tolerance_value,
        }


def _blank_image_test() -> None:
    service = RecognitionService()
    blank = np.zeros((80, 80, 3), dtype=np.uint8)
    try:
        service.encode_face(blank)
        print("blank image test: unexpected success")
    except RecognitionServiceError as exc:
        print(f"blank image test: {exc}")


if __name__ == "__main__":
    _blank_image_test()
