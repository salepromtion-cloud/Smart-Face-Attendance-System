import cv2
import numpy as np
import onnxruntime as ort
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class LivenessServiceError(Exception):
    """Raised when liveness inference cannot be initialized or executed safely."""


@dataclass
class LivenessResult:
    is_live: bool
    live_score: Optional[float]
    scores: List[float]
    probabilities: Optional[List[float]]
    details: Dict[str, Any]


class LivenessService:
    """MiniFASNetV2 anti-spoofing inference layer.

    This service accepts an already-cropped face image and performs ONNX Runtime
    inference against the verified MiniFASNetV2 model. It intentionally does NOT
    handle camera access, face detection, identity recognition, or attendance
    workflow logic.

    The exact ONNX artifact does not include class labels. The output is a raw
    3-class score/logit vector. The semantic index corresponding to "live" is not
    encoded in the model and therefore remains configuration-dependent. Until a
    verified live_class_index and threshold are supplied by the caller, this
    service remains fail-closed and returns is_live=False.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        live_class_index: Optional[int] = None,
        threshold: Optional[float] = None,
        providers: Optional[List[str]] = None,
    ) -> None:
        project_root = Path(__file__).resolve().parents[2]
        default_model_path = project_root / "models" / "minifasnet_v2.onnx"

        self.model_path = Path(model_path) if model_path else default_model_path
        self.live_class_index = live_class_index
        self.threshold = float(threshold) if threshold is not None else None
        self.providers = providers or ["CPUExecutionProvider"]

        self.session: Optional[ort.InferenceSession] = None
        self.model_metadata: Dict[str, Any] = {}

        self.input_name: Optional[str] = None
        self.input_shape: Optional[Tuple[Any, ...]] = None
        self.input_type: Optional[str] = None
        self.output_name: Optional[str] = None
        self.output_shape: Optional[Tuple[Any, ...]] = None
        self.output_type: Optional[str] = None

        self.expected_channels: Optional[int] = None
        self.expected_height: Optional[int] = None
        self.expected_width: Optional[int] = None

        self._initialize_session()

    def _initialize_session(self) -> None:
        if not self.model_path.exists():
            raise LivenessServiceError(f"Model file not found: {self.model_path}")

        try:
            self.session = ort.InferenceSession(str(self.model_path), providers=self.providers)
        except Exception as exc:  # pragma: no cover - runtime environment only
            raise LivenessServiceError(f"Unable to create ONNX Runtime session: {exc}") from exc

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()

        if not inputs:
            raise LivenessServiceError("Model has no input tensor(s).")
        if not outputs:
            raise LivenessServiceError("Model has no output tensor(s).")

        input_meta = inputs[0]
        output_meta = outputs[0]

        self.input_name = input_meta.name
        self.input_shape = tuple(input_meta.shape)
        self.input_type = input_meta.type
        self.output_name = output_meta.name
        self.output_shape = tuple(output_meta.shape)
        self.output_type = output_meta.type

        self.expected_channels = self._extract_dimension(self.input_shape, 1, default=None)
        self.expected_height = self._extract_dimension(self.input_shape, 2, default=None)
        self.expected_width = self._extract_dimension(self.input_shape, 3, default=None)

        if self.expected_channels is None:
            # ONNX symbolic batch/sequence dimensions may appear as 'batch' for N,
            # while C/H/W are numeric in the verified model. If a non-numeric value
            # appears in the channel position, fail safely instead of guessing.
            raise LivenessServiceError(
                f"Model metadata did not expose a valid channel dimension: {self.input_shape}."
            )

        if self.expected_height is None or self.expected_width is None:
            raise LivenessServiceError(
                "Model metadata did not expose valid spatial dimensions for the input. "
                f"Actual input shape: {self.input_shape}."
            )

        self._validate_model_interface()

        meta = self.session.get_modelmeta()
        self.model_metadata = {
            "name": getattr(meta, "name", None),
            "producer_name": getattr(meta, "producer_name", None),
            "graph_name": getattr(meta, "graph_name", None),
            "domain": getattr(meta, "domain", None),
            "version": getattr(meta, "version", None),
            "custom_metadata_map": getattr(meta, "custom_metadata_map", None) or {},
        }

    def _validate_model_interface(self) -> None:
        if self.input_name != "input":
            raise LivenessServiceError(
                f"Unexpected ONNX input name: {self.input_name}. Expected 'input'."
            )

        if self.output_name != "output":
            raise LivenessServiceError(
                f"Unexpected ONNX output name: {self.output_name}. Expected 'output'."
            )

        if self.input_shape is None or len(self.input_shape) != 4:
            raise LivenessServiceError(
                f"Unexpected ONNX input shape: {self.input_shape}. Expected 4D [N, C, H, W]."
            )

        if self.output_shape is None or len(self.output_shape) != 2:
            raise LivenessServiceError(
                f"Unexpected ONNX output shape: {self.output_shape}. Expected 2D [N, 3]."
            )

        if self.expected_channels != 3:
            raise LivenessServiceError(
                "MiniFASNetV2 expected 3 input channels. Actual value: "
                f"{self.expected_channels}."
            )

        if self.input_type != "tensor(float)":
            raise LivenessServiceError(
                f"Unexpected ONNX input type: {self.input_type}. Expected 'tensor(float)'."
            )

        if self.output_type != "tensor(float)":
            raise LivenessServiceError(
                f"Unexpected ONNX output type: {self.output_type}. Expected 'tensor(float)'."
            )

        if self.output_shape[-1] != 3:
            raise LivenessServiceError(
                "MiniFASNetV2 output must be 3 raw scores/logits. Actual output shape: "
                f"{self.output_shape}."
            )

    @staticmethod
    def _extract_dimension(shape: Optional[Tuple[Any, ...]], index: int, default: Optional[int] = None) -> Optional[int]:
        if not shape or len(shape) <= index:
            return default
        value = shape[index]
        return int(value) if isinstance(value, (int, np.integer)) else default

    def _preprocess_face_image(self, face_image: np.ndarray) -> np.ndarray:
        self._validate_face_image(face_image)

        target_height = int(self.expected_height)
        target_width = int(self.expected_width)

        # OpenCV uses BGR. The verified model metadata is [N, 3, H, W] and uses
        # RGB order for image input pipelines. This conversion is explicit.
        rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        rgb_image = cv2.resize(rgb_image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

        rgb_image = rgb_image.astype(np.float32, copy=False)
        rgb_image = rgb_image / 255.0

        chw = np.transpose(rgb_image, (2, 0, 1))
        batch = np.expand_dims(chw, axis=0)
        return batch.astype(np.float32, copy=False)

    @staticmethod
    def _validate_face_image(face_image: np.ndarray) -> None:
        if not isinstance(face_image, np.ndarray):
            raise LivenessServiceError("face_image must be a numpy.ndarray.")
        if face_image.size == 0:
            raise LivenessServiceError("face_image is empty.")
        if face_image.ndim != 3:
            raise LivenessServiceError("face_image must have shape (H, W, C).")
        if face_image.shape[2] != 3:
            raise LivenessServiceError("face_image must have exactly 3 channels.")
        if not np.isfinite(face_image).all():
            raise LivenessServiceError("face_image contains invalid non-finite pixel values.")

    @staticmethod
    def _softmax(scores: List[float]) -> List[float]:
        if not scores:
            return []
        values = np.asarray(scores, dtype=np.float64)
        shifted = values - np.max(values)
        exp_values = np.exp(shifted)
        total = np.sum(exp_values)
        if total == 0.0:
            return [0.0 for _ in scores]
        return [float(v) for v in exp_values / total]

    def _run_inference(self, face_image: np.ndarray) -> Dict[str, Any]:
        if self.session is None:
            raise LivenessServiceError("ONNX Runtime session is not initialized.")

        input_tensor = self._preprocess_face_image(face_image)
        input_feed = {self.input_name: input_tensor}
        raw_output = self.session.run([self.output_name], input_feed)[0]

        if raw_output is None:
            raise LivenessServiceError("Model returned no output.")

        output_array = np.asarray(raw_output)
        if output_array.ndim == 0:
            scores = [float(output_array)]
        else:
            scores = [float(value) for value in output_array.reshape(-1).tolist()]

        if len(scores) != 3:
            raise LivenessServiceError(
                f"Unexpected raw score count: {len(scores)}. Expected 3 scores/logits."
            )

        probabilities = self._softmax(scores)

        details: Dict[str, Any] = {
            "model_path": str(self.model_path),
            "input_name": self.input_name,
            "input_shape": list(self.input_shape) if self.input_shape else None,
            "input_type": self.input_type,
            "output_name": self.output_name,
            "output_shape": list(self.output_shape) if self.output_shape else None,
            "output_type": self.output_type,
            "preprocessed_shape": list(input_tensor.shape),
            "channel_conversion": "BGR -> RGB",
            "expected_input_channels": self.expected_channels,
            "expected_input_height": self.expected_height,
            "expected_input_width": self.expected_width,
            "raw_scores": scores,
            "softmax_probabilities": probabilities,
            "configured_live_class_index": self.live_class_index,
            "configured_threshold": self.threshold,
            "class_mapping_known": self.live_class_index is not None and self.threshold is not None,
            "class_order_unknown": True,
            "class_order_evidence": (
                "The exact ONNX artifact exposes three raw output scores with shape [batch, 3], "
                "but it contains no class names, labels, or metadata that define which index represents live. "
                "This class mapping must be supplied separately and verified by the original model source or documentation."
            ),
        }

        return {"scores": scores, "probabilities": probabilities, "details": details}

    def check_liveness(self, face_image: np.ndarray) -> LivenessResult:
        try:
            inference = self._run_inference(face_image)
            scores = inference["scores"]
            probabilities = inference["probabilities"]
            details = inference["details"]

            if self.live_class_index is None:
                details["fail_closed_reason"] = (
                    "live_class_index is not configured. The model exposes three raw scores but "
                    "does not declare which index corresponds to live."
                )
                return LivenessResult(
                    is_live=False,
                    live_score=None,
                    scores=scores,
                    probabilities=probabilities,
                    details=details,
                )

            if self.threshold is None:
                details["fail_closed_reason"] = (
                    "threshold is not configured. No production liveness threshold is available."
                )
                return LivenessResult(
                    is_live=False,
                    live_score=None,
                    scores=scores,
                    probabilities=probabilities,
                    details=details,
                )

            if self.live_class_index < 0 or self.live_class_index >= len(scores):
                details["fail_closed_reason"] = (
                    "live_class_index is invalid for the 3-class output vector."
                )
                return LivenessResult(
                    is_live=False,
                    live_score=None,
                    scores=scores,
                    probabilities=probabilities,
                    details=details,
                )

            live_score = float(scores[self.live_class_index])
            details["decision_rule"] = f"score[{self.live_class_index}] >= threshold"
            details["live_score_selected"] = live_score

            is_live = bool(live_score >= self.threshold)
            return LivenessResult(
                is_live=is_live,
                live_score=live_score,
                scores=scores,
                probabilities=probabilities,
                details=details,
            )
        except Exception as exc:
            details = {
                "fail_closed_reason": f"liveness inference failed: {exc}",
                "fatal_error": True,
                "class_mapping_known": False,
            }
            return LivenessResult(
                is_live=False,
                live_score=None,
                scores=[],
                probabilities=None,
                details=details,
            )

    def close(self) -> None:
        if self.session is not None:
            self.session = None

