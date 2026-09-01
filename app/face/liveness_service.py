import cv2
import numpy as np
from typing import Tuple, Dict, List, Optional
import mediapipe as mp
from dataclasses import dataclass


@dataclass
class LivenessResult:
    """Result of liveness detection"""
    is_alive: bool
    confidence: float
    details: Dict[str, any]


class LivenessService:
    """
    Service for detecting face liveness to prevent spoofing attacks.
    Uses multiple techniques including eye blinking, head movement, and texture analysis.
    """
    
    def __init__(self, blink_threshold: float = 0.2, movement_threshold: float = 5.0):
        """
        Initialize liveness service with MediaPipe Face Mesh
        
        Args:
            blink_threshold: Eye aspect ratio threshold for blink detection
            movement_threshold: Minimum head movement in pixels to detect
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.blink_threshold = blink_threshold
        self.movement_threshold = movement_threshold
        
        # Eye landmark indices for MediaPipe Face Mesh
        self.LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.MOUTH_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
        self.FACE_CONTOUR_INDICES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        
    def detect_eye_blink(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Detect eye blink using eye aspect ratio
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (blink_detected, eye_aspect_ratio)
        """
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if not results.multi_face_landmarks:
            return False, 0.0
        
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]
        
        # Calculate eye aspect ratio for both eyes
        left_eye_points = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in self.LEFT_EYE_INDICES])
        right_eye_points = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in self.RIGHT_EYE_INDICES])
        
        left_ear = self._calculate_eye_aspect_ratio(left_eye_points)
        right_ear = self._calculate_eye_aspect_ratio(right_eye_points)
        
        avg_ear = (left_ear + right_ear) / 2.0
        blink_detected = avg_ear < self.blink_threshold
        
        return blink_detected, avg_ear
    
    def detect_head_movement(self, frame: np.ndarray, prev_landmarks: Optional[List] = None) -> Tuple[bool, float, List]:
        """
        Detect head movement by tracking face landmarks
        
        Args:
            frame: Input video frame
            prev_landmarks: Previous frame landmarks for comparison
            
        Returns:
            Tuple of (movement_detected, movement_distance, current_landmarks)
        """
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if not results.multi_face_landmarks:
            return False, 0.0, None
        
        current_landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]
        
        if prev_landmarks is None:
            return False, 0.0, current_landmarks
        
        # Calculate movement using nose tip and face center
        curr_nose = np.array([current_landmarks[1].x * w, current_landmarks[1].y * h])
        prev_nose = np.array([prev_landmarks[1].x * w, prev_landmarks[1].y * h])
        
        movement_distance = np.linalg.norm(curr_nose - prev_nose)
        movement_detected = movement_distance > self.movement_threshold
        
        return movement_detected, movement_distance, current_landmarks
    
    def analyze_texture_quality(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Analyze texture quality to detect printed/spoofed images
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (quality_good, sharpness_score)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate Laplacian variance (sharpness metric)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Extract face region
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if not results.multi_face_landmarks:
            return False, laplacian_var
        
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]
        
        # Get face bounding box
        face_points = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in self.FACE_CONTOUR_INDICES])
        x_min, y_min = face_points.min(axis=0).astype(int)
        x_max, y_max = face_points.max(axis=0).astype(int)
        
        # Ensure bounds are within frame
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)
        
        face_region = gray[y_min:y_max, x_min:x_max]
        
        if face_region.size == 0:
            return False, laplacian_var
        
        # Calculate local binary patterns for texture analysis
        face_laplacian_var = cv2.Laplacian(face_region, cv2.CV_64F).var()
        
        # Good quality threshold (higher variance = sharper/more texture)
        quality_good = face_laplacian_var > 100
        
        return quality_good, face_laplacian_var
    
    def detect_mouth_movement(self, frame: np.ndarray, prev_landmarks: Optional[List] = None) -> Tuple[bool, float]:
        """
        Detect mouth movement for liveness verification
        
        Args:
            frame: Input video frame
            prev_landmarks: Previous frame landmarks
            
        Returns:
            Tuple of (movement_detected, mouth_aspect_ratio)
        """
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if not results.multi_face_landmarks:
            return False, 0.0
        
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]
        
        # Get mouth points
        mouth_points = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in self.MOUTH_INDICES])
        
        # Calculate mouth aspect ratio
        mouth_height = np.linalg.norm(mouth_points[0] - mouth_points[5])
        mouth_width = np.linalg.norm(mouth_points[3] - mouth_points[4])
        
        if mouth_width == 0:
            return False, 0.0
        
        mouth_aspect_ratio = mouth_height / mouth_width
        
        if prev_landmarks is None:
            return False, mouth_aspect_ratio
        
        # Compare with previous mouth aspect ratio
        prev_mouth_points = np.array([[prev_landmarks[i].x * w, prev_landmarks[i].y * h] for i in self.MOUTH_INDICES])
        prev_mouth_height = np.linalg.norm(prev_mouth_points[0] - prev_mouth_points[5])
        prev_mouth_width = np.linalg.norm(prev_mouth_points[3] - prev_mouth_points[4])
        
        if prev_mouth_width == 0:
            return False, mouth_aspect_ratio
        
        prev_mouth_aspect_ratio = prev_mouth_height / prev_mouth_width
        movement_detected = abs(mouth_aspect_ratio - prev_mouth_aspect_ratio) > 0.1
        
        return movement_detected, mouth_aspect_ratio
    
    def verify_liveness(self, frames: List[np.ndarray]) -> LivenessResult:
        """
        Verify liveness using multiple techniques on a sequence of frames
        
        Args:
            frames: List of video frames
            
        Returns:
            LivenessResult with is_alive flag and confidence score
        """
        if not frames or len(frames) < 2:
            return LivenessResult(is_alive=False, confidence=0.0, details={"error": "Insufficient frames"})
        
        blink_count = 0
        head_movement_detected = False
        mouth_movement_detected = False
        quality_scores = []
        prev_landmarks = None
        
        details = {
            "blink_count": 0,
            "head_movement": False,
            "mouth_movement": False,
            "avg_texture_quality": 0.0,
            "total_frames_processed": len(frames)
        }
        
        for i, frame in enumerate(frames):
            # Detect blink
            blink, ear = self.detect_eye_blink(frame)
            if blink:
                blink_count += 1
            
            # Detect head movement
            h_move, h_dist, curr_landmarks = self.detect_head_movement(frame, prev_landmarks)
            if h_move:
                head_movement_detected = True
            
            # Analyze texture quality
            quality, texture_score = self.analyze_texture_quality(frame)
            quality_scores.append(texture_score)
            
            # Detect mouth movement
            m_move, mar = self.detect_mouth_movement(frame, prev_landmarks)
            if m_move:
                mouth_movement_detected = True
            
            prev_landmarks = curr_landmarks
        
        # Calculate liveness score
        details["blink_count"] = blink_count
        details["head_movement"] = head_movement_detected
        details["mouth_movement"] = mouth_movement_detected
        details["avg_texture_quality"] = np.mean(quality_scores) if quality_scores else 0.0
        
        # Determine liveness based on criteria
        liveness_indicators = [
            blink_count >= 1,
            head_movement_detected,
            mouth_movement_detected,
            details["avg_texture_quality"] > 100
        ]
        
        # Require at least 2 indicators for positive liveness detection
        positive_indicators = sum(liveness_indicators)
        confidence = positive_indicators / len(liveness_indicators)
        is_alive = positive_indicators >= 2
        
        return LivenessResult(
            is_alive=is_alive,
            confidence=confidence,
            details=details
        )
    
    @staticmethod
    def _calculate_eye_aspect_ratio(eye_points: np.ndarray) -> float:
        """
        Calculate eye aspect ratio (EAR) for blink detection
        
        Args:
            eye_points: Array of eye landmark points
            
        Returns:
            Eye aspect ratio value
        """
        if len(eye_points) < 6:
            return 0.0
        
        # Calculate distances
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        
        # Calculate aspect ratio
        ear = (A + B) / (2.0 * C + 1e-6)
        return ear
    
    def release(self):
        """Release resources"""
        if self.face_mesh:
            self.face_mesh.close()
