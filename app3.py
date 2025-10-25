import cv2
import numpy as np
import time
import os
from insightface.app import FaceAnalysis
from phone_detector import PhoneDetector # Import our utility module

# --- Configuration Constants ---
AUTHORIZED_THRESHOLD = 0.4
FOCUS_ALERT_TIME = 3.0
PHONE_ALERT_SECONDS = 1.0
FOCUS_THRESHOLD = 60.0

# --- PERFORMANCE OPTIMIZATIONS ---
# We will only run these expensive models periodically
FACE_CHECK_INTERVAL = 5.0   # Run face detection every 1.0 second
PHONE_CHECK_INTERVAL = 2.5  # Run phone detection every 2.5 seconds

# Attention weights
W_FACE = 0.40
W_HEAD = 0.50
W_PHONE = 0.10

class ProctoringSession:
    """
    Manages the state and logic for a single proctoring session.
    This class is synchronous (blocking) and designed to be run
    in a separate thread by a non-blocking server (like socket.py).
    """
    
    def __init__(self):
        print(f"[ProctoringSession] New session created. Loading models...")
        self.face_app = FaceAnalysis(name='buffalo_l')
        self.face_app.prepare(ctx_id=-0)
        
        self.phone_detector = PhoneDetector(model_name="yolov8n.pt")
        
        self.ref_embedding = None
        
        # --- State Tracking Variables ---
        self.last_face_check_time = 0.0
        self.last_faces = []
        self.last_authorized = False
        
        # --- New state for phone detection caching ---
        self.last_phone_check_time = 0.0
        self.last_phone_present = False # Cache the last known phone state
        
        self.phone_presence_start_time = None
        self.distracted_start_time = None
        print("[ProctoringSession] All models loaded and session initialized.")

    def register_user_from_frame(self, frame_bytes: bytes) -> tuple[bool, str]:
        if self.ref_embedding is not None:
            return False, "User is already registered."

        try:
            frame = self._decode_frame(frame_bytes)
            if frame is None:
                return False, "Invalid image data."
                
            faces = self.face_app.get(frame)
            
            if not faces:
                return False, "No face found in registration frame."
            if len(faces) > 1:
                return False, "Multiple faces found. Only one person allowed."
                
            self.ref_embedding = faces[0].embedding
            self.last_faces = faces
            self.last_authorized = True
            
            # Start the timers now
            now = time.time()
            self.last_face_check_time = now
            self.last_phone_check_time = now
            
            print("Registration successful. Reference embedding stored.")
            return True, "User registered successfully."
                
        except Exception as e:
            print(f"ERROR during registration: {e}")
            return False, "An internal error occurred."

    def process_frame(self, frame_bytes: bytes) -> list[dict]:
        try:
            frame = self._decode_frame(frame_bytes)
            if frame is None:
                return [{"event": "ERROR", "message": "Invalid frame data."}]

            now = time.time()
            events = []
            
            # --- 1. Face Detection (Runs periodically) ---
            if now - self.last_face_check_time >= FACE_CHECK_INTERVAL:
                self.last_face_check_time = now
                faces = self.face_app.get(frame)
                self.last_faces = faces # Cache the result
                
                if not faces:
                    self.last_authorized = False
                    events.append({"event": "FACE_NOT_FOUND", "timestamp": now})
                
                elif len(faces) > 1:
                    self.last_authorized = False
                    events.append({"event": "MULTIPLE_FACES", "count": len(faces), "timestamp": now})
                
                else: # Exactly one face
                    sim = np.dot(self.ref_embedding, faces[0].embedding) / (
                        np.linalg.norm(self.ref_embedding) * np.linalg.norm(faces[0].embedding)
                    )
                    if sim <= AUTHORIZED_THRESHOLD:
                        self.last_authorized = False
                        events.append({"event": "UNAUTHORIZED_PERSON", "similarity": float(sim), "timestamp": now})
                    else:
                        self.last_authorized = True
            
            # Use the cached state from the last check
            person_count = len(self.last_faces)
            face_present = person_count == 1
            authorized = self.last_authorized

            # --- 2. Phone Detection (Runs periodically) ---
            if now - self.last_phone_check_time >= PHONE_CHECK_INTERVAL:
                self.last_phone_check_time = now
                phone_present, _ = self.phone_detector.detect(frame)
                self.last_phone_present = phone_present # Cache the result
            
            # Use the cached state
            phone_present = self.last_phone_present

            if phone_present:
                if self.phone_presence_start_time is None:
                    self.phone_presence_start_time = now
                elif now - self.phone_presence_start_time >= PHONE_ALERT_SECONDS:
                    events.append({"event": "PHONE_DETECTED", "timestamp": now})
            else:
                self.phone_presence_start_time = None

            # --- 3. Focus & Attention Scoring (Runs every frame, but is fast) ---
            head_score = self._compute_head_score(self.last_faces[0]) if (face_present and authorized) else 0.0
            face_score = 1.0 if (face_present and authorized) else 0.0
            phone_pen = 1.0 if phone_present else 0.0

            if not face_present or not authorized:
                attention_score = 0.0
            else:
                base_score = ((W_FACE*face_score + W_HEAD*head_score + W_PHONE*(1 - phone_pen)) /
                              (W_FACE + W_HEAD + W_PHONE))
                attention_score = base_score * 100
                if head_score > 0.8:
                    attention_score = min(100, attention_score * 1.1)

            # --- 4. Distraction Alert Logic (Runs every frame, but is fast) ---
            if attention_score < FOCUS_THRESHOLD and authorized:
                if self.distracted_start_time is None:
                    self.distracted_start_time = now
                elif now - self.distracted_start_time >= FOCUS_ALERT_TIME:
                    events.append({"event": "LOOKING_AWAY", "score": int(attention_score), "timestamp": now})
            else:
                self.distracted_start_time = None

            return events
            
        except Exception as e:
            print(f"ERROR processing frame: {e}")
            return [{"event": "ERROR", "message": "An internal error occurred."}]
            
    def _decode_frame(self, frame_bytes: bytes) -> np.ndarray | None:
        try:
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return frame
        except Exception:
            return None

    def _compute_head_score(self, face) -> float:
        if face.kps is None or len(face.kps) < 3:
            return 0.0
        
        try:
            left_eye, right_eye, nose = face.kps[0], face.kps[1], face.kps[2]
            
            eye_distance = max((right_eye[0] - left_eye[0]), 1e-6)
            nose_center_offset = (nose[0] - (left_eye[0] + right_eye[0])/2.0) / eye_distance
            
            face_height = max(face.bbox[3] - face.bbox[1], 1e-6)
            eye_y_avg = (left_eye[1] + right_eye[1]) / 2.0
            vert_offset = (nose[1] - eye_y_avg) / face_height
            
            HORIZ_THRESHOLD, VERT_THRESHOLD = 0.35, 0.25
            SEVERE_HORIZ, SEVERE_VERT = 0.5, 0.4
            
            horiz_score = max(0.0, 1.0 - (abs(nose_center_offset) / HORIZ_THRESHOLD))
            vert_score = max(0.0, 1.0 - (abs(vert_offset) / VERT_THRESHOLD))
            
            head_score = (horiz_score + vert_score) / 2.0
            
            if abs(nose_center_offset) > SEVERE_HORIZ or abs(vert_offset) > SEVERE_VERT:
                head_score *= 0.3
            
            if abs(nose_center_offset) < HORIZ_THRESHOLD/2 and abs(vert_offset) < VERT_THRESHOLD/2:
                head_score = min(1.0, head_score * 1.2)
                
            return head_score
        except Exception:
            return 0.0

