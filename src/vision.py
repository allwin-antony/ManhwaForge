import cv2
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

class VisualSceneAnalyzer:
    def __init__(self, mode="auto"):
        self.mode = mode

    def analyze_page(self, img_path):
        """
        Analyzes image visual composition, color atmosphere, and focal point without hallucinating
        fake story objects.
        """
        if not os.path.exists(img_path):
            return {
                "atmosphere": "Neutral",
                "action_type": "Static",
                "visual_description": "Comic panel",
                "focal_point": [0.5, 0.5]
            }

        img = cv2.imread(img_path)
        if img is None:
            return {
                "atmosphere": "Neutral",
                "action_type": "Static",
                "visual_description": "Comic panel",
                "focal_point": [0.5, 0.5]
            }

        h, w, _ = img.shape
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_val = float(np.mean(gray))

        # Color Tone Analysis
        # Blood / Intense Red
        red_mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
        red_ratio = float(np.sum((red_mask1 > 0) | (red_mask2 > 0)) / (h * w))

        # Motion & Dynamic Lines
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0) / (h * w))

        # Objective visual atmosphere description
        desc_parts = []
        if red_ratio > 0.08:
            desc_parts.append("Intense combat panel with heavy red/blood accents")
            atmosphere = "High Combat"
            action_type = "Action"
        elif mean_val < 60:
            desc_parts.append("Dark, shadowy atmosphere with high contrast")
            atmosphere = "Dark Atmosphere"
            action_type = "Atmospheric"
        elif mean_val > 180:
            desc_parts.append("Bright, clean lighting")
            atmosphere = "Bright Scene"
            action_type = "Dialogue"
        else:
            desc_parts.append("Standard narrative framing")
            atmosphere = "Narrative"
            action_type = "Dialogue"

        if edge_density > 0.06:
            desc_parts.append("Kinetic motion and dynamic action lines")
            action_type = "Dynamic Action"

        # Focal Center Computation
        y_proj = np.sum(edges, axis=1)
        x_proj = np.sum(edges, axis=0)
        focal_y = float(np.argmax(y_proj) / h) if h > 0 and np.max(y_proj) > 0 else 0.5
        focal_x = float(np.argmax(x_proj) / w) if w > 0 and np.max(x_proj) > 0 else 0.5
        focal_x = max(0.2, min(0.8, focal_x))
        focal_y = max(0.2, min(0.8, focal_y))

        return {
            "atmosphere": atmosphere,
            "action_type": action_type,
            "visual_description": "; ".join(desc_parts),
            "focal_point": [round(focal_x, 2), round(focal_y, 2)]
        }
