import cv2
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

class VisualSceneAnalyzer:
    def __init__(self, mode="auto"):
        self.mode = mode

    def analyze_panel(self, img_path, bubbles=None):
        """Alias for analyze_page."""
        return self.analyze_page(img_path)

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
            atmosphere = "Bright Atmosphere"
            action_type = "Standard"
        else:
            desc_parts.append("Standard comic framing")
            atmosphere = "Balanced"
            action_type = "Standard"

        if edge_density > 0.12:
            desc_parts.append("Dynamic line work")

        # Dynamic Focal Point Detection (Center of Visual Mass/Interest)
        focal_x = 0.5
        focal_y = 0.5
        try:
            # Look for highest contrast/detail region in upper 60% of panel (character faces)
            upper_crop = edges[:int(h * 0.7), :]
            M = cv2.moments(upper_crop)
            if M["m00"] > 0:
                focal_x = float((M["m10"] / M["m00"]) / w)
                focal_y = float((M["m01"] / M["m00"]) / h)
                focal_x = max(0.2, min(0.8, focal_x))
                focal_y = max(0.2, min(0.8, focal_y))
        except Exception:
            pass

        return {
            "atmosphere": atmosphere,
            "action_type": action_type,
            "visual_description": ", ".join(desc_parts),
            "focal_point": [round(focal_x, 2), round(focal_y, 2)]
        }

    def cleanup(self):
        pass
