from paddleocr import PaddleOCR
import numpy as np
import logging
import os
import gc

# Suppress PaddleOCR debug logging
logging.getLogger("ppocr").setLevel(logging.WARNING)
logging.getLogger("paddlex").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Common SFX words that are not narrative dialogue
SFX_LIST = {
    "thud", "huff", "pant", "bam", "clack", "clang", "gasp", "sigh",
    "crack", "boom", "slash", "whoosh", "tap", "shh", "ugh", "ah", "eek", "waak"
}

class OCRProcessor:
    def __init__(self, lang='en', use_gpu=True):
        """
        Initialize the PaddleOCR engine with comic-optimized parameters.
        Disables document orientation classification and unwarping to prevent comic rotation errors.
        """
        self.ocr = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True
        )

    def extract_text(self, image_path):
        """
        Extracts text from a single image and groups into natural dialogue speech bubbles.
        """
        try:
            results = self.ocr.ocr(image_path)
            return self._parse_and_group(image_path, results)
        except Exception as e:
            logger.error(f"Error extracting text from {image_path}: {e}")
            return []

    def extract_text_batch(self, image_paths):
        """
        Extracts text from multiple images in batch.
        """
        all_results = {}
        for img_path in image_paths:
            all_results[img_path] = self.extract_text(img_path)
        return all_results

    def _parse_and_group(self, image_path, results):
        """
        Parses PaddleOCR results (both PaddleOCR 3.x and legacy formats),
        filters low confidence text and SFX, and clusters into speech bubbles.
        """
        if not results:
            return []
            
        r = results[0] if isinstance(results, list) else results
        raw_items = []

        # PaddleOCR 3.x / PaddleX OCRResult format
        if isinstance(r, dict) or hasattr(r, 'get') or hasattr(r, 'keys'):
            texts = r.get('rec_texts', []) if hasattr(r, 'get') else r['rec_texts']
            scores = r.get('rec_scores', []) if hasattr(r, 'get') else r['rec_scores']
            polys = r.get('rec_polys', []) if hasattr(r, 'get') else r['rec_polys']

            for text, score, poly in zip(texts, scores, polys):
                text_str = str(text).strip()
                if score < 0.50 or not text_str:
                    continue
                pts = np.array(poly)
                min_x, min_y = float(np.min(pts[:, 0])), float(np.min(pts[:, 1]))
                max_x, max_y = float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))
                raw_items.append({
                    'text': text_str,
                    'confidence': float(score),
                    'box': [min_x, min_y, max_x, max_y],
                    'center_y': (min_y + max_y) / 2.0,
                    'center_x': (min_x + max_x) / 2.0,
                    'image_path': image_path
                })
        # Legacy PaddleOCR 2.x format: list of [box, (text, score)]
        elif isinstance(r, (list, tuple)):
            for line in r:
                try:
                    if not isinstance(line, (list, tuple)) or len(line) < 2:
                        continue
                    pts = np.array(line[0])
                    text_info = line[1]
                    text_str = str(text_info[0]).strip() if isinstance(text_info, (list, tuple)) else str(text_info).strip()
                    score = float(text_info[1]) if isinstance(text_info, (list, tuple)) and len(text_info) > 1 else 1.0
                    if score < 0.50 or not text_str:
                        continue
                    min_x, min_y = float(np.min(pts[:, 0])), float(np.min(pts[:, 1]))
                    max_x, max_y = float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))
                    raw_items.append({
                        'text': text_str,
                        'confidence': score,
                        'box': [min_x, min_y, max_x, max_y],
                        'center_y': (min_y + max_y) / 2.0,
                        'center_x': (min_x + max_x) / 2.0,
                        'image_path': image_path
                    })
                except Exception as e:
                    logger.warning(f"Error parsing legacy OCR line: {e}")

        if not raw_items:
            return []

        # Sort primarily by top-to-bottom reading position
        raw_items.sort(key=lambda item: item['box'][1])

        # Cluster vertically & horizontally adjacent lines into unified dialogue bubbles
        bubbles = []
        current_bubble = []

        for item in raw_items:
            if not current_bubble:
                current_bubble.append(item)
                continue

            prev = current_bubble[-1]
            v_dist = item['box'][1] - prev['box'][3]  # current top - prev bottom
            h_overlap = max(0, min(item['box'][2], prev['box'][2]) - max(item['box'][0], prev['box'][0]))

            if v_dist < 65 and (h_overlap > 0 or abs(item['center_x'] - prev['center_x']) < 140):
                current_bubble.append(item)
            else:
                combined_text = " ".join([b['text'] for b in current_bubble]).strip()
                is_sfx = combined_text.lower() in SFX_LIST or (len(combined_text) <= 2 and not combined_text.isdigit())
                bubbles.append({
                    'text': combined_text,
                    'is_sfx': is_sfx,
                    'confidence': float(np.mean([b['confidence'] for b in current_bubble])),
                    'box': [
                        min(b['box'][0] for b in current_bubble),
                        min(b['box'][1] for b in current_bubble),
                        max(b['box'][2] for b in current_bubble),
                        max(b['box'][3] for b in current_bubble),
                    ],
                    'image_path': image_path
                })
                current_bubble = [item]

        if current_bubble:
            combined_text = " ".join([b['text'] for b in current_bubble]).strip()
            is_sfx = combined_text.lower() in SFX_LIST or (len(combined_text) <= 2 and not combined_text.isdigit())
            bubbles.append({
                'text': combined_text,
                'is_sfx': is_sfx,
                'confidence': float(np.mean([b['confidence'] for b in current_bubble])),
                'box': [
                    min(b['box'][0] for b in current_bubble),
                    min(b['box'][1] for b in current_bubble),
                    max(b['box'][2] for b in current_bubble),
                    max(b['box'][3] for b in current_bubble),
                ],
                'image_path': image_path
            })

        return bubbles

    def cleanup(self):
        """Free memory and tensors."""
        del self.ocr
        gc.collect()
