import requests
import json
import os
import re
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

def clean_dialogue(text):
    """Normalizes raw OCR text into readable sentence casing."""
    text = text.strip()
    if not text:
        return ""
    sentences = re.split(r'([.!?]\s*)', text.lower())
    cap = "".join([s.capitalize() for s in sentences])
    for name in ["Jinwoo", "Sung", "Joohee", "Song", "Kim", "Choi", "Seoul", "Korea", "E-rank", "D-rank", "S-rank"]:
        cap = re.sub(rf'\b{name.lower()}\b', name, cap, flags=re.IGNORECASE)
    return cap

def sanitize_narration(text):
    """Strips any stray vision descriptors, markdown, or prompt artifacts from narration."""
    banned_phrases = [
        r'\bkinetic motion and dynamic action lines\b',
        r'\bdynamic action lines\b',
        r'\bkinetic motion\b',
        r'\bstandard narrative framing\b',
        r'\bbright, clean lighting\b',
        r'\bcomic panel\b',
        r'\bvisual description\b',
        r'\bpage_\d+(_p\d+)?\b'
    ]
    for pattern in banned_phrases:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    text = re.sub(r'[\'"]', '', text).strip()
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'^[,\.\s]+|[,\.\s]+$', '', text).strip()
    if text and not text.endswith(('.', '!', '?')):
        text += '.'
    return text

SYSTEM_PROMPT = """You are an engaging YouTube anime/manhwa recap narrator.
Convert the given comic panel into an entertaining third-person recap narration sentence (12-20 words).

RULES:
1. Speak as a third-person narrator describing what happens on screen.
2. The main protagonist is Sung Jinwoo (humanity's weakest hunter).
3. Weave dialogue and character interactions naturally into the storytelling.
4. Flow naturally from the previous story sentence.
5. Output ONLY the narration sentence. Do not include quotes, prefixes, or explanations.
"""

class ScriptGenerator:
    def __init__(self, model="qwen2.5:3b", host="http://localhost:11434", lore=None, batch_size=1):
        self.model = model
        self.host = host
        self.chat_url = f"{host}/api/chat"
        self.lore = lore
        self.batch_size = batch_size

    def _extract_frame_label(self, path):
        base = os.path.splitext(os.path.basename(path))[0]
        match = re.search(r"page_(\d+)(?:_p(\d+))?", base)
        if match:
            p_num = int(match.group(1))
            sub = match.group(2)
            return f"Page {p_num} (Panel {int(sub)})" if sub else f"Page {p_num}"
        return base

    def _generate_single_panel(self, path, ocr_dict, vis_dict, previous_context=""):
        bubbles = ocr_dict.get(path, [])
        dialogues = [clean_dialogue(b['text']) for b in bubbles if not b.get('is_sfx') and len(b['text'].strip()) > 1]
        dialogue_str = " | ".join(dialogues) if dialogues else "(Silent visual action/reaction)"
        label = self._extract_frame_label(path)

        prev_str = f"PREVIOUS STORY LINE: {previous_context}\n" if previous_context else ""
        user_prompt = f"""{prev_str}CURRENT PANEL [{label}]:
- Dialogue on screen: "{dialogue_str}"

Write the next third-person recap narration sentence:"""

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": 0.35,
                "top_p": 0.9,
                "repeat_penalty": 1.35,
                "num_gpu": 99
            }
        }

        try:
            resp = requests.post(self.chat_url, json=payload, timeout=25)
            resp.raise_for_status()
            raw_text = resp.json()["message"]["content"].strip()
            # Clean first line
            raw_text = raw_text.split('\n')[0].strip()
            clean_narr = sanitize_narration(raw_text)
            if clean_narr and len(clean_narr) > 10:
                return clean_narr
        except Exception as e:
            logger.warning(f"Generation failed for {label}: {e}")

        # Graceful dialogue fallback
        if dialogues:
            return f"Jinwoo listens quietly as words echo around him: '{dialogues[0]}'."
        return "Tension builds quietly as the hunters prepare for what lies ahead."

    def generate_synced_script(self, page_ocr_dict, visual_data_dict=None, batch_size=1):
        sorted_img_paths = sorted(page_ocr_dict.keys())
        visual_data_dict = visual_data_dict or {}
        segments = []
        last_story_line = ""
        total_frames = len(sorted_img_paths)
        
        logger.info(f"Running Sliding Story Director across {total_frames} panel frames...")
        
        for path in tqdm(sorted_img_paths, desc="Directing Story Narration"):
            narration = self._generate_single_panel(
                path,
                page_ocr_dict,
                visual_data_dict,
                previous_context=last_story_line
            )
            
            frame_label = self._extract_frame_label(path)
            vis_info = visual_data_dict.get(path, {})
            segments.append({
                "image_path": path,
                "frame_label": frame_label,
                "script": narration,
                "focal_point": vis_info.get("focal_point", [0.5, 0.5])
            })
            last_story_line = narration
                
        return segments
