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
        r'\bpage_\d+(_p\d+)?\b',
        r'^\s*here is.*?:',
        r'^\s*narration:\s*',
        r'^\s*recap:\s*',
        r'pour encourager le vainqueur'
    ]
    for pattern in banned_phrases:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Fix spacing & punctuation quirks
    text = re.sub(r'\bJinwoos\b', "Jinwoo's", text)
    text = re.sub(r'\bSungs\b', "Sung's", text)
    text = re.sub(r'\bJoohees\b', "Joohee's", text)
    text = re.sub(r'\bJinwoomakes\b', "Jinwoo makes", text)
    text = re.sub(r'\bJinwoonods\b', "Jinwoo nods", text)
    text = re.sub(r'\bJinwoospeaks\b', "Jinwoo speaks", text)
    text = re.sub(r'[\'"]', '', text).strip()
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'^[,\.\s]+|[,\.\s]+$', '', text).strip()
    if text and not text.endswith(('.', '!', '?')):
        text += '.'
    return text

SYSTEM_PROMPT = """You are a top-tier YouTube anime recap narrator (like Manga Recap or AniRecap).
Convert each comic panel into an entertaining, varied third-person story recap sentence (12-18 words).

IMPORTANT CONTEXT:
- Protagonist: Sung Jinwoo (also called Jinwoo) - humanity's weakest E-rank hunter.
- Ally: Joohee - a caring B-rank healer who worries about Jinwoo's safety.
- Leader: Mr. Song - the experienced veteran raid leader.
- Other hunters: Arrogant raid members and gossiping bystanders.

STRICT NEGATIVE CONSTRAINTS:
- NEVER use the words 'chuckle', 'chuckles', 'smirk', 'smirks', 'mock', 'shrug', or foreign phrases.
- Use VARIED, active verbs: 'notices', 'gossips', 'arrives', 'wonders', 'realizes', 'warns', 'explains', 'steps forward', 'greets', 'asks', 'apologizes', 'lectures'.
- Do NOT output explanations or preamble. Output ONLY the narration sentence.

FEW-SHOT EXAMPLES:
Dialogue: "THE WEAKEST HUNTER OF ALL MANKIND... HE WAS HOSPITALIZED AGAIN."
Narration: Whispers immediately spread through the crowd as the veteran hunters mock Jinwoo's infamous reputation as humanity's weakest hunter.

Dialogue: "UH, MAY I HAVE SOME COFFEE? | OH, HUNTER SUNG, SORRY WE JUST RAN OUT..."
Narration: When Jinwoo sheepishly steps over to grab a warm cup of coffee, the vendor apologizes after realizing they just ran out.

Dialogue: "ACK! JINWOO! YOU GOT HURT AGAIN?!"
Narration: Joohee spots the fresh bandages wrapped across Jinwoo's arm and rushes over in shock to lecture him.
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

    def _generate_single_panel(self, path, ocr_dict, vis_dict, last_lead_word=""):
        bubbles = ocr_dict.get(path, [])
        dialogues = [clean_dialogue(b['text']) for b in bubbles if not b.get('is_sfx') and len(b['text'].strip()) > 1]
        dialogue_str = " | ".join(dialogues) if dialogues else "(Visual reaction or action)"
        label = self._extract_frame_label(path)

        avoid_prompt = f" (do NOT start with '{last_lead_word}')" if last_lead_word else ""
        user_prompt = f"""PANEL [{label}] DIALOGUE: "{dialogue_str}"
Write one punchy, engaging YouTube recap sentence{avoid_prompt}:"""

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": 0.55,
                "top_p": 0.9,
                "repeat_penalty": 1.5,
                "presence_penalty": 0.6,
                "num_gpu": 99
            }
        }

        try:
            resp = requests.post(self.chat_url, json=payload, timeout=25)
            resp.raise_for_status()
            raw_text = resp.json()["message"]["content"].strip()
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
        last_lead_word = ""
        total_frames = len(sorted_img_paths)
        
        logger.info(f"Running Pro YouTube Story Director across {total_frames} panel frames...")
        
        for path in tqdm(sorted_img_paths, desc="Directing Story Narration"):
            narration = self._generate_single_panel(
                path,
                page_ocr_dict,
                visual_data_dict,
                last_lead_word=last_lead_word
            )
            
            words = narration.split()
            last_lead_word = words[0] if words else ""
            
            frame_label = self._extract_frame_label(path)
            vis_info = visual_data_dict.get(path, {})
            segments.append({
                "image_path": path,
                "frame_label": frame_label,
                "script": narration,
                "focal_point": vis_info.get("focal_point", [0.5, 0.5])
            })
                
        return segments
