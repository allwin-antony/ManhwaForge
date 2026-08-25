import requests
import json
import os
import re
import yaml
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "default.yaml")

def clean_dialogue(text):
    """Normalizes raw OCR text into readable sentence casing."""
    text = text.strip()
    if not text:
        return ""
    sentences = re.split(r'([.!?]\s*)', text.lower())
    cap = "".join([s.capitalize() for s in sentences])
    for name in ["Jinwoo", "Sung", "Joohee", "Song", "Kim", "Choi", "Seoul", "Korea", "Dokja", "Joonghyuk", "Sangah", "E-rank", "D-rank", "S-rank"]:
        cap = re.sub(rf'\b{name.lower()}\b', name, cap, flags=re.IGNORECASE)
    return cap

def sanitize_narration(text, banned_phrases=None):
    """Strips any stray vision descriptors, markdown, or prompt artifacts from narration."""
    default_banned = [
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
        r'^\s*recap:\s*'
    ]
    banned = default_banned + (banned_phrases or [])
    for pattern in banned:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Fix spacing & apostrophes for character names
    text = re.sub(r'\bJinwoos\b', "Jinwoo's", text)
    text = re.sub(r'\bSungs\b', "Sung's", text)
    text = re.sub(r'\bJoohees\b', "Joohee's", text)
    text = re.sub(r'\bDokjas\b', "Dokja's", text)
    text = re.sub(r'\bJinwoomakes\b', "Jinwoo makes", text)
    text = re.sub(r'\bJinwoonods\b', "Jinwoo nods", text)
    text = re.sub(r'\bJinwoospeaks\b', "Jinwoo speaks", text)
    text = re.sub(r'[\'"]', '', text).strip()
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'^[,\.\s]+|[,\.\s]+$', '', text).strip()
    if text and not text.endswith(('.', '!', '?')):
        text += '.'
    return text

class ScriptGenerator:
    def __init__(self, config=None, model=None, host=None, lore=None, batch_size=1):
        self.config = self._load_config(config)
        
        # Override with explicit parameters if provided
        model_cfg = self.config.get("model", {})
        self.model = model or model_cfg.get("name", "qwen2.5:3b")
        self.host = host or model_cfg.get("host", "http://localhost:11434")
        self.chat_url = f"{self.host}/api/chat"
        self.batch_size = batch_size
        
        self.temperature = model_cfg.get("temperature", 0.55)
        self.top_p = model_cfg.get("top_p", 0.9)
        self.repeat_penalty = model_cfg.get("repeat_penalty", 1.5)
        self.presence_penalty = model_cfg.get("presence_penalty", 0.6)
        self.num_gpu = model_cfg.get("num_gpu", 99)
        
        # Merge external lore if provided
        if lore:
            if isinstance(lore, str):
                try:
                    self.config["lore"] = json.loads(lore)
                except Exception:
                    self.config["lore"] = {"custom_lore": lore}
            elif isinstance(lore, dict):
                self.config["lore"] = lore

        self.banned_phrases = self.config.get("prompt", {}).get("banned_phrases", [])
        self.system_prompt = self._build_system_prompt()

    def _load_config(self, config_input):
        if isinstance(config_input, dict):
            return config_input
        
        path = config_input if (config_input and os.path.exists(config_input)) else DEFAULT_CONFIG_PATH
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    if path.endswith(('.yaml', '.yml')):
                        return yaml.safe_load(f) or {}
                    elif path.endswith('.json'):
                        return json.load(f) or {}
            except Exception as e:
                logger.warning(f"Failed to load config {path}: {e}")
        return {}

    def _build_system_prompt(self):
        p_cfg = self.config.get("prompt", {})
        l_cfg = self.config.get("lore", {})
        
        persona = p_cfg.get("persona", "You are a professional YouTube anime/manhwa recap narrator.\nConvert each comic panel into an entertaining third-person story recap sentence.")
        rules = p_cfg.get("rules", [])
        neg_constraints = p_cfg.get("negative_constraints", [])
        few_shots = p_cfg.get("few_shot_examples", [])
        
        parts = [persona.strip(), ""]
        
        # Format Lore Context
        if l_cfg:
            parts.append("STORY & CHARACTER CONTEXT:")
            if "series_title" in l_cfg:
                parts.append(f"- Series: {l_cfg['series_title']}")
            if "protagonist" in l_cfg:
                parts.append(f"- Protagonist: {l_cfg['protagonist']}")
            if "setting" in l_cfg:
                parts.append(f"- Setting: {l_cfg['setting']}")
            if "cast" in l_cfg and isinstance(l_cfg["cast"], list):
                for char in l_cfg["cast"]:
                    parts.append(f"- {char.get('name', 'Character')}: {char.get('role', '')}")
            parts.append("")

        if rules:
            parts.append("RULES:")
            for r in rules:
                parts.append(f"- {r}")
            parts.append("")

        if neg_constraints:
            parts.append("NEGATIVE CONSTRAINTS:")
            for nc in neg_constraints:
                parts.append(f"- {nc}")
            parts.append("")

        if few_shots:
            parts.append("FEW-SHOT EXAMPLES:")
            for ex in few_shots:
                parts.append(f"Dialogue: \"{ex.get('dialogue', '')}\"")
                parts.append(f"Narration: {ex.get('narration', '')}")
            parts.append("")

        return "\n".join(parts).strip()

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
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "repeat_penalty": self.repeat_penalty,
                "presence_penalty": self.presence_penalty,
                "num_gpu": self.num_gpu
            }
        }

        try:
            resp = requests.post(self.chat_url, json=payload, timeout=25)
            resp.raise_for_status()
            raw_text = resp.json()["message"]["content"].strip()
            raw_text = raw_text.split('\n')[0].strip()
            clean_narr = sanitize_narration(raw_text, self.banned_phrases)
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
        
        logger.info(f"Running Config-Driven YouTube Story Director across {total_frames} panel frames...")
        
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
