# ⚔️ ManhwaForge

> **Note:** This is purely a **hobby project** built for the love of the game and a passion for manhwa / anime recaps — it is **not** a production-ready enterprise product. Built to explore local AI, computer vision, and automated video generation.

ManhwaForge is a 100% free, fully local, open-source pipeline that transforms manhwa & webtoon PDFs/images into complete, cinematic **YouTube / TikTok style recap videos** complete with natural voiceover narration, smooth camera panning (Ken Burns effect), and synchronized burnt-in subtitles.

---

## ✨ Key Features

- ⚙️ **Config-Driven Prompts & Lore (Zero Code Editing):** Easily customize system prompts, character cast, negative constraints, and few-shot examples inside `configs/*.yaml` without touching Python code.
- 🖼️ **Smart Panel Slicing (OpenCV):** Automatically detects panel borders, gutter whitespace, and dark gutters to cut vertical manhwa pages into discrete story frames.
- 💬 **Local Dialogue Extraction (PaddleOCR):** Extracts speech bubble dialogues and sound effects locally on your GPU without paid cloud APIs.
- 🧠 **Story Recap Director (Ollama / Local LLM):** Generates engaging, third-person YouTube-style story recap scripts grounded directly in the panel dialogue with strict anti-repetition constraints.
- 🎙️ **Neural Voiceover (Edge-TTS):** High-speed, high-fidelity neural text-to-speech with multiple natural voice presets and retry backoff.
- 🎬 **Cinematic Video Renderer (FFmpeg):** Creates smooth Ken Burns zooms and focal pans, synchronized frame-by-frame with speech durations and stylish burnt-in subtitles.
- 📱 **Multi-Format Support:** Render for **YouTube (16:9 Landscape)** or **TikTok / YouTube Shorts (9:16 Portrait)**.

---

## 🛠️ Architecture

```
Manhwa PDF / Images
        │
        ▼
[Stage 0: Smart Panel Cutter (OpenCV)]
        │
        ▼
[Stage 1: Speech Bubble OCR (PaddleOCR)]
        │
        ▼
[Stage 2: Script Director (Ollama + configs/*.yaml)]
        │
        ▼
[Stage 3: Neural TTS & Subtitle Generator (Edge-TTS)]
        │
        ▼
[Stage 4: FFmpeg Cinematic Video Compositor]
        │
        ▼
Final Recap Video (.mp4)
```

---

## 🚀 Quickstart

### 1. Prerequisites
- **Python 3.10+**
- **FFmpeg** (`sudo apt install ffmpeg` on Ubuntu/Debian)
- **Ollama** installed with your favorite local model (e.g. `ollama run qwen2.5:3b`)

### 2. Installation
```bash
git clone https://github.com/allwin-antony/ManhwaForge.git
cd ManhwaForge

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Generate a Recap Video

```bash
# Generate a YouTube (16:9) recap video for pages 20 to 25 using default config (Solo Leveling)
python main.py \
  --pdf "My_Manhwa_Volume.pdf" \
  --output output_recap \
  --start-page 20 \
  --end-page 25 \
  --voice energetic_male \
  --format youtube

# Use a custom config for another series (e.g. Omniscient Reader's Viewpoint)
python main.py \
  --pdf "ORV_Volume_01.pdf" \
  --config configs/omniscient_reader.yaml \
  --output output_orv \
  --voice dramatic_male \
  --format youtube

# Generate a TikTok / Shorts (9:16) vertical video
python main.py \
  --pdf "My_Manhwa_Volume.pdf" \
  --output output_shorts \
  --start-page 1 \
  --end-page 5 \
  --voice calm_female \
  --format shorts
```

---

## ⚙️ Customizing Lore & Prompts

You can create or edit any YAML file in `configs/` to tailor the narration style for any webtoon:

```yaml
# configs/my_manhwa.yaml
model:
  name: "qwen2.5:3b"
  temperature: 0.55
  repeat_penalty: 1.5

prompt:
  persona: "You are a suspenseful YouTube webtoon recap narrator."
  rules:
    - "Explain what happens in lively, third-person present tense."
    - "Use active verbs: discovers, warns, intervenes, confronts, realizes."
  negative_constraints:
    - "NEVER use the words 'chuckle', 'chuckles', 'smirk', 'smirks'."

lore:
  series_title: "My Series Title"
  protagonist: "Character Name - Description"
  cast:
    - name: "Ally Name"
      role: "Healer / Companion"
  setting: "Modern fantasy world"
```

---

## 🎭 Voice Presets
- `energetic_male` (`en-US-ChristopherNeural`) - Best for action, hype, and battle scenes.
- `dramatic_male` (`en-US-EricNeural`) - Deep, suspenseful storytelling.
- `calm_female` (`en-US-JennyNeural`) - Smooth, clear recap narration.
- `storyteller_uk` (`en-GB-RyanNeural`) - Classical British storytelling tone.
- `deep_male` (`en-US-GuyNeural`) - Authoritative, grounded narration.

---

## 📜 Disclaimer
This project is for educational and hobby purposes. All manhwa, webtoons, and copyrighted materials belong to their respective creators and publishers.
