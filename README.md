# ⚔️ ManhwaForge

> **Note:** This is purely a **hobby project** built for the love of the game and a passion for manhwa / anime recaps — it is **not** a production-ready enterprise product. Built to explore local AI, computer vision, and automated video generation.

ManhwaForge is a 100% free, fully local, open-source pipeline that transforms manhwa & webtoon PDFs/images into complete, cinematic **YouTube / TikTok style recap videos** complete with natural voiceover narration, smooth camera panning (Ken Burns effect), and synchronized burnt-in subtitles.

---

## ✨ Features

- 🖼️ **Smart Panel Slicing (OpenCV):** Automatically detects panel borders, gutter whitespace, and dark gutters to cut long vertical manhwa strips into individual story frame panels.
- 💬 **Local Dialogue Extraction (PaddleOCR):** Extracts speech bubble dialogues and sound effects locally on GPU without expensive Cloud Vision APIs.
- 🧠 **Story Recap Director (Ollama / Local LLMs):** Generates engaging, third-person YouTube-style story recap scripts grounded directly in the panel dialogue, with zero hardcoding and strict anti-repetition constraints.
- 🎙️ **Neural Voiceover (Edge-TTS):** High-speed, high-fidelity neural text-to-speech with multiple natural voice presets.
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
[Stage 2: Script Director (Ollama / Qwen2.5)]
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
# Generate a YouTube (16:9) recap video for pages 20 to 25 of a volume
python main.py \
  --pdf "My_Manhwa_Volume.pdf" \
  --output output_recap \
  --start-page 20 \
  --end-page 25 \
  --voice energetic_male \
  --format youtube

# Generate a TikTok / Shorts (9:16) vertical video
python main.py \
  --pdf "My_Manhwa_Volume.pdf" \
  --output output_shorts \
  --start-page 1 \
  --end-page 5 \
  --voice calm_female \
  --format tiktok
```

---

## 🎭 Voice Presets
- `energetic_male` (`en-US-ChristopherNeural`) - Best for action, hype, and battle scenes.
- `dramatic_male` (`en-US-EricNeural`) - Deep, suspenseful storytelling.
- `calm_female` (`en-US-JennyNeural`) - Smooth, clear recap narration.
- `expressive_female` (`en-US-AvaNeural`) - High emotion and character-driven moments.

---

## 📜 Disclaimer
This project is for educational and hobby purposes. All manhwa, webtoons, and copyrighted materials belong to their respective creators and publishers.
