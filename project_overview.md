# Project: Automated Manhwa Recap Video Generator

## 🎯 Goal
Create a high-volume, fully automated system for generating engaging manhwa recap videos with dramatic narrator voiceover.

## 🔄 Workflow
The pipeline consists of four main stages:

1.  **Stage 0: Pre-Processing**
    *   **Input**: Digital manhwa comic (PDF).
    *   **Action**: Convert PDF to high-resolution images (one per page).
    *   **Tool**: PyMuPDF.

2.  **Stage 1: Text & Layout Extraction (OCR)**
    *   **Action**: Extract text from speech bubbles and determine reading order.
    *   **Tool**: PaddleOCR (or EasyOCR).
    *   **Challenge**: Sorting bounding boxes for correct sequential reading order.

3.  **Stage 2: Script Generation**
    *   **Action**: Rewrite raw text into a dramatic, cinematic narrator script.
    *   **Tool**: Ollama (Llama 3 8B or Mistral 7B).

4.  **Stage 3: Video Production**
    *   **Action**: Generate voiceover, sync with images, apply pans/zooms, and render.
    *   **Tools**:
        *   **TTS**: Coqui TTS or MeloTTS.
        *   **Video**: FFmpeg / MoviePy.

## 💻 Assets & Environment
*   **Hardware**: NVIDIA RTX 4060 (8GB VRAM).
    *   *Crucial for AI inference (LLM & OCR) and video rendering.*
*   **Software**:
    *   **OS**: Linux.
    *   **Language**: Python (Glue code).
    *   **LLM Runner**: Ollama (Installed).
    *   **OCR**: PaddleOCR (Installation in progress).

## 📊 Current Status
*   **PaddleOCR Installation**: Currently installing `paddlepaddle-gpu==3.2.0` to verify compatibility with the system.
*   **Project Phase**: Initialization & Tool Verification.

## ✅ Decisions & Configurations
1.  **OCR Choice**: Primary: PaddleOCR. Fallback: Alternative model (e.g., EasyOCR) if PaddleOCR fails.
2.  **Test Data**: Sample PDF provided in project directory.
3.  **Video Format**: Standard YouTube (16:9).
4.  **Voice Style**: Dramatic narrator (to be refined in Stage 2).
