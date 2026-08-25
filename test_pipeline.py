import os
import sys
import unittest
import json
import logging

sys.path.insert(0, '/home/allwin/Projects/manhwa_recap')

from src.processing import pdf_to_images
from src.ocr import OCRProcessor
from src.vision import VisualSceneAnalyzer
from src.llm import ScriptGenerator
from src.audio import TTSGenerator
from src.video import VideoEditor

logging.basicConfig(level=logging.INFO)

class TestManhwaPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pdf_path = "Solo Leveling (Manhwa) Volume 01 - PDF Room.pdf"
        cls.test_dir = "test_run_artifacts"
        os.makedirs(cls.test_dir, exist_ok=True)

    def test_01_pdf_extraction(self):
        """Test PDF page range extraction."""
        img_paths = pdf_to_images(
            self.pdf_path, 
            os.path.join(self.test_dir, "images"), 
            start_page=6, 
            end_page=7
        )
        self.assertEqual(len(img_paths), 2)
        for p in img_paths:
            self.assertTrue(os.path.exists(p))

    def test_02_visual_layout_analysis(self):
        """Test OpenCV visual layout and atmosphere extraction."""
        analyzer = VisualSceneAnalyzer(mode="layout")
        img_path = os.path.join(self.test_dir, "images", "page_0006.png")
        data = analyzer.analyze_page(img_path)
        self.assertIn("atmosphere", data)
        self.assertIn("action_type", data)
        self.assertIn("focal_point", data)

    def test_03_ocr_extraction(self):
        """Test PaddleOCR 3.x bubble extraction."""
        ocr = OCRProcessor()
        img_path = os.path.join(self.test_dir, "images", "page_0006.png")
        bubbles = ocr.extract_text(img_path)
        self.assertIsInstance(bubbles, list)
        ocr.cleanup()

    def test_04_story_director_llm(self):
        """Test Story Director script generation with visual situations."""
        gen = ScriptGenerator(model="qwen2.5:3b")
        dummy_ocr = {
            "output/images/page_0006.png": [{"text": "The gate opened...", "is_sfx": False}]
        }
        dummy_vis = {
            "output/images/page_0006.png": {
                "visual_description": "Dark shadowy cavern with glowing blue gates.",
                "focal_point": [0.5, 0.5]
            }
        }
        segments = gen.generate_synced_script(dummy_ocr, visual_data_dict=dummy_vis)
        self.assertEqual(len(segments), 1)
        self.assertIn("script", segments[0])
        self.assertGreater(len(segments[0]["script"]), 20)

    def test_05_tts_and_video(self):
        """Test TTS and focal video rendering."""
        tts = TTSGenerator(voice="dramatic_male")
        audio_path = os.path.join(self.test_dir, "audio.mp3")
        srt_path = os.path.join(self.test_dir, "audio.srt")
        tts.generate_audio("Jinwoo gazes into the dark abyss.", audio_path, srt_path)
        
        editor = VideoEditor()
        img_path = os.path.join(self.test_dir, "images", "page_0006.png")
        clip_path = os.path.join(self.test_dir, "test_clip.mp4")
        ok = editor.render_segment_clip(
            img_path, audio_path, clip_path,
            video_format="youtube", srt_path=srt_path, focal_point=[0.5, 0.8]
        )
        self.assertTrue(ok)
        self.assertTrue(os.path.exists(clip_path))

if __name__ == "__main__":
    unittest.main()
