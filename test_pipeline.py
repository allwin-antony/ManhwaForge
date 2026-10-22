import unittest
import os
import shutil
import cv2
import numpy as np
from src.processing import pdf_to_images
from src.ocr import OCRProcessor
from src.vision import VisualSceneAnalyzer
from src.llm import ScriptGenerator
from src.audio import TTSGenerator
from src.video import VideoEditor

class TestManhwaRecapPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = "test_output"
        os.makedirs(os.path.join(cls.test_dir, "images"), exist_ok=True)
        
        # Create a dummy image for testing
        dummy_img = np.zeros((1000, 700, 3), dtype=np.uint8)
        # Add white panel
        dummy_img[50:450, 50:650] = 255
        # Add text-like rectangle inside
        cv2.rectangle(dummy_img, (100, 100), (300, 180), (0, 0, 0), -1)
        
        cls.sample_img_path = os.path.join(cls.test_dir, "images", "page_0006.png")
        cv2.imwrite(cls.sample_img_path, dummy_img)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    def test_01_vision_analyzer(self):
        """Test Visual Scene Analyzer on cropped panel."""
        analyzer = VisualSceneAnalyzer(mode="local")
        bubbles = [{"text": "Jinwoo stands ready.", "is_sfx": False}]
        result = analyzer.analyze_panel(self.sample_img_path, bubbles)
        
        self.assertIn("focal_point", result)
        self.assertEqual(len(result["focal_point"]), 2)
        analyzer.cleanup()

    def test_02_config_loading(self):
        """Test YAML config loading in ScriptGenerator."""
        gen = ScriptGenerator(config="configs/default.yaml")
        self.assertIn("Solo Leveling", gen.system_prompt)
        self.assertEqual(gen.model, "qwen2.5:3b")

    def test_03_story_director_llm(self):
        """Test Story Director script generation with config."""
        gen = ScriptGenerator(config="configs/default.yaml")
        dummy_ocr = {
            self.sample_img_path: [{"text": "The gate opened...", "is_sfx": False}]
        }
        dummy_vis = {
            self.sample_img_path: {
                "focal_point": [0.5, 0.5]
            }
        }
        segments = gen.generate_synced_script(dummy_ocr, visual_data_dict=dummy_vis)
        self.assertEqual(len(segments), 1)
        self.assertIn("script", segments[0])
        self.assertGreater(len(segments[0]["script"]), 15)

    def test_04_tts_generate_sync(self):
        """Test TTSGenerator generate_sync, duration calculation, and ASS subtitle generation."""
        tts = TTSGenerator(voice="calm_female")
        self.assertEqual(tts.voice, "en-US-JennyNeural")
        
        audio_out = os.path.join(self.test_dir, "test_segment.mp3")
        sub_out = os.path.join(self.test_dir, "test_segment.ass")
        
        audio_path, sub_path, duration = tts.generate_sync(
            "Sung Jinwoo steps forward into the dungeon.",
            audio_out,
            sub_out,
            format_type="youtube"
        )
        
        self.assertTrue(os.path.exists(audio_path))
        self.assertGreater(os.path.getsize(audio_path), 100)
        self.assertTrue(os.path.exists(sub_path))
        self.assertGreater(duration, 0.0)
        
        # Verify duration method directly
        dur = tts.get_audio_duration(audio_path)
        self.assertAlmostEqual(dur, duration, delta=0.5)
        
        # Verify ASS content
        with open(sub_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("[Script Info]", content)
        self.assertIn("[Events]", content)

    def test_05_voice_argument_choice(self):
        """Test that main.py CLI accepts calm_female as a valid --voice choice."""
        import argparse
        from main import VOICE_MAP
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--voice", type=str, default="dramatic_male", choices=list(VOICE_MAP.keys()))
        args = parser.parse_args(["--voice", "calm_female"])
        self.assertEqual(args.voice, "calm_female")

    def test_06_requirements_has_imageio_ffmpeg(self):
        """Test that imageio-ffmpeg is listed in requirements.txt."""
        req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
        with open(req_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("imageio-ffmpeg", content)

if __name__ == "__main__":
    unittest.main()

