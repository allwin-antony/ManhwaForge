import argparse
import os
import sys
import json
import logging
import gc
from tqdm import tqdm
from src.processing import pdf_to_images
from src.ocr import OCRProcessor
from src.vision import VisualSceneAnalyzer
from src.llm import ScriptGenerator
from src.audio import TTSGenerator
from src.video import VideoEditor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class ManhwaRecapPipeline:
    def __init__(self, output_dir, model="qwen2.5:3b", voice="dramatic_male", video_format="youtube", vision_mode="auto", lore=None):
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.audio_dir = os.path.join(output_dir, "audio_segments")
        self.subs_dir = os.path.join(output_dir, "subtitles")
        self.state_file = os.path.join(output_dir, "state.json")
        
        self.model = model
        self.voice = voice
        self.video_format = video_format
        self.vision_mode = vision_mode
        self.lore = self._resolve_lore(lore)
        
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)
        os.makedirs(self.subs_dir, exist_ok=True)
        
        self.state = self._load_state()

    def _resolve_lore(self, lore_arg):
        if lore_arg and os.path.exists(lore_arg):
            try:
                with open(lore_arg, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return json.dumps(data)
                    return str(data)
            except Exception as e:
                logger.warning(f"Failed to read lore file {lore_arg}: {e}")
        elif lore_arg:
            return lore_arg
            
        local_lore = os.path.join(self.output_dir, "lore.json")
        if os.path.exists(local_lore):
            try:
                with open(local_lore, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except Exception:
                pass
        return None

    def _load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
        return {
            "ocr_results": {}, 
            "visual_data": {},
            "script_segments": [],
            "video_path": None
        }

    def _save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=4)

    def run_stage_0(self, pdf_path, start_page=None, end_page=None, slice_panels=True):
        logger.info(f"Stage 0: Converting {pdf_path} (Pages {start_page or 1} to {end_page or 'end'}) to story panel frames...")
        image_paths = pdf_to_images(pdf_path, self.images_dir, start_page=start_page, end_page=end_page, slice_panels=slice_panels)
        logger.info(f"Stage 0 Complete. {len(image_paths)} panel frames saved.")
        return image_paths

    def run_stage_1(self, limit=None, force=False):
        image_files = sorted([
            os.path.join(self.images_dir, f) 
            for f in os.listdir(self.images_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        if limit:
            image_files = image_files[:limit]
            
        if not image_files:
            logger.error("No images found. Run Stage 0 first.")
            return

        logger.info(f"Stage 1: Extracting Dialogue (OCR) and Visual Situations for {len(image_files)} panel frames...")
        ocr = OCRProcessor()
        vision_analyzer = VisualSceneAnalyzer(mode=self.vision_mode)
        
        if force or "visual_data" not in self.state:
            self.state["visual_data"] = {}

        to_process = [
            img for img in image_files 
            if force or img not in self.state["ocr_results"] or not self.state["ocr_results"][img] or img not in self.state["visual_data"]
        ]
        
        if to_process:
            for img_path in tqdm(to_process, desc="Analyzing Panel Frames"):
                bubbles = ocr.extract_text(img_path)
                self.state["ocr_results"][img_path] = bubbles
                
                vis_info = vision_analyzer.analyze_page(img_path)
                self.state["visual_data"][img_path] = vis_info
                
                self._save_state()
            logger.info(f"Processed {len(to_process)} panel frames.")
        else:
            logger.info("All panel frames already have OCR and visual data.")
            
        ocr.cleanup()
        del ocr
        gc.collect()

    def run_stage_2(self, force=False):
        if not force and self.state.get("script_segments") and any(s.get("script") for s in self.state["script_segments"]):
            logger.info("Script segments already exist in state. Skipping generation.")
            return

        image_files = sorted([
            os.path.join(self.images_dir, f) 
            for f in os.listdir(self.images_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        current_ocr = {img: self.state["ocr_results"].get(img, []) for img in image_files}
        current_vis = {img: self.state.get("visual_data", {}).get(img, {}) for img in image_files}

        logger.info(f"Stage 2: Running Scene-Batched Story Director using {self.model} for {len(image_files)} panel frames...")
        generator = ScriptGenerator(model=self.model, lore=self.lore)
        
        segments = generator.generate_synced_script(
            current_ocr,
            visual_data_dict=current_vis
        )

        self.state["script_segments"] = segments
        self._save_state()
        
        script_path = os.path.join(self.output_dir, "script.txt")
        full_text = "\n\n".join([
            f"[{s.get('frame_label', f'Frame {i+1}')} - {os.path.basename(s.get('image_path', ''))}]\n{s.get('script', '')}"
            for i, s in enumerate(segments)
        ])
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        logger.info(f"Stage 2 Complete. Frame-locked script saved to {script_path}")

    def run_stage_3(self, voice=None, burn_subtitles=True, bgm_path=None, ken_burns=True, force_audio=False):
        segments = self.state.get("script_segments", [])
        if not segments:
            logger.error("No script segments found. Run Stage 2 first.")
            return

        voice = voice or self.voice
        tts = TTSGenerator(voice=voice)
        
        logger.info(f"Stage 3: Generating voiceover audio ({voice}) and subtitles for {len(segments)} segments...")
        for i, seg in enumerate(tqdm(segments, desc="Generating Voiceover")):
            script = seg.get("script", "").strip()
            if not script:
                continue
                
            seg_audio_path = os.path.join(self.audio_dir, f"seg_{i:04d}.mp3")
            seg_srt_path = os.path.join(self.subs_dir, f"seg_{i:04d}.srt")
            
            if force_audio or not os.path.exists(seg_audio_path) or os.path.getsize(seg_audio_path) < 500:
                tts.generate_audio(script, seg_audio_path, seg_srt_path)
                
            seg["audio_path"] = seg_audio_path
            seg["subtitle_path"] = seg_srt_path
            
        self._save_state()

        video_path = os.path.join(self.output_dir, "final_recap_video.mp4")
        logger.info(f"Stage 4: Rendering final {self.video_format.upper()} video with frame-perfect subtitles...")
        
        valid_segments = [
            s for s in segments 
            if s.get("audio_path") and os.path.exists(s["audio_path"]) and os.path.getsize(s["audio_path"]) > 500
        ]
        if not valid_segments:
            logger.error("No valid audio segments available for video rendering.")
            return

        editor = VideoEditor()
        success = editor.create_synced_video(
            valid_segments, 
            video_path,
            video_format=self.video_format,
            ken_burns=ken_burns,
            burn_subtitles=burn_subtitles,
            bgm_path=bgm_path
        )
        
        if success:
            self.state["video_path"] = video_path
            self._save_state()
            logger.info(f"🎉 Pipeline Complete! Video ready at: {video_path}")

def main():
    parser = argparse.ArgumentParser(description="Professional Automated Manhwa Recap Video Generator")
    parser.add_argument("--pdf", type=str, help="Path to input manhwa PDF file")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--start-page", type=int, default=None, help="1-indexed starting page number")
    parser.add_argument("--end-page", type=int, default=None, help="1-indexed ending page number")
    parser.add_argument("--stage", type=int, choices=[0, 1, 2, 3], help="Execute up to a specific stage")
    parser.add_argument("--limit", type=int, help="Limit number of images to process")
    parser.add_argument("--model", type=str, default="qwen2.5:3b", help="Ollama LLM model")
    parser.add_argument("--voice", type=str, default="dramatic_male", choices=["dramatic_male", "energetic_male", "storyteller_uk", "dramatic_female", "deep_male"], help="Voice preset")
    parser.add_argument("--format", type=str, default="youtube", choices=["youtube", "shorts"], help="Video format (youtube=16:9, shorts=9:16)")
    parser.add_argument("--vision", type=str, default="auto", choices=["auto", "local", "layout"], help="Visual situation analysis mode")
    parser.add_argument("--lore", type=str, default=None, help="Path to lore.json or custom premise string (optional, defaults to automatic discovery)")
    parser.add_argument("--bgm", type=str, default=None, help="Path to background music file")
    parser.add_argument("--no-panel-slice", action="store_true", help="Disable automatic comic panel slicing (keep full pages)")
    parser.add_argument("--force", action="store_true", help="Force regenerate script, audio, and video")
    parser.add_argument("--no-subtitles", action="store_true", help="Disable burning subtitles onto video")
    parser.add_argument("--no-ken-burns", action="store_true", help="Disable Ken Burns pan/zoom effect")
    
    args = parser.parse_args()

    pipeline = ManhwaRecapPipeline(
        output_dir=args.output,
        model=args.model,
        voice=args.voice,
        video_format=args.format,
        vision_mode=args.vision,
        lore=args.lore
    )

    if args.pdf:
        pipeline.run_stage_0(
            args.pdf, 
            start_page=args.start_page, 
            end_page=args.end_page,
            slice_panels=not args.no_panel_slice
        )
    
    if args.stage is None or args.stage >= 1:
        pipeline.run_stage_1(limit=args.limit, force=args.force)
        
    if args.stage is None or args.stage >= 2:
        pipeline.run_stage_2(force=args.force)
        
    if args.stage is None or args.stage >= 3:
        pipeline.run_stage_3(
            voice=args.voice,
            burn_subtitles=not args.no_subtitles,
            bgm_path=args.bgm,
            ken_burns=not args.no_ken_burns,
            force_audio=args.force
        )

if __name__ == "__main__":
    main()
