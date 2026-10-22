import os
import subprocess
import logging
import imageio_ffmpeg
from tqdm import tqdm

logger = logging.getLogger(__name__)

class VideoEditor:
    def __init__(self, fps=24, resolution=(1920, 1080)):
        """
        Initialize the hardware/FFmpeg-accelerated video editor.
        """
        self.fps = fps
        self.resolution = resolution
        self.ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    def _build_filter_graph(self, img_w_target, img_h_target, ken_burns=True, video_format="youtube", srt_path=None, focal_point=None):
        """
        Builds high-performance FFmpeg filtergraph for cinematic manga/manhwa framing.
        """
        w, h = img_w_target, img_h_target
        focal_point = focal_point or [0.5, 0.5]
        
        if video_format == "shorts":
            # 9:16 vertical layout (1080x1920)
            # Background fills screen with blur, foreground fits nicely in center with subtle vertical focus
            fg_filter = f"[0:v]scale={w}:-1[fg];"
            bg_filter = f"[0:v]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},boxblur=25:5,eq=brightness=-0.20[bg];"
            overlay = f"[bg][fg]overlay=(W-w)/2:(H-h)/2"
        else:
            # 16:9 standard YouTube layout (1920x1080)
            # Background scaled & blurred, foreground scaled to fit height with zero destructive cropping
            bg_filter = f"[0:v]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},boxblur=20:5,eq=brightness=-0.15[bg];"
            fg_filter = f"[0:v]scale=-1:{h}[fg];"
            overlay = f"[bg][fg]overlay=(W-w)/2:(H-h)/2"
            
        combined_filter = f"{bg_filter}{fg_filter}{overlay}"
        
        # Burn subtitles if subtitle file is present
        if srt_path and os.path.exists(srt_path):
            escaped_srt = srt_path.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
            if escaped_srt.lower().endswith(".ass"):
                combined_filter += f",subtitles='{escaped_srt}'"
            else:
                margin_v = 180 if video_format == "shorts" else 50
                font_size = 28 if video_format == "shorts" else 22
                sub_style = (
                    f"FontSize={font_size},PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
                    f"BorderStyle=3,Outline=2,Shadow=1,Alignment=2,MarginV={margin_v},FontName=Arial"
                )
                combined_filter += f",subtitles='{escaped_srt}':force_style='{sub_style}'"
            
        combined_filter += "[v]"
        return combined_filter

    def render_segment_clip(self, img_path, audio_path, output_clip_path, video_format="youtube", ken_burns=True, srt_path=None, focal_point=None):
        """
        Renders a single video clip matching the exact audio duration with visual focal awareness.
        """
        w, h = self.resolution
        if video_format == "shorts":
            w, h = 1080, 1920

        filter_complex = self._build_filter_graph(
            w, h, 
            ken_burns=ken_burns, 
            video_format=video_format, 
            srt_path=srt_path,
            focal_point=focal_point
        )

        cmd = [
            self.ffmpeg_exe, "-y",
            "-loop", "1", "-i", img_path,
            "-i", audio_path,
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "1:a",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-r", str(self.fps),
            "-shortest",
            output_clip_path
        ]

        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0:
            logger.error(f"FFmpeg clip render failed for {img_path}:\n{res.stderr.decode('utf-8', errors='ignore')}")
            return False
        return True

    def create_synced_video(self, segments, output_path, video_format="youtube", ken_burns=True, burn_subtitles=True, bgm_path=None, bgm_volume=0.12):
        """
        Renders all synchronized segment clips and losslessly concatenates them into the final video.
        
        Args:
            segments (list): [{"image_path": str, "audio_path": str, "subtitle_path": str, "script": str, "focal_point": list}]
            output_path (str): Final video path.
            video_format (str): 'youtube' (16:9) or 'shorts' (9:16).
            ken_burns (bool): Whether to apply subtle motion/pan.
            burn_subtitles (bool): Whether to burn subtitles onto clips.
            bgm_path (str, optional): Background music file.
            bgm_volume (float): Background music volume (default 0.12).
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        temp_dir = os.path.join(os.path.dirname(output_path), "temp_clips")
        os.makedirs(temp_dir, exist_ok=True)

        rendered_clips = []
        
        logger.info(f"Rendering {len(segments)} synchronized video segments...")
        for i, seg in enumerate(tqdm(segments, desc="Rendering Video Segments")):
            img_path = seg.get("image_path")
            audio_path = seg.get("audio_path")
            srt_path = seg.get("subtitle_path") if burn_subtitles else None
            focal_point = seg.get("focal_point", [0.5, 0.5])
            
            if not img_path or not audio_path or not os.path.exists(audio_path):
                continue
                
            clip_path = os.path.join(temp_dir, f"clip_{i:04d}.mp4")
            success = self.render_segment_clip(
                img_path, audio_path, clip_path,
                video_format=video_format,
                ken_burns=ken_burns,
                srt_path=srt_path,
                focal_point=focal_point
            )
            if success:
                rendered_clips.append(clip_path)

        if not rendered_clips:
            logger.error("No clips were successfully rendered.")
            return False

        # Concat list file
        concat_list_file = os.path.join(temp_dir, "concat_list.txt")
        with open(concat_list_file, "w", encoding="utf-8") as f:
            for c in rendered_clips:
                clean_clip = os.path.abspath(c).replace("\\", "/")
                f.write(f"file '{clean_clip}'\n")

        # Lossless concat
        raw_concat_video = os.path.join(temp_dir, "raw_concat.mp4") if bgm_path else output_path
        
        concat_cmd = [
            self.ffmpeg_exe, "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list_file,
            "-c", "copy",
            raw_concat_video
        ]
        
        logger.info("Concatenating video segments...")
        res = subprocess.run(concat_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0:
            logger.error(f"Concat failed: {res.stderr.decode('utf-8', errors='ignore')}")
            return False

        # Mix BGM if provided
        if bgm_path and os.path.exists(bgm_path):
            logger.info(f"Mixing background music ({os.path.basename(bgm_path)}) at {bgm_volume*100:.0f}% volume...")
            bgm_cmd = [
                self.ffmpeg_exe, "-y",
                "-i", raw_concat_video,
                "-stream_loop", "-1", "-i", bgm_path,
                "-filter_complex", f"[1:a]volume={bgm_volume}[bgm];[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=2[aout]",
                "-map", "0:v",
                "-map", "[aout]",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                output_path
            ]
            res_bgm = subprocess.run(bgm_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res_bgm.returncode != 0:
                logger.error(f"BGM mix failed: {res_bgm.stderr.decode('utf-8', errors='ignore')}")
                return False

        logger.info(f"🎉 Final video successfully created at: {output_path}")
        return True
