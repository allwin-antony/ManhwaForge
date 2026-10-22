import asyncio
import os
import re
import logging
import edge_tts
import time
import shutil
import random
import subprocess
import imageio_ffmpeg

logger = logging.getLogger(__name__)

VOICE_MAP = {
    "dramatic_male": "en-US-ChristopherNeural",
    "energetic_male": "en-US-GuyNeural",
    "storyteller_uk": "en-GB-RyanNeural",
    "dramatic_female": "en-US-AriaNeural",
    "calm_female": "en-US-JennyNeural",
    "deep_male": "en-US-AndrewNeural"
}

class TTSGenerator:
    def __init__(self, voice="dramatic_male", rate="+15%", pitch="+0Hz"):
        """
        Initialize the high-resilience Edge-TTS audio generator.
        Default rate is +15% for snappy, engaging YouTube recap pacing.
        """
        self.voice = VOICE_MAP.get(voice, voice)
        self.rate = rate
        self.pitch = pitch
        self.ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    def _clean_script(self, text):
        """Sanitizes text for optimal TTS pronunciation."""
        text = re.sub(r'[\*\#\_\[\]\(\)]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def _srt_to_ass(srt_text, format_type="youtube"):
        """
        Converts SubRip (SRT) format to Advanced SubStation Alpha (ASS) format.
        """
        width, height = (1080, 1920) if format_type == "shorts" else (1920, 1080)
        font_size = 28 if format_type == "shorts" else 22
        margin_v = 180 if format_type == "shorts" else 50

        ass_lines = [
            "[Script Info]",
            "ScriptType: v4.00+",
            f"PlayResX: {width}",
            f"PlayResY: {height}",
            "ScaledBorderAndShadow: yes",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
            "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            f"Style: Default,Arial,{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,2,1,2,20,20,{margin_v},1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
        ]

        def srt_time_to_ass(t_str):
            clean_t = t_str.strip().replace(",", ".")
            parts = clean_t.split(":")
            if len(parts) == 3:
                h, m, s = parts
                if "." in s:
                    sec, ms = s.split(".")
                    cs = round(int(ms.ljust(3, "0")[:3]) / 10)
                    if cs >= 100: cs = 99
                    return f"{int(h)}:{m}:{sec}.{cs:02d}"
                return f"{int(h)}:{m}:{s}.00"
            return "0:00:00.00"

        dialogue_added = False
        blocks = re.split(r'\n\s*\n', srt_text.strip())
        for block in blocks:
            lines = [l.strip() for l in block.splitlines() if l.strip()]
            time_idx = -1
            for idx, line in enumerate(lines):
                if "-->" in line:
                    time_idx = idx
                    break
            if time_idx != -1:
                start_raw, end_raw = lines[time_idx].split("-->")
                start_ass = srt_time_to_ass(start_raw)
                end_ass = srt_time_to_ass(end_raw)
                text = "\\N".join(lines[time_idx + 1:])
                if text:
                    ass_lines.append(f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{text}")
                    dialogue_added = True

        if not dialogue_added:
            ass_lines.append("Dialogue: 0,0:00:00.00,0:00:04.00,Default,,0,0,0,,")

        return "\n".join(ass_lines) + "\n"

    def _save_subtitles(self, srt_content, output_sub_path, format_type="youtube"):
        """Saves subtitles as .ass or .srt depending on destination file extension."""
        if not output_sub_path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(output_sub_path)), exist_ok=True)
        if output_sub_path.lower().endswith(".ass"):
            final_content = self._srt_to_ass(srt_content, format_type=format_type)
        else:
            final_content = srt_content
        with open(output_sub_path, "w", encoding="utf-8") as f:
            f.write(final_content)

    def get_audio_duration(self, audio_path):
        """
        Calculates exact duration (in seconds) of an audio file using FFmpeg.
        """
        if not audio_path or not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            return 0.0
        try:
            cmd = [self.ffmpeg_exe, "-i", audio_path]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", res.stderr)
            if match:
                h, m, s = match.groups()
                return round(int(h) * 3600 + int(m) * 60 + float(s), 2)
        except Exception as e:
            logger.warning(f"Could not determine audio duration for {audio_path}: {e}")
        return 0.0

    def _create_fallback_audio_and_srt(self, text, output_audio_path, output_srt_path=None, format_type="youtube", duration=4.0):
        """
        Generates emergency fallback audio + subtitle if cloud service is unreachable,
        preventing dropped video frames or failed pipelines.
        """
        try:
            # Generate clean audio with ffmpeg
            cmd = [
                self.ffmpeg_exe, "-y", "-f", "lavfi",
                "-i", "anullsrc=r=24000:cl=mono",
                "-t", str(duration),
                "-q:a", "9", "-acodec", "libmp3lame",
                output_audio_path
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
            if output_srt_path:
                dur_ms = int(duration * 1000)
                sec = dur_ms // 1000
                ms = dur_ms % 1000
                srt_content = (
                    f"1\n"
                    f"00:00:00,000 --> 00:00:{sec:02d},{ms:03d}\n"
                    f"{text}\n"
                )
                self._save_subtitles(srt_content, output_srt_path, format_type=format_type)
            logger.info(f"Fallback audio placeholder created for: '{text[:30]}...'")
            return True
        except Exception as e:
            logger.error(f"Fallback audio creation failed: {e}")
            return False

    async def _async_generate_audio(self, text, output_audio_path, output_srt_path=None, format_type="youtube", max_retries=6):
        """Async implementation with exponential jitter backoff and fallback resilience."""
        clean_text = self._clean_script(text)
        if not clean_text:
            clean_text = "..."

        temp_audio = output_audio_path + ".tmp.mp3"

        for attempt in range(1, max_retries + 1):
            try:
                communicate = edge_tts.Communicate(
                    clean_text, 
                    self.voice, 
                    rate=self.rate, 
                    pitch=self.pitch
                )
                submaker = edge_tts.SubMaker()
                
                with open(temp_audio, "wb") as audio_file:
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            audio_file.write(chunk["data"])
                        elif chunk["type"] == "SentenceBoundary" or chunk["type"] == "WordBoundary":
                            submaker.feed(chunk)
                
                # Check valid audio size (>500 bytes)
                if os.path.exists(temp_audio) and os.path.getsize(temp_audio) > 500:
                    shutil.move(temp_audio, output_audio_path)
                    
                    # Write subtitles if requested
                    if output_srt_path:
                        srt_content = submaker.get_srt()
                        if not srt_content.strip():
                            # Fallback srt format if submaker boundary missing
                            srt_content = f"1\n00:00:00,000 --> 00:00:04,000\n{clean_text}\n"
                        self._save_subtitles(srt_content, output_srt_path, format_type=format_type)
                    return True
                else:
                    raise IOError(f"Generated audio too small ({os.path.getsize(temp_audio) if os.path.exists(temp_audio) else 0} bytes)")
                    
            except Exception as e:
                logger.warning(f"TTS attempt {attempt}/{max_retries} for '{clean_text[:25]}...': {e}")
                if os.path.exists(temp_audio):
                    try: os.remove(temp_audio)
                    except Exception: pass
                    
                if attempt < max_retries:
                    # Exponential backoff with random jitter (e.g. 1.5s, 3s, 5s)
                    sleep_time = (1.5 * attempt) + random.uniform(0.5, 1.5)
                    await asyncio.sleep(sleep_time)
                    
        # If all retries failed due to local network/Wi-Fi dropout, trigger graceful fallback
        logger.warning(f"Cloud TTS unavailable after {max_retries} attempts. Activating emergency audio resilience.")
        return self._create_fallback_audio_and_srt(clean_text, output_audio_path, output_srt_path, format_type=format_type)

    def generate_sync(self, text, output_audio_path, output_sub_path=None, format_type="youtube"):
        """
        Synchronously generates audio and subtitle files, returning paths and duration.
        
        Args:
            text (str): Script text to generate voiceover for.
            output_audio_path (str): Filepath for target MP3 file.
            output_sub_path (str, optional): Filepath for target subtitle file (.ass or .srt).
            format_type (str): 'youtube' or 'shorts'.
            
        Returns:
            tuple: (output_audio_path, output_sub_path, duration)
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_audio_path)), exist_ok=True)
        if output_sub_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_sub_path)), exist_ok=True)

        asyncio.run(self._async_generate_audio(
            text, 
            output_audio_path, 
            output_sub_path, 
            format_type=format_type
        ))

        duration = self.get_audio_duration(output_audio_path)
        return output_audio_path, output_sub_path, duration

    def generate_audio(self, text, output_audio_path, output_srt_path=None, format_type="youtube"):
        """Synchronous wrapper for async Edge-TTS generation."""
        os.makedirs(os.path.dirname(os.path.abspath(output_audio_path)), exist_ok=True)
        if output_srt_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_srt_path)), exist_ok=True)
            
        return asyncio.run(self._async_generate_audio(text, output_audio_path, output_srt_path, format_type=format_type))
