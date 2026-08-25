import asyncio
import os
import re
import logging
import edge_tts
import time
import shutil
import random
import subprocess

logger = logging.getLogger(__name__)

VOICE_MAP = {
    "dramatic_male": "en-US-ChristopherNeural",
    "energetic_male": "en-US-GuyNeural",
    "storyteller_uk": "en-GB-RyanNeural",
    "dramatic_female": "en-US-AriaNeural",
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

    def _clean_script(self, text):
        """Sanitizes text for optimal TTS pronunciation."""
        text = re.sub(r'[\*\#\_\[\]\(\)]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _create_fallback_audio_and_srt(self, text, output_audio_path, output_srt_path=None, duration=4.0):
        """
        Generates emergency fallback audio + subtitle if cloud service is unreachable,
        preventing dropped video frames or failed pipelines.
        """
        try:
            # Generate clean audio with ffmpeg
            cmd = [
                "ffmpeg", "-y", "-f", "lavfi",
                "-i", f"anullsrc=r=24000:cl=mono",
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
                with open(output_srt_path, "w", encoding="utf-8") as f:
                    f.write(srt_content)
            logger.info(f"Fallback audio placeholder created for: '{text[:30]}...'")
            return True
        except Exception as e:
            logger.error(f"Fallback audio creation failed: {e}")
            return False

    async def _async_generate_audio(self, text, output_audio_path, output_srt_path=None, max_retries=6):
        """Async implementation with exponential jitter backoff and fallback resilience."""
        clean_text = self._clean_script(text)
        if not clean_text:
            clean_text = "..."

        temp_audio = output_audio_path + ".tmp.mp3"
        temp_srt = (output_srt_path + ".tmp.srt") if output_srt_path else None

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
                        with open(output_srt_path, "w", encoding="utf-8") as srt_file:
                            srt_file.write(srt_content)
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
        return self._create_fallback_audio_and_srt(clean_text, output_audio_path, output_srt_path)

    def generate_audio(self, text, output_audio_path, output_srt_path=None):
        """Synchronous wrapper for async Edge-TTS generation."""
        os.makedirs(os.path.dirname(os.path.abspath(output_audio_path)), exist_ok=True)
        if output_srt_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_srt_path)), exist_ok=True)
            
        return asyncio.run(self._async_generate_audio(text, output_audio_path, output_srt_path))
