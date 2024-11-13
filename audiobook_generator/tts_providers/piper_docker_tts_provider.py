from concurrent.futures import ThreadPoolExecutor
import os
import asyncio
from multiprocessing import cpu_count
import logging
from typing import Optional, Union, List, Tuple

from pydub import AudioSegment
from wyoming.client import AsyncTcpClient
from wyoming.tts import Synthesize

from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.core.audio_tags import AudioTags
from audiobook_generator.core.utils import set_audio_tags
from audiobook_generator.tts_providers.base_tts_provider import BaseTTSProvider

logger = logging.getLogger(__name__)

__all__ = ["PiperDockerTTSProvider"]


class PiperCommWithPauses:
    def __init__(
        self,
        text: str,
        break_string: str = "    ",
        break_duration: int = 1250,
        output_format: str = "mp3",
        **kwargs,
    ):
        self.full_text = text
        self.host = os.getenv("PIPER_HOST", "piper")
        self.port = int(os.getenv("PIPER_PORT", 10200))
        self.break_string = break_string
        self.break_duration = int(break_duration)
        self.output_format = output_format

        self.parsed = self.parse_text()

    def parse_text(self) -> List[str]:
        logger.debug(
            f"Parsing the text, looking for breaks/pauses using break string: '{self.break_string}'"
        )
        if self.break_string not in self.full_text or not self.break_string:
            logger.debug("No breaks/pauses found in the text")
            return [self.full_text]

        parts = self.full_text.split(self.break_string)
        logger.debug(f"Split into {len(parts)} parts")
        return parts

    def generate_pause(self, duration_ms: int) -> AudioSegment:
        logger.debug(f"Generating pause of {duration_ms} ms")
        # Generate a silent AudioSegment as a pause
        silent = AudioSegment.silent(duration=duration_ms)
        return silent

    def synthesize(self, text: str) -> Tuple[bytes, int, int, int]:
        """Sends a synthesis request to the Piper TTS server and returns the audio data and metadata."""
        logger.debug(f"Synthesizing text: {text[:50]}...")

        async def run():
            return await synthesize_speech(text, host=self.host, port=self.port)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            audio_data, sample_rate, sample_width, channels = loop.run_until_complete(
                run()
            )
        finally:
            loop.close()
            asyncio.set_event_loop(None)
        if audio_data is None:
            audio_data = b""
        return audio_data, sample_rate, sample_width, channels

    def synthesize_and_convert(
        self, idx_text: Tuple[int, str]
    ) -> Tuple[int, AudioSegment]:
        """Synthesizes text and returns a tuple of index and AudioSegment."""
        idx, text = idx_text
        audio_data, rate, width, channels = self.synthesize(text)
        # Ensure sample_width is in bytes per sample
        if width > 4:  # Assume width is in bits
            width = width // 8
        # Convert audio data (bytes) to AudioSegment
        audio_segment = AudioSegment(
            data=audio_data,
            sample_width=width,
            frame_rate=rate,
            channels=channels,
        )
        return idx, audio_segment

    def chunkify(self) -> AudioSegment:
        num_workers = min(cpu_count(), len(self.parsed))
        logger.debug(f"Starting chunkify process with {num_workers} workers")
        audio_segments = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Prepare the list of texts with their indices
            indexed_texts = list(enumerate(self.parsed))
            # Submit tasks and maintain the order via indices
            futures = {
                executor.submit(self.synthesize_and_convert, idx_text): idx_text[0]
                for idx_text in indexed_texts
            }
            # Collect results and store them in a dictionary
            results = {}
            for future in futures:
                idx = futures[future]
                try:
                    idx, audio_segment = future.result()
                    results[idx] = audio_segment
                except Exception as e:
                    logger.error(f"An error occurred during synthesis: {e}")

            # Reconstruct the audio segments in order
            for idx in range(len(self.parsed)):
                audio_segment = results.get(idx)
                if audio_segment:
                    audio_segments.append(audio_segment)
                    if idx < len(self.parsed) - 1 and self.break_duration > 0:
                        # Insert pause
                        pause_segment = self.generate_pause(self.break_duration)
                        audio_segments.append(pause_segment)
                else:
                    logger.error(f"Missing audio segment at index {idx}")

        # Stitch the audio segments together
        combined = sum(audio_segments, AudioSegment.empty())
        logger.debug("Chunkify process completed")
        return combined

    def save(self, audio_fname: Union[str, bytes]) -> None:
        combined = self.chunkify()
        # Export the combined audio to the desired format
        combined.export(audio_fname, format=self.output_format)
        logger.info(f"Audio saved to: {audio_fname}")


async def synthesize_speech(text, host="localhost", port=10200):
    client = AsyncTcpClient(host, port)
    synthesize = Synthesize(text=text)
    request_event = synthesize.event()

    audio_data = bytearray()
    sample_rate = 22050  # Default sample rate
    sample_width = 2  # Default to 16-bit audio
    channels = 1  # Default to mono

    async with client:
        await client.write_event(request_event)

        while True:
            response_event = await client.read_event()
            if response_event is None:
                break

            if response_event.type == "audio-start":
                # Extract audio metadata if available
                sample_rate = response_event.data.get("rate", sample_rate)
                sample_width = response_event.data.get("width", sample_width)
                channels = response_event.data.get("channels", channels)
            elif response_event.type == "audio-chunk" and response_event.payload:
                audio_data.extend(response_event.payload)
            elif response_event.type == "audio-stop":
                return bytes(audio_data), sample_rate, sample_width, channels
            else:
                raise ValueError(f"Unexpected event type: {response_event.type}")
    return None, sample_rate, sample_width, channels


class PiperDockerTTSProvider(BaseTTSProvider):
    def __init__(self, config: GeneralConfig):
        # TTS provider specific config
        config.output_format = config.output_format or "mp3"
        config.break_duration = int(config.break_duration or 1250)  # in milliseconds

        self.price = 0.000  # Piper is free to use
        super().__init__(config)

    def __str__(self) -> str:
        return f"PiperDockerTTSProvider(config={self.config})"

    def validate_config(self):
        # Add any necessary validation for the config here
        pass

    def text_to_speech(
        self,
        text: str,
        output_file: str,
        audio_tags: AudioTags,
    ):
        piper_comm = PiperCommWithPauses(
            text=text,
            break_string=self.get_break_string().strip(),
            break_duration=self.config.break_duration,
            output_format=self.config.output_format,
        )

        piper_comm.save(output_file)

        set_audio_tags(output_file, audio_tags)

    def estimate_cost(self, total_chars):
        return 0  # Piper is free

    def get_break_string(self):
        return "    "  # Four spaces as the default break string

    def get_output_file_extension(self):
        return self.config.output_format
