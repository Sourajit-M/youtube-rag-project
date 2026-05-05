import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import webvtt


def fetch_transcript(video_id: str) -> Optional[str]:
    """
    Downloads and parses the transcript for a YouTube video.
    Returns clean transcript text, or None if unavailable.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        output_template = str(Path(tmpdir) / "%(id)s.%(ext)s")

        cmd = [
            "yt-dlp",
            "--write-auto-sub",
            "--write-sub",
            "--sub-lang", "en",
            "--sub-format", "vtt",
            "--skip-download",
            "--js-runtimes", "node",
            "--quiet",
            "-o", output_template,
            f"https://www.youtube.com/watch?v={video_id}",
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=60)
        except subprocess.CalledProcessError as e:
            print(f"yt-dlp failed for {video_id}: {e.stderr.decode()[:200]}")
            return None
        except subprocess.TimeoutExpired:
            print(f"yt-dlp timed out for {video_id}")
            return None

        vtt_files = list(Path(tmpdir).glob("*.vtt"))
        if not vtt_files:
            print(f"No transcript found for {video_id}")
            return None

        return _parse_vtt(vtt_files[0])


def _parse_vtt(vtt_path: Path) -> Optional[str]:
    """Parse VTT file into clean deduplicated text."""
    try:
        seen: set[str] = set()
        lines: list[str] = []

        for caption in webvtt.read(str(vtt_path)):
            text = caption.text.strip()
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'<[^>]+>', '', text)

            if text and text not in seen:
                seen.add(text)
                lines.append(text)

        return ' '.join(lines) if lines else None

    except Exception as e:
        print(f"VTT parse error: {e}")
        return None