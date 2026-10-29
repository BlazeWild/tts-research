"""Generate a saved Nepali WAV without using the web interface."""
from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().with_name(".env"))

from tts_engine import IndicParlerEngine
from voice_styles import EMOTIONS, PITCHES, SPEEDS, make_description


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="Nepali text to speak")
    parser.add_argument("--emotion", choices=EMOTIONS, default="Warm conversation")
    parser.add_argument("--speed", choices=SPEEDS, default="Normal")
    parser.add_argument("--pitch", choices=PITCHES, default="Natural")
    parser.add_argument("--description", default="", help="Override the voice description")
    args = parser.parse_args()

    description = args.description or make_description(args.emotion, args.speed, args.pitch)
    result = IndicParlerEngine().synthesize(args.text, description)
    duration = len(result.audio) / result.sample_rate
    print(f"Saved {duration:.1f}s audio to {result.saved_path}")
    print(f"Generation took {result.elapsed_seconds:.1f}s on {result.device}")


if __name__ == "__main__":
    main()
