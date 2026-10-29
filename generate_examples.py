"""Generate the standard Nepali evaluation samples with one model load."""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().with_name(".env"))

from tts_engine import IndicParlerEngine
from voice_styles import make_description


EXAMPLES = [
    {
        "name": "01_welcome",
        "text": "नमस्कार! नागरिक सेवामा तपाईंलाई स्वागत छ। म तपाईंलाई कसरी सहयोग गर्न सक्छु?",
        "emotion": "Warm conversation",
        "speed": "Normal",
        "pitch": "Natural",
    },
    {
        "name": "02_news",
        "text": "आज देशका अधिकांश स्थानमा मौसम सफा रहने अनुमान गरिएको छ।",
        "emotion": "News",
        "speed": "Normal",
        "pitch": "Natural",
    },
    {
        "name": "03_happy_approval",
        "text": "खुसीको खबर! तपाईंको आवेदन सफलतापूर्वक स्वीकृत भएको छ।",
        "emotion": "Happy",
        "speed": "Normal",
        "pitch": "High",
    },
    {
        "name": "04_instruction",
        "text": "कृपया आफ्नो नागरिकता नम्बर स्पष्ट रूपमा लेखेर पुनः प्रयास गर्नुहोस्।",
        "emotion": "Command",
        "speed": "Slow",
        "pitch": "Natural",
    },
    {
        "name": "05_assistant_answer",
        "text": "तपाईंको प्रश्नका लागि धन्यवाद। म आवश्यक जानकारी खोजेर स्पष्ट जवाफ दिनेछु।",
        "emotion": "Narration",
        "speed": "Normal",
        "pitch": "Natural",
    },
]


def main() -> None:
    engine = IndicParlerEngine()
    print(f"Generating {len(EXAMPLES)} examples; the model is loaded only once.")
    for index, sample in enumerate(EXAMPLES, start=1):
        description = make_description(
            sample["emotion"], sample["speed"], sample["pitch"]
        )
        result = engine.synthesize(sample["text"], description)
        # Replace the timestamped UI-style name with a stable evaluation name.
        stable_path = result.saved_path.parent / f"{sample['name']}.wav"
        result.saved_path.replace(stable_path)
        duration = len(result.audio) / result.sample_rate
        print(
            f"[{index}/{len(EXAMPLES)}] {stable_path.name}: {duration:.1f}s "
            f"audio in {result.elapsed_seconds:.1f}s"
        )
    print(f"Finished. Audio files are in: {result.saved_path.parent}")


if __name__ == "__main__":
    main()
