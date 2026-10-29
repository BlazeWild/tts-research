import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from voice_styles import make_description


def test_nepali_description_uses_recommended_speaker():
    description = make_description("Happy", "Fast", "High")
    assert "Amrita" in description
    assert "happy" in description
    assert "fast" in description
    assert "high pitch" in description
    assert "no background noise" in description
