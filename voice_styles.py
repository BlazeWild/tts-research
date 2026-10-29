EMOTIONS = {
    "Warm conversation": "warm, friendly, and conversational",
    "Happy": "happy, energetic, and expressive",
    "Neutral": "neutral, calm, and natural",
    "News": "confident and professional, like a news presenter",
    "Narration": "expressive and engaging, like a storyteller",
    "Sad": "sad, gentle, and emotionally restrained",
    "Surprise": "surprised, animated, and expressive",
    "Command": "firm, clear, and authoritative",
}

SPEEDS = {"Slow": "slow", "Normal": "moderate", "Fast": "fast"}
PITCHES = {"Low": "low", "Natural": "natural", "High": "high"}


def make_description(emotion: str, speed: str, pitch: str) -> str:
    return (
        f"Amrita speaks in a {EMOTIONS[emotion]} tone, at a {SPEEDS[speed]} pace "
        f"with a {PITCHES[pitch]} pitch. Her Nepali pronunciation is clear and natural. "
        "The recording is very clear, close-sounding, and high quality, with no background noise."
    )
