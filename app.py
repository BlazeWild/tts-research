from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

# Resolve relative to this file, so launching from the research root or another
# virtual environment still loads tts-research/.env.
load_dotenv(Path(__file__).resolve().with_name(".env"))

from tts_engine import IndicParlerEngine
from voice_styles import EMOTIONS, PITCHES, SPEEDS, make_description

engine = IndicParlerEngine()


def generate(text: str, emotion: str, speed: str, pitch: str, custom: str):
    description = custom.strip() or make_description(emotion, speed, pitch)
    try:
        result = engine.synthesize(text, description)
    except Exception as exc:
        message = str(exc)
        if "gated" in message.lower() or "401" in message or "403" in message:
            message = (
                "Model access denied. Accept the model terms on Hugging Face, then set HF_TOKEN "
                "before starting the app. Original error: " + message
            )
        elif "out of memory" in message.lower():
            message = (
                "GPU memory ran out. Close other GPU apps and retry, or launch with "
                "TTS_DEVICE=cpu (slower). Original error: " + message
            )
        raise gr.Error(message)
    duration = len(result.audio) / result.sample_rate
    status = (
        f"Generated {duration:.1f}s audio in {result.elapsed_seconds:.1f}s · "
        f"{result.device} · {result.sample_rate:,} Hz"
    )
    return (result.sample_rate, result.audio), status, description


with gr.Blocks(title="Nepali Expressive TTS") as demo:
    gr.Markdown(
        "# नेपाली Expressive TTS\n"
        "Type a Nepali response, choose how it should sound, and generate speech with "
        "**AI4Bharat Indic Parler-TTS**. The first generation downloads and loads the model."
    )
    with gr.Row():
        with gr.Column(scale=3):
            text = gr.Textbox(
                label="नेपाली पाठ",
                value="नमस्कार! तपाईंलाई आज के सहयोग चाहिएको छ?",
                lines=6,
                placeholder="यहाँ नेपालीमा लेख्नुहोस्…",
            )
            with gr.Row():
                emotion = gr.Dropdown(list(EMOTIONS), value="Warm conversation", label="Emotion")
                speed = gr.Radio(list(SPEEDS), value="Normal", label="Speed")
                pitch = gr.Radio(list(PITCHES), value="Natural", label="Pitch")
            custom = gr.Textbox(
                label="Advanced voice description (optional, English works best)",
                lines=3,
                placeholder="Leave blank to use Amrita and the controls above.",
            )
            submit = gr.Button("Generate speech", variant="primary")
        with gr.Column(scale=2):
            audio = gr.Audio(label="Generated audio", type="numpy", autoplay=True)
            status = gr.Textbox(label="Run information", interactive=False)
            used_description = gr.Textbox(label="Description sent to the model", interactive=False, lines=5)

    gr.Examples(
        examples=[
            ["नमस्कार! नागरिक सेवामा तपाईंलाई स्वागत छ। म तपाईंलाई कसरी सहयोग गर्न सक्छु?", "Warm conversation", "Normal", "Natural", ""],
            ["आज देशका अधिकांश स्थानमा मौसम सफा रहने अनुमान गरिएको छ।", "News", "Normal", "Natural", ""],
            ["वाह! तपाईंको आवेदन सफलतापूर्वक स्वीकृत भयो!", "Happy", "Fast", "High", ""],
        ],
        inputs=[text, emotion, speed, pitch, custom],
    )
    submit.click(generate, [text, emotion, speed, pitch, custom], [audio, status, used_description])
    text.submit(generate, [text, emotion, speed, pitch, custom], [audio, status, used_description])


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name=os.getenv("TTS_HOST", "127.0.0.1"),
        server_port=int(os.getenv("TTS_PORT", "7860")),
        show_error=True,
    )
