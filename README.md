# Nepali expressive TTS research

Small local GPU web app for testing [`ai4bharat/indic-parler-tts`](https://huggingface.co/ai4bharat/indic-parler-tts) with Nepali text, Amrita's voice, and emotion/style controls.

## What this tests

- Direct Nepali text-to-speech in the browser
- Amrita, the model card's recommended Nepali speaker
- Emotion, pace, pitch, and a free-form voice description
- GPU FP16 or CPU BF16/FP16 inference
- Downloadable WAV output and basic latency/audio-duration reporting

The checkpoint is gated. Open the model page, sign in, accept its access terms, and create a read token **from that same Hugging Face account** before the first run.

## Install and run

Python 3.10 or 3.11 is recommended. From this directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
cp .env.example .env
# Edit .env and replace hf_your_read_token with your real token.
python app.py
```

Then open <http://127.0.0.1:7860>. The initial request downloads roughly several GB of model files and will take longer than later requests.

### Fixing `Model access denied`

1. Sign into Hugging Face and visit <https://huggingface.co/ai4bharat/indic-parler-tts>.
2. Click the button to agree/request access. Being able to view the page is not enough; the account must have model-file access.
3. Create a token at <https://huggingface.co/settings/tokens>. A fine-grained token must grant read access to `ai4bharat/indic-parler-tts`; a classic read token also works.
4. Put it in `.env` with no quotes or spaces:

   ```dotenv
   HF_TOKEN=hf_actual_token_here
   ```

5. Stop the running app completely and start `python app.py` again. Environment changes are not picked up by an already-running Python process.

Optionally verify the same token before launching:

```bash
source .venv/bin/activate
huggingface-cli whoami --token "$(sed -n 's/^HF_TOKEN=//p' .env)"
```

Do not paste the token into chat, screenshots, source control, or the browser UI. `.env` is excluded by this project's `.gitignore`.

For another computer on the same network:

```bash
TTS_HOST=0.0.0.0 python app.py
```

For a CPU-only compatibility test (slow):

```bash
TTS_DEVICE=cpu python app.py
```

## Local CPU mode

Parler-TTS does not support Transformers' automatic CPU/GPU device map. On a 4 GB RTX 3050, use CPU BF16 instead:

```bash
TTS_DEVICE=cpu TTS_DTYPE=bfloat16 python app.py
```

BF16 halves model memory compared with FP32, but CPU generation is slow. If the processor lacks BF16 kernels, try `TTS_DTYPE=float16`; some CPU operations may still require FP32 depending on the installed PyTorch build. The recommended test environment is the included Kaggle GPU notebook.

## Kaggle: one-run GPU notebook

Open [`kaggle/indic_parler_tts_kaggle.ipynb`](kaggle/indic_parler_tts_kaggle.ipynb) in Kaggle, select a GPU accelerator, add a private Kaggle Secret named `HF_TOKEN`, and choose **Run all**. The notebook clones this repository, installs dependencies, loads CUDA FP16, generates three WAV files, and renders an audio player for each one.

## Agent integration

This is intentionally isolated from the existing voice agent. Once audio quality is accepted, `IndicParlerEngine.synthesize(answer, description)` can replace the Edge TTS call in `Nepali_AI_VOICE_RAG/local_agent/tts/tts_engine.py`. Keep the model loaded once at process startup; do not reload it per answer.

## Practical evaluation

Try the same 10–20 sentences with Neutral, News, Happy, and Sad. Have a native Nepali listener rate pronunciation, naturalness, emotion match, and annoying artifacts from 1–5. Include names, numbers, dates, English abbreviations, long sentences, and punctuation. The model card reports stronger results for the fine-tuned checkpoint but only one named Nepali speaker, so speaker choice is not a variable here.
