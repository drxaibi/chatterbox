import random
import re
import time
import threading
import os
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve
import numpy as np
import torch
import gradio as gr
from chatterbox.tts_turbo import ChatterboxTurboTTS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = None
MODEL_LOCK = threading.Lock()
MODEL_LOAD_ERROR = None

EVENT_TAGS = [
    "[clear throat]", "[sigh]", "[shush]", "[cough]", "[groan]",
    "[sniff]", "[gasp]", "[chuckle]", "[laugh]"
]
EVENT_TAG_DEFS = [{"label": tag[1:-1], "value": tag} for tag in EVENT_TAGS]

REFERENCE_PRESETS = {
    "Default Female Podcast": "https://storage.googleapis.com/chatterbox-demo-samples/prompts/female_random_podcast.wav",
    "English Female (MTL)": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/en_f1.flac",
    "French Female (MTL)": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fr_f1.flac",
    "German Female (MTL)": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/de_f1.flac",
    "Custom Upload/Mic": None,
}

REFERENCE_CACHE_DIR = Path(".cache") / "reference_presets"
LOCAL_VOICES_DIR = Path("voices")

# --- REFINED CSS ---
# 1. tag-container: Forces the row to wrap items instead of scrolling. Removes borders/backgrounds.
# 2. tag-btn: Sets the specific look (indigo theme) and stops them from stretching.
CUSTOM_CSS = """
:root {
    --cb-editor-bg: #f9fafb;
    --cb-editor-border: #d1d5db;
    --cb-editor-text: #111827;
}

@media (prefers-color-scheme: dark) {
    :root {
        --cb-editor-bg: #111827;
        --cb-editor-border: #374151;
        --cb-editor-text: #f9fafb;
    }
}

.tag-container {
    display: grid !important;
    grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
    gap: 8px !important;
    margin-top: 5px !important;
    margin-bottom: 10px !important;
    border: none !important;
    background: transparent !important;
}

.tag-btn {
    width: 100% !important;
    height: 32px !important;
    font-size: 13px !important;
    background: #eef2ff !important;
    border: 1px solid #c7d2fe !important;
    color: #3730a3 !important;
    border-radius: 6px !important;
    padding: 0 10px !important;
    margin: 0 !important;
    box-shadow: none !important;
}

.tag-btn:hover {
    background: #c7d2fe !important;
    transform: translateY(-1px);
}

"""

INSERT_TAG_JS = """
(tag_val, current_text) => {
    current_text = current_text || "";
    tag_val = (tag_val || "").trim();
    if (!tag_val) return current_text;

    if (!tag_val.startsWith("[")) {
        tag_val = `[${tag_val}]`;
    }

    const textarea = document.querySelector('#main_textbox textarea');
    if (!textarea) {
        return current_text ? `${current_text} ${tag_val}` : tag_val;
    }

    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;

    let prefix = " ";
    let suffix = " ";

    if (start === 0) prefix = "";
    else if (current_text[start - 1] === ' ') prefix = "";

    if (end < current_text.length && current_text[end] === ' ') suffix = "";

    return current_text.slice(0, start) + prefix + tag_val + suffix + current_text.slice(end);
}
"""


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_model(raise_on_error: bool = True):
    global MODEL, MODEL_LOAD_ERROR
    with MODEL_LOCK:
        if MODEL is not None:
            return MODEL

        print(f"Loading Chatterbox-Turbo on {DEVICE}...")
        try:
            MODEL = ChatterboxTurboTTS.from_pretrained(DEVICE)
            MODEL_LOAD_ERROR = None
            print("Chatterbox-Turbo model ready.")
            return MODEL
        except Exception as exc:
            MODEL_LOAD_ERROR = str(exc)
            print(f"Turbo model load failed: {exc}")
            if raise_on_error:
                raise
            return None


def ensure_reference_presets() -> dict[str, str | None]:
    LOCAL_VOICES_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local_presets: dict[str, str | None] = {}

    local_voice_files: list[Path] = []
    for pattern in ["*.wav", "*.flac", "*.mp3", "*.WAV", "*.FLAC", "*.MP3"]:
        local_voice_files.extend(sorted(LOCAL_VOICES_DIR.glob(pattern)))

    for voice_file in local_voice_files:
        local_presets[voice_file.stem] = str(voice_file)

    for name, source in REFERENCE_PRESETS.items():
        if source is None:
            local_presets[name] = None
            continue

        if str(source).lower().startswith("http"):
            parsed = urlparse(source)
            suffix = Path(parsed.path).suffix or ".wav"
            safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", name.lower())
            local_path = REFERENCE_CACHE_DIR / f"{safe_name}{suffix}"

            if not local_path.exists():
                urlretrieve(source, local_path)

            local_presets[name] = str(local_path)
        else:
            local_presets[name] = str(source)

    return local_presets


LOCAL_REFERENCE_PRESETS = ensure_reference_presets()


def get_reference_preset_choices() -> list[str]:
    return list(LOCAL_REFERENCE_PRESETS.keys())


def split_text_into_chunks(text: str, max_chars: int) -> list[str]:
    def split_long_sentence(sentence: str) -> list[str]:
        sentence = sentence.strip()
        if len(sentence) <= max_chars:
            return [sentence]

        parts: list[str] = []
        remaining = sentence

        while len(remaining) > max_chars:
            window = remaining[:max_chars]

            boundary = -1
            for char in [".", "!", "?", ";", ":", ","]:
                pos = window.rfind(char)
                if pos > boundary:
                    boundary = pos + 1

            if boundary <= 0:
                boundary = window.rfind(" ")

            if boundary <= 0:
                boundary = max_chars

            candidate = remaining[:boundary].strip()
            if candidate.count("[") != candidate.count("]"):
                open_idx = candidate.rfind("[")
                if open_idx > 0:
                    prev_space = remaining.rfind(" ", 0, open_idx)
                    if prev_space > 0:
                        boundary = prev_space
                        candidate = remaining[:boundary].strip()

            if not candidate:
                boundary = max_chars
                candidate = remaining[:boundary].strip()

            parts.append(candidate)
            remaining = remaining[boundary:].strip()

        if remaining:
            parts.append(remaining)

        return parts

    text = " ".join((text or "").split())
    if not text:
        return ["You need to add some text for me to talk."]

    sentences = [s.strip() for s in re.split(r"(?<=[.!?。！？])\s+", text) if s.strip()]
    if not sentences:
        sentences = [text]

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(split_long_sentence(sentence))
            continue

        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return chunks


def describe_chunk_plan(text: str, chunk_size: int) -> str:
    chunks = split_text_into_chunks(text, int(chunk_size))
    lengths = [len(chunk) for chunk in chunks]
    if not lengths:
        return "Estimated chunks: 0"

    preview = ", ".join(str(length) for length in lengths[:12])
    if len(lengths) > 12:
        preview += ", ..."
    return (
        f"Estimated chunks: {len(chunks)} | Chunk size target: {int(chunk_size)} chars | "
        f"Chunk lengths: [{preview}]"
    )


def resolve_reference_audio(reference_preset: str, audio_prompt_path: str | None) -> str | None:
    if audio_prompt_path and str(audio_prompt_path).strip():
        return audio_prompt_path
    return LOCAL_REFERENCE_PRESETS.get(reference_preset)


def get_preset_preview_path(reference_preset: str) -> str | None:
    return LOCAL_REFERENCE_PRESETS.get(reference_preset)


def on_reference_mode_change(reference_preset: str):
    if reference_preset == "Custom Upload/Mic":
        return (
            gr.update(visible=True),
            gr.update(visible=False, value=None),
        )

    return (
        gr.update(visible=False, value=None),
        gr.update(visible=True, value=get_preset_preview_path(reference_preset)),
    )


def refresh_reference_presets(selected: str | None):
    global LOCAL_REFERENCE_PRESETS
    LOCAL_REFERENCE_PRESETS = ensure_reference_presets()
    choices = get_reference_preset_choices()

    if selected not in choices:
        selected = "Custom Upload/Mic" if "Custom Upload/Mic" in choices else (choices[0] if choices else None)

    ref_update, preview_update = on_reference_mode_change(selected) if selected else (
        gr.update(visible=True, value=None),
        gr.update(visible=False, value=None),
    )

    return (
        gr.update(choices=choices, value=selected),
        ref_update,
        preview_update,
    )


def sync_text_views(text: str, chunk_size: int):
    text = text or ""
    return describe_chunk_plan(text, int(chunk_size))


def generate(
        model,
        text,
        reference_preset,
        audio_prompt_path,
        temperature,
        seed_num,
        min_p,
        top_p,
        top_k,
        repetition_penalty,
        norm_loudness,
        chunk_size,
        pause_ms,
        progress=gr.Progress(),
):
    if model is None:
        model = load_model(raise_on_error=False)

    if model is None:
        message = MODEL_LOAD_ERROR or "Unknown model load error"
        if "paging file is too small" in message.lower() or "os error 1455" in message.lower():
            raise gr.Error(
                "Turbo model couldn't load due to low virtual memory (Windows paging file). "
                "Increase paging file size and close memory-heavy apps, then retry."
            )
        raise gr.Error(f"Turbo model couldn't load: {message}")

    if seed_num != 0:
        set_seed(int(seed_num))

    effective_ref = resolve_reference_audio(reference_preset, audio_prompt_path)
    chunks = split_text_into_chunks(text, int(chunk_size))

    silence_samples = int(model.sr * (float(pause_ms) / 1000.0))
    audio_segments = []

    started = time.time()
    try:
        progress(0, desc=f"Preparing generation for {len(chunks)} chunk(s)")
        print(f"[Plan] total_chunks={len(chunks)} | chunk_size={int(chunk_size)} | text_chars={len((text or '').strip())}")

        if effective_ref:
            model.prepare_conditionals(
                effective_ref,
                exaggeration=0.0,
                norm_loudness=norm_loudness,
            )

        for idx, chunk_text in enumerate(chunks):
            chunk_started = time.time()
            progress((idx) / max(len(chunks), 1), desc=f"Generating chunk {idx + 1}/{len(chunks)} | chars={len(chunk_text)}")
            wav = model.generate(
                chunk_text,
                audio_prompt_path=None,
                temperature=temperature,
                min_p=min_p,
                top_p=top_p,
                top_k=int(top_k),
                repetition_penalty=repetition_penalty,
                norm_loudness=norm_loudness,
            )
            wav_np = wav.squeeze(0).numpy()
            audio_segments.append(wav_np)

            chunk_elapsed = time.time() - chunk_started
            chunk_audio_seconds = len(wav_np) / float(model.sr)
            print(
                f"[Chunk {idx + 1}/{len(chunks)}] chars={len(chunk_text)} | "
                f"audio_s={chunk_audio_seconds:.2f} | wall_s={chunk_elapsed:.2f}"
            )

            if silence_samples > 0 and idx < len(chunks) - 1:
                audio_segments.append(np.zeros(silence_samples, dtype=np.float32))

        final_wav = np.concatenate(audio_segments) if audio_segments else np.zeros(1, dtype=np.float32)
        progress(1.0, desc="Finalizing audio")
        elapsed = time.time() - started
        print(
            f"[Done] total_chunks={len(chunks)} | elapsed_s={elapsed:.2f} | "
            f"text_chars={len((text or '').strip())}"
        )
        return (model.sr, final_wav)
    except Exception as exc:
        raise gr.Error(f"Generation failed: {exc}")


with gr.Blocks(title="Chatterbox Turbo", css=CUSTOM_CSS) as demo:
    gr.Markdown("## 🎛️ Chatterbox Suite")
    gr.Markdown("### ⚡ Turbo TTS")
    gr.Markdown("Turbo supports paralinguistic tags like `[laugh]`, `[cough]`, `[chuckle]`. Keep tags sparse for natural delivery.")

    model_state = gr.State(None)

    with gr.Row():
        with gr.Column():
            initial_text = "Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. It's the SkyNet T-800 series and it's got basically everything. Including AI integration with ChatGPT and um all that jazz. Would you like me to get some prices for you?"
            gr.Markdown("**Script Input**")
            text = gr.Textbox(
                value=initial_text,
                show_label=False,
                lines=10,
                elem_id="main_textbox"
            )
            chunk_plan_output = gr.Textbox(
                label="Chunk Plan",
                value=describe_chunk_plan(
                    initial_text,
                    360,
                ),
                interactive=False,
            )

            # --- Event Tags ---
            # Switched back to Row, but applied specific CSS to force wrapping
            with gr.Row(elem_classes=["tag-container"]):
                for tag in EVENT_TAG_DEFS:
                    # elem_classes targets the button specifically
                    btn = gr.Button(tag["label"], elem_classes=["tag-btn"])

                    btn.click(
                        fn=None,
                        inputs=[btn, text],
                        outputs=text,
                        js=INSERT_TAG_JS
                    )

            reference_preset = gr.Dropdown(
                choices=get_reference_preset_choices(),
                value="Custom Upload/Mic",
                label="Reference Voice Preset"
            )
            refresh_presets_btn = gr.Button("Refresh Voices", size="sm")
            preset_preview = gr.Audio(
                label="Preset Preview",
                value=None,
                type="filepath",
                interactive=False,
                visible=False,
            )

            ref_wav = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Custom Reference Audio (overrides preset if provided)",
                value=None,
                visible=True,
            )

            run_btn = gr.Button("Generate ⚡", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Output Audio")

            with gr.Accordion("Advanced Options", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 2.0, step=.05, label="Temperature", value=0.7)
                top_p = gr.Slider(0.00, 1.00, step=0.01, label="Top P", value=0.9)
                top_k = gr.Slider(0, 1000, step=10, label="Top K", value=1000)
                repetition_penalty = gr.Slider(1.00, 2.00, step=0.05, label="Repetition Penalty", value=1.15)
                min_p = gr.Slider(0.00, 1.00, step=0.01, label="Min P (Set to 0 to disable)", value=0.00)
                norm_loudness = gr.Checkbox(value=True, label="Normalize Loudness (-27 LUFS)")
                chunk_size = gr.Slider(120, 600, step=10, value=360, label="Chunk size (chars per generation step)")
                pause_ms = gr.Slider(0, 500, step=25, value=120, label="Pause between chunks (ms)")

    reference_preset.change(
        fn=on_reference_mode_change,
        inputs=[reference_preset],
        outputs=[ref_wav, preset_preview],
        show_progress=False,
    )

    refresh_presets_btn.click(
        fn=refresh_reference_presets,
        inputs=[reference_preset],
        outputs=[reference_preset, ref_wav, preset_preview],
        show_progress=False,
    )

    text.change(
        fn=sync_text_views,
        inputs=[text, chunk_size],
        outputs=[chunk_plan_output],
        show_progress=False,
    )

    text.input(
        fn=sync_text_views,
        inputs=[text, chunk_size],
        outputs=[chunk_plan_output],
        show_progress=False,
    )

    chunk_size.change(
        fn=sync_text_views,
        inputs=[text, chunk_size],
        outputs=[chunk_plan_output],
        show_progress=False,
    )

    run_btn.click(
        fn=generate,
        inputs=[
            model_state,
            text,
            reference_preset,
            ref_wav,
            temp,
            seed_num,
            min_p,
            top_p,
            top_k,
            repetition_penalty,
            norm_loudness,
            chunk_size,
            pause_ms,
        ],
        outputs=[audio_output],
    )

if __name__ == "__main__":
    print("Preloading Turbo assets and model (first run may take several minutes)...")
    load_model(raise_on_error=False)

    launch_kwargs = {
        "share": False,
        "inbrowser": True,
        "server_name": "127.0.0.1",
        "server_port": 7861,
        "prevent_thread_lock": False,
    }
    server_port_env = os.getenv("GRADIO_SERVER_PORT")
    if server_port_env:
        launch_kwargs["server_port"] = int(server_port_env)

    demo.queue(
        max_size=50,
        default_concurrency_limit=1,
    ).launch(**launch_kwargs)
