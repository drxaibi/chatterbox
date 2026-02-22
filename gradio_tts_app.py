import random
import re
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve
import numpy as np
import torch
import gradio as gr
from chatterbox.tts import ChatterboxTTS


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EVENT_TAGS = [
    "[clear throat]", "[sigh]", "[shush]", "[cough]", "[groan]",
    "[sniff]", "[gasp]", "[chuckle]", "[laugh]"
]

REFERENCE_PRESETS = {
    "Default Female Podcast": "https://storage.googleapis.com/chatterbox-demo-samples/prompts/female_random_podcast.wav",
    "English Female (MTL)": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/en_f1.flac",
    "French Female (MTL)": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fr_f1.flac",
    "German Female (MTL)": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/de_f1.flac",
    "Custom Upload/Mic": None,
}

REFERENCE_CACHE_DIR = Path(".cache") / "reference_presets"
LOCAL_VOICES_DIR = Path("voices")

INSERT_TAG_JS = """
(tag_val, current_text) => {
    const textarea = document.querySelector('#main_textbox textarea');
    if (!textarea) return (current_text ? current_text : "") + " " + tag_val;

    current_text = current_text || "";
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


def load_model():
    model = ChatterboxTTS.from_pretrained(DEVICE)
    return model


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


def contains_event_tags(text: str) -> bool:
    return bool(re.search(r"\[(clear throat|sigh|shush|cough|groan|sniff|gasp|chuckle|laugh)\]", text or "", flags=re.IGNORECASE))


def strip_event_tags(text: str) -> str:
    return re.sub(r"\[(clear throat|sigh|shush|cough|groan|sniff|gasp|chuckle|laugh)\]", "", text or "", flags=re.IGNORECASE)


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


def generate(
    model,
    text,
    reference_preset,
    audio_prompt_path,
    exaggeration,
    temperature,
    seed_num,
    cfgw,
    min_p,
    top_p,
    repetition_penalty,
    chunk_size,
    pause_ms,
    progress=gr.Progress(),
):
    if model is None:
        model = ChatterboxTTS.from_pretrained(DEVICE)

    if seed_num != 0:
        set_seed(int(seed_num))

    effective_ref = resolve_reference_audio(reference_preset, audio_prompt_path)
    has_tags = contains_event_tags(text)
    clean_text = strip_event_tags(text) if has_tags else text
    chunks = split_text_into_chunks(clean_text, int(chunk_size))

    silence_samples = int(model.sr * (float(pause_ms) / 1000.0))
    audio_segments = []

    try:
        progress(0, desc=f"Preparing generation for {len(chunks)} chunk(s)")

        for idx, chunk_text in enumerate(chunks):
            progress((idx) / max(len(chunks), 1), desc=f"Generating chunk {idx + 1}/{len(chunks)}")
            wav = model.generate(
                chunk_text,
                audio_prompt_path=effective_ref,
                exaggeration=exaggeration,
                temperature=temperature,
                cfg_weight=cfgw,
                min_p=min_p,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            audio_segments.append(wav.squeeze(0).numpy())

            if silence_samples > 0 and idx < len(chunks) - 1:
                audio_segments.append(np.zeros(silence_samples, dtype=np.float32))

        final_wav = np.concatenate(audio_segments) if audio_segments else np.zeros(1, dtype=np.float32)
        progress(1.0, desc="Finalizing audio")
        return (model.sr, final_wav)
    except Exception as exc:
        raise gr.Error(f"Generation failed: {exc}")


with gr.Blocks() as demo:
    model_state = gr.State(None)  # Loaded once per session/user
    gr.Markdown("## 🎛️ Chatterbox Suite")
    gr.Markdown("### 🗣️ Standard TTS")
    gr.Markdown("Standard Chatterbox model: best for general TTS/voice cloning. Event tags like `[laugh]` are Turbo-only.")

    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                value="Now let's make my mum's favourite. So three mars bars into the pan. Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse. A sprinkle of olive oil and some tomato ketchup. Now smell that. Oh boy this is going to be incredible.",
                label="Text to synthesize (supports long scripts with auto chunking)",
                lines=10,
                elem_id="main_textbox"
            )
            chunk_plan_output = gr.Textbox(
                label="Chunk Plan",
                value=describe_chunk_plan(
                    "Now let's make my mum's favourite. So three mars bars into the pan. Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse. A sprinkle of olive oil and some tomato ketchup. Now smell that. Oh boy this is going to be incredible.",
                    300,
                ),
                interactive=False,
            )

            with gr.Row():
                for tag in EVENT_TAGS:
                    btn = gr.Button(tag)
                    btn.click(
                        fn=None,
                        inputs=[btn, text],
                        outputs=text,
                        js=INSERT_TAG_JS,
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
            exaggeration = gr.Slider(0.25, 2, step=.05, label="Exaggeration (Neutral = 0.5, extreme values can be unstable)", value=.5)
            cfg_weight = gr.Slider(0.0, 1, step=.05, label="CFG/Pace", value=0.5)

            with gr.Accordion("More options", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 5, step=.05, label="temperature", value=.8)
                min_p = gr.Slider(0.00, 1.00, step=0.01, label="min_p || Newer Sampler. Recommend 0.02 > 0.1. Handles Higher Temperatures better. 0.00 Disables", value=0.05)
                top_p = gr.Slider(0.00, 1.00, step=0.01, label="top_p || Original Sampler. 1.0 Disables(recommended). Original 0.8", value=1.00)
                repetition_penalty = gr.Slider(1.00, 2.00, step=0.1, label="repetition_penalty", value=1.2)
                chunk_size = gr.Slider(120, 600, step=10, value=300, label="Chunk size (chars per generation step)")
                pause_ms = gr.Slider(0, 500, step=25, value=120, label="Pause between chunks (ms)")

            run_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Output Audio")

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

    text.input(
        fn=describe_chunk_plan,
        inputs=[text, chunk_size],
        outputs=[chunk_plan_output],
        show_progress=False,
    )

    chunk_size.change(
        fn=describe_chunk_plan,
        inputs=[text, chunk_size],
        outputs=[chunk_plan_output],
        show_progress=False,
    )

    demo.load(fn=load_model, inputs=[], outputs=model_state)

    run_btn.click(
        fn=generate,
        inputs=[
            model_state,
            text,
            reference_preset,
            ref_wav,
            exaggeration,
            temp,
            seed_num,
            cfg_weight,
            min_p,
            top_p,
            repetition_penalty,
            chunk_size,
            pause_ms,
        ],
        outputs=[audio_output],
    )

if __name__ == "__main__":
    demo.queue(
        max_size=50,
        default_concurrency_limit=1,
    ).launch(
        share=False,
        inbrowser=True,
        server_name="127.0.0.1",
        server_port=7860,
        prevent_thread_lock=False,
    )
