import random
import os
import re
import time
import numpy as np
import torch
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
import gradio as gr

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running on device: {DEVICE}")

# --- Global Model Initialization ---
MODEL = None
LOCAL_VOICES_DIR = Path("voices")
REFERENCE_CACHE_DIR = Path(".cache") / "reference_presets_mtl"

LANGUAGE_CONFIG = {
    "ar": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ar_f/ar_prompts2.flac",
        "text": "في الشهر الماضي، وصلنا إلى معلم جديد بمليارين من المشاهدات على قناتنا على يوتيوب."
    },
    "da": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/da_m1.flac",
        "text": "Sidste måned nåede vi en ny milepæl med to milliarder visninger på vores YouTube-kanal."
    },
    "de": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/de_f1.flac",
        "text": "Letzten Monat haben wir einen neuen Meilenstein erreicht: zwei Milliarden Aufrufe auf unserem YouTube-Kanal."
    },
    "el": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/el_m.flac",
        "text": "Τον περασμένο μήνα, φτάσαμε σε ένα νέο ορόσημο με δύο δισεκατομμύρια προβολές στο κανάλι μας στο YouTube."
    },
    "en": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/en_f1.flac",
        "text": "Last month, we reached a new milestone with two billion views on our YouTube channel."
    },
    "es": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/es_f1.flac",
        "text": "El mes pasado alcanzamos un nuevo hito: dos mil millones de visualizaciones en nuestro canal de YouTube."
    },
    "fi": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fi_m.flac",
        "text": "Viime kuussa saavutimme uuden virstanpylvään kahden miljardin katselukerran kanssa YouTube-kanavallamme."
    },
    "fr": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fr_f1.flac",
        "text": "Le mois dernier, nous avons atteint un nouveau jalon avec deux milliards de vues sur notre chaîne YouTube."
    },
    "he": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/he_m1.flac",
        "text": "בחודש שעבר הגענו לאבן דרך חדשה עם שני מיליארד צפיות בערוץ היוטיוב שלנו."
    },
    "hi": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/hi_f1.flac",
        "text": "पिछले महीने हमने एक नया मील का पत्थर छुआ: हमारे YouTube चैनल पर दो अरब व्यूज़।"
    },
    "it": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/it_m1.flac",
        "text": "Il mese scorso abbiamo raggiunto un nuovo traguardo: due miliardi di visualizzazioni sul nostro canale YouTube."
    },
    "ja": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ja/ja_prompts1.flac",
        "text": "先月、私たちのYouTubeチャンネルで二十億回の再生回数という新たなマイルストーンに到達しました。"
    },
    "ko": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ko_f.flac",
        "text": "지난달 우리는 유튜브 채널에서 이십억 조회수라는 새로운 이정표에 도달했습니다."
    },
    "ms": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ms_f.flac",
        "text": "Bulan lepas, kami mencapai pencapaian baru dengan dua bilion tontonan di saluran YouTube kami."
    },
    "nl": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/nl_m.flac",
        "text": "Vorige maand bereikten we een nieuwe mijlpaal met twee miljard weergaven op ons YouTube-kanaal."
    },
    "no": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/no_f1.flac",
        "text": "Forrige måned nådde vi en ny milepæl med to milliarder visninger på YouTube-kanalen vår."
    },
    "pl": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/pl_m.flac",
        "text": "W zeszłym miesiącu osiągnęliśmy nowy kamień milowy z dwoma miliardami wyświetleń na naszym kanale YouTube."
    },
    "pt": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/pt_m1.flac",
        "text": "No mês passado, alcançámos um novo marco: dois mil milhões de visualizações no nosso canal do YouTube."
    },
    "ru": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ru_m.flac",
        "text": "В прошлом месяце мы достигли нового рубежа: два миллиарда просмотров на нашем YouTube-канале."
    },
    "sv": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/sv_f.flac",
        "text": "Förra månaden nådde vi en ny milstolpe med två miljarder visningar på vår YouTube-kanal."
    },
    "sw": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/sw_m.flac",
        "text": "Mwezi uliopita, tulifika hatua mpya ya maoni ya bilioni mbili kweny kituo chetu cha YouTube."
    },
    "tr": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/tr_m.flac",
        "text": "Geçen ay YouTube kanalımızda iki milyar görüntüleme ile yeni bir dönüm noktasına ulaştık."
    },
    "zh": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/zh_f2.flac",
        "text": "上个月，我们达到了一个新的里程碑. 我们的YouTube频道观看次数达到了二十亿次，这绝对令人难以置信。"
    },
}

# --- UI Helpers ---
def default_audio_for_ui(lang: str) -> str | None:
    return LANGUAGE_CONFIG.get(lang, {}).get("audio")


def ensure_local_audio(source: str | None, key_hint: str = "ref") -> str | None:
    if not source:
        return None

    source = str(source).strip()
    if not source:
        return None

    if source.lower().startswith("http"):
        REFERENCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        parsed = urlparse(source)
        suffix = Path(parsed.path).suffix or ".wav"
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in key_hint.lower())
        local_path = REFERENCE_CACHE_DIR / f"{safe_name}{suffix}"
        if not local_path.exists():
            urlretrieve(source, local_path)
        return str(local_path)

    return source


def discover_local_voice_presets() -> dict[str, str]:
    LOCAL_VOICES_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, str] = {}
    files: list[Path] = []
    for pattern in ["*.wav", "*.flac", "*.mp3", "*.WAV", "*.FLAC", "*.MP3"]:
        files.extend(sorted(LOCAL_VOICES_DIR.glob(pattern)))
    for path in files:
        results[path.stem] = str(path)
    return results


LOCAL_VOICE_PRESETS = discover_local_voice_presets()


def get_reference_preset_choices() -> list[str]:
    return ["Language Default", "Custom Upload/Mic", *LOCAL_VOICE_PRESETS.keys()]


def default_text_for_ui(lang: str) -> str:
    return LANGUAGE_CONFIG.get(lang, {}).get("text", "")


def get_supported_languages_display() -> str:
    """Generate a formatted display of all supported languages."""
    language_items = []
    for code, name in sorted(SUPPORTED_LANGUAGES.items()):
        language_items.append(f"**{name}** (`{code}`)")
    
    # Split into 2 lines
    mid = len(language_items) // 2
    line1 = " • ".join(language_items[:mid])
    line2 = " • ".join(language_items[mid:])
    
    return f"""
### 🌍 Supported Languages ({len(SUPPORTED_LANGUAGES)} total)
{line1}

{line2}
"""


def get_or_load_model():
    """Loads the ChatterboxMultilingualTTS model if it hasn't been loaded already,
    and ensures it's on the correct device."""
    global MODEL
    if MODEL is None:
        print("Model not loaded, initializing...")
        try:
            MODEL = ChatterboxMultilingualTTS.from_pretrained(DEVICE)
            if hasattr(MODEL, 'to') and str(MODEL.device) != DEVICE:
                MODEL.to(DEVICE)
            print(f"Model loaded successfully. Internal device: {getattr(MODEL, 'device', 'N/A')}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    return MODEL

print("ℹ️ Multilingual model will load on first Generate (initial download can take several minutes).")

def set_seed(seed: int):
    """Sets the random seed for reproducibility across torch, numpy, and random."""
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    
def resolve_audio_prompt(language_id: str, reference_preset: str, provided_path: str | None) -> str | None:
    """
    Decide which audio prompt to use:
    - If user provided a path (upload/mic/url), use it.
    - Else, fall back to language-specific default (if any).
    """
    if reference_preset == "Custom Upload/Mic":
        return ensure_local_audio(provided_path, key_hint="custom_upload")

    if reference_preset == "Language Default":
        default_url = LANGUAGE_CONFIG.get(language_id, {}).get("audio")
        return ensure_local_audio(default_url, key_hint=f"lang_{language_id}")

    if reference_preset in LOCAL_VOICE_PRESETS:
        return LOCAL_VOICE_PRESETS[reference_preset]

    return ensure_local_audio(provided_path, key_hint="fallback")


def on_reference_mode_change(reference_preset: str, language_id: str):
    if reference_preset == "Custom Upload/Mic":
        return (
            gr.update(visible=True, value=None),
            gr.update(visible=False, value=None),
        )

    if reference_preset == "Language Default":
        preview = ensure_local_audio(default_audio_for_ui(language_id), key_hint=f"lang_{language_id}")
        return (
            gr.update(visible=False, value=None),
            gr.update(visible=True, value=preview),
        )

    preview = LOCAL_VOICE_PRESETS.get(reference_preset)
    return (
        gr.update(visible=False, value=None),
        gr.update(visible=True, value=preview),
    )


def refresh_reference_presets(selected: str | None, language_id: str):
    global LOCAL_VOICE_PRESETS
    LOCAL_VOICE_PRESETS = discover_local_voice_presets()
    choices = get_reference_preset_choices()

    if selected not in choices:
        selected = "Language Default"

    ref_update, preview_update = on_reference_mode_change(selected, language_id)
    return (
        gr.update(choices=choices, value=selected),
        ref_update,
        preview_update,
    )


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
            for char in [".", "!", "?", ";", ":", ",", "。", "！", "？"]:
                pos = window.rfind(char)
                if pos > boundary:
                    boundary = pos + 1

            if boundary <= 0:
                boundary = window.rfind(" ")

            if boundary <= 0:
                boundary = max_chars

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


def sync_text_views(text: str, chunk_size: int):
    text = text or ""
    return describe_chunk_plan(text, int(chunk_size))


def generate_tts_audio(
    text_input: str,
    language_id: str,
    reference_preset: str,
    audio_prompt_path_input: str = None,
    exaggeration_input: float = 0.5,
    temperature_input: float = 0.8,
    seed_num_input: int = 0,
    cfgw_input: float = 0.5,
    min_p_input: float = 0.05,
    top_p_input: float = 1.0,
    repetition_penalty_input: float = 2.0,
    chunk_size_input: int = 300,
    pause_ms_input: int = 120,
    progress=gr.Progress(),
):
    """
    Generate high-quality speech audio from text using Chatterbox Multilingual model with optional reference audio styling.
    Supported languages: English, French, German, Spanish, Italian, Portuguese, and Hindi.
    
    This tool synthesizes natural-sounding speech from input text. When a reference audio file 
    is provided, it captures the speaker's voice characteristics and speaking style. The generated audio 
    maintains the prosody, tone, and vocal qualities of the reference speaker, or uses default voice if no reference is provided.

    Args:
        text_input (str): The text to synthesize into speech (maximum 300 characters)
        language_id (str): The language code for synthesis (eg. en, fr, de, es, it, pt, hi)
        audio_prompt_path_input (str, optional): File path or URL to the reference audio file that defines the target voice style. Defaults to None.
        exaggeration_input (float, optional): Controls speech expressiveness (0.25-2.0, neutral=0.5, extreme values may be unstable). Defaults to 0.5.
        temperature_input (float, optional): Controls randomness in generation (0.05-5.0, higher=more varied). Defaults to 0.8.
        seed_num_input (int, optional): Random seed for reproducible results (0 for random generation). Defaults to 0.
        cfgw_input (float, optional): CFG/Pace weight controlling generation guidance (0.2-1.0). Defaults to 0.5, 0 for language transfer. 

    Returns:
        tuple[int, np.ndarray]: A tuple containing the sample rate (int) and the generated audio waveform (numpy.ndarray)
    """
    current_model = get_or_load_model()

    if current_model is None:
        raise RuntimeError("TTS model is not loaded.")

    if seed_num_input != 0:
        set_seed(int(seed_num_input))

    print(f"Generating audio for text: '{(text_input or '')[:50]}...'")
    
    # Handle optional audio prompt
    chosen_prompt = resolve_audio_prompt(language_id, reference_preset, audio_prompt_path_input)

    if chosen_prompt:
        print(f"Using audio prompt: {chosen_prompt}")
        current_model.prepare_conditionals(chosen_prompt, exaggeration=exaggeration_input)
    else:
        print("No audio prompt provided; using default voice.")

    chunks = split_text_into_chunks(text_input, int(chunk_size_input))
    silence_samples = int(current_model.sr * (float(pause_ms_input) / 1000.0))
    audio_segments = []

    started = time.time()
    try:
        progress(0, desc=f"Preparing generation for {len(chunks)} chunk(s)")

        for idx, chunk_text in enumerate(chunks):
            progress((idx) / max(len(chunks), 1), desc=f"Generating chunk {idx + 1}/{len(chunks)}")
            wav = current_model.generate(
                chunk_text,
                language_id=language_id,
                audio_prompt_path=None,
                exaggeration=exaggeration_input,
                cfg_weight=cfgw_input,
                temperature=temperature_input,
                repetition_penalty=repetition_penalty_input,
                min_p=min_p_input,
                top_p=top_p_input,
            )
            audio_segments.append(wav.squeeze(0).numpy())

            if silence_samples > 0 and idx < len(chunks) - 1:
                audio_segments.append(np.zeros(silence_samples, dtype=np.float32))

        final_wav = np.concatenate(audio_segments) if audio_segments else np.zeros(1, dtype=np.float32)
        progress(1.0, desc="Finalizing audio")
        elapsed = time.time() - started
        print(
            f"Generated {len(chunks)} chunk(s) in {elapsed:.1f}s on {DEVICE}. "
            f"Text length: {len((text_input or '').strip())} chars."
        )
        print("Audio generation complete.")
        return (current_model.sr, final_wav)
    except Exception as exc:
        raise gr.Error(f"Generation failed: {exc}")

with gr.Blocks() as demo:
    gr.Markdown(
        """
        ## 🎛️ Chatterbox Suite
        ### 🌍 Multilingual TTS
        High-quality multilingual speech from text with reference voice styling (23 languages, long-form chunking supported).
        """
    )
    
    # Display supported languages
    gr.Markdown(get_supported_languages_display())
    with gr.Row():
        with gr.Column():
            initial_lang = "fr"
            text = gr.Textbox(
                value=default_text_for_ui(initial_lang),
                label="Text to synthesize (supports long scripts with auto chunking)",
                lines=10
            )

            chunk_plan_output = gr.Textbox(
                label="Chunk Plan",
                value=describe_chunk_plan(default_text_for_ui(initial_lang), 300),
                interactive=False,
            )
            
            language_id = gr.Dropdown(
                choices=list(ChatterboxMultilingualTTS.get_supported_languages().keys()),
                value=initial_lang,
                label="Language",
                info="Select the language for text-to-speech synthesis"
            )

            reference_preset = gr.Dropdown(
                choices=get_reference_preset_choices(),
                value="Language Default",
                label="Reference Voice Preset"
            )
            refresh_presets_btn = gr.Button("Refresh Voices", size="sm")

            preset_preview = gr.Audio(
                label="Preset Preview",
                value=ensure_local_audio(default_audio_for_ui(initial_lang), key_hint=f"lang_{initial_lang}"),
                type="filepath",
                interactive=False,
                visible=True,
            )
            
            ref_wav = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Reference Audio File (Optional)",
                value=None,
                visible=False,
            )
            
            gr.Markdown(
                "💡 **Note**: Ensure that the reference clip matches the specified language tag. Otherwise, language transfer outputs may inherit the accent of the reference clip's language. To mitigate this, set the CFG weight to 0.",
                elem_classes=["audio-note"]
            )
            
            exaggeration = gr.Slider(
                0.25, 2, step=.05, label="Exaggeration (Neutral = 0.5, extreme values can be unstable)", value=.5
            )
            cfg_weight = gr.Slider(
                0.2, 1, step=.05, label="CFG/Pace", value=0.5
            )

            with gr.Accordion("More options", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 5, step=.05, label="Temperature", value=.8)
                min_p = gr.Slider(0.00, 1.00, step=0.01, label="min_p", value=0.05)
                top_p = gr.Slider(0.00, 1.00, step=0.01, label="top_p", value=1.00)
                repetition_penalty = gr.Slider(1.00, 3.00, step=0.1, label="repetition_penalty", value=2.0)
                chunk_size = gr.Slider(120, 600, step=10, value=300, label="Chunk size (chars per generation step)")
                pause_ms = gr.Slider(0, 500, step=25, value=120, label="Pause between chunks (ms)")

            run_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Output Audio")

        def on_language_change(lang, current_ref, current_text, current_preset):
            ref_update, preview_update = on_reference_mode_change(current_preset, lang)
            text_value = (current_text or "").strip()
            if text_value:
                return current_text, ref_update, preview_update
            return default_text_for_ui(lang), ref_update, preview_update

        language_id.change(
            fn=on_language_change,
            inputs=[language_id, ref_wav, text, reference_preset],
            outputs=[text, ref_wav, preset_preview],
            show_progress=False
        )

        reference_preset.change(
            fn=on_reference_mode_change,
            inputs=[reference_preset, language_id],
            outputs=[ref_wav, preset_preview],
            show_progress=False
        )

        refresh_presets_btn.click(
            fn=refresh_reference_presets,
            inputs=[reference_preset, language_id],
            outputs=[reference_preset, ref_wav, preset_preview],
            show_progress=False
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
        fn=generate_tts_audio,
        inputs=[
            text,
            language_id,
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

launch_kwargs = {
    "share": False,
    "inbrowser": True,
    "server_name": "127.0.0.1",
    "prevent_thread_lock": False,
}

server_port_env = os.getenv("GRADIO_SERVER_PORT")
if server_port_env:
    launch_kwargs["server_port"] = int(server_port_env)

demo.launch(**launch_kwargs)
