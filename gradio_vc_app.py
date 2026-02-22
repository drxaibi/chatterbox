import os
import time
import torch
import gradio as gr
from pathlib import Path
from chatterbox.vc import ChatterboxVC


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOCAL_VOICES_DIR = Path("voices")
MODEL = None
print(f"🚀 VC running on device: {DEVICE}")


def get_or_load_model():
    global MODEL
    if MODEL is None:
        print("Model not loaded, initializing VC model...")
        MODEL = ChatterboxVC.from_pretrained(DEVICE)
        print("VC model ready.")
    return MODEL


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
    return ["Default Voice", "Custom Upload/Mic", *LOCAL_VOICE_PRESETS.keys()]


def on_reference_mode_change(reference_preset: str):
    if reference_preset == "Custom Upload/Mic":
        return (
            gr.update(visible=True, value=None),
            gr.update(visible=False, value=None),
        )

    if reference_preset == "Default Voice":
        return (
            gr.update(visible=False, value=None),
            gr.update(visible=False, value=None),
        )

    preview = LOCAL_VOICE_PRESETS.get(reference_preset)
    return (
        gr.update(visible=False, value=None),
        gr.update(visible=True, value=preview),
    )


def refresh_reference_presets(selected: str | None):
    global LOCAL_VOICE_PRESETS
    LOCAL_VOICE_PRESETS = discover_local_voice_presets()
    choices = get_reference_preset_choices()

    if selected not in choices:
        selected = "Default Voice"

    target_update, preview_update = on_reference_mode_change(selected)
    return (
        gr.update(choices=choices, value=selected),
        target_update,
        preview_update,
    )


def generate(audio, reference_preset, target_voice_path):
    started = time.time()
    print("Starting voice conversion...")
    current_model = get_or_load_model()

    resolved_target = None
    if reference_preset == "Custom Upload/Mic":
        resolved_target = target_voice_path
    elif reference_preset in LOCAL_VOICE_PRESETS:
        resolved_target = LOCAL_VOICE_PRESETS[reference_preset]

    wav = current_model.generate(
        audio, target_voice_path=resolved_target,
    )
    elapsed = time.time() - started
    print(
        f"VC complete in {elapsed:.1f}s | preset={reference_preset} | "
        f"custom_target={'yes' if bool(target_voice_path) else 'no'}"
    )
    return current_model.sr, wav.squeeze(0).numpy()


with gr.Blocks() as demo:
    gr.Markdown("## 🎛️ Chatterbox Suite")
    gr.Markdown("### 🎤 Voice Conversion")

    input_audio = gr.Audio(
        sources=["upload", "microphone"],
        type="filepath",
        label="Input Audio",
    )

    reference_preset = gr.Dropdown(
        choices=get_reference_preset_choices(),
        value="Default Voice",
        label="Target Voice Preset",
    )
    refresh_presets_btn = gr.Button("Refresh Voices", size="sm")

    target_voice_path = gr.Audio(
        sources=["upload", "microphone"],
        type="filepath",
        label="Target Voice Audio (Custom)",
        value=None,
        visible=False,
    )

    preset_preview = gr.Audio(
        label="Preset Preview",
        type="filepath",
        interactive=False,
        visible=False,
    )

    convert_btn = gr.Button("Convert")
    output_audio = gr.Audio(label="Converted Audio")

    reference_preset.change(
        fn=on_reference_mode_change,
        inputs=[reference_preset],
        outputs=[target_voice_path, preset_preview],
        show_progress=False,
    )

    refresh_presets_btn.click(
        fn=refresh_reference_presets,
        inputs=[reference_preset],
        outputs=[reference_preset, target_voice_path, preset_preview],
        show_progress=False,
    )

    convert_btn.click(
        fn=generate,
        inputs=[input_audio, reference_preset, target_voice_path],
        outputs=[output_audio],
    )

if __name__ == "__main__":
    print("Starting VC web app...")
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
