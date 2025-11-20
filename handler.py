import os
import uuid
import time
import base64
import traceback
import torchaudio
import soundfile as sf
import runpod
from io import BytesIO
from acestep.pipeline_ace_step import ACEStepPipeline

INIT_ERROR_FILE = "/tmp/init_error.log"
model = None

try:
    if os.path.exists(INIT_ERROR_FILE):
        os.remove(INIT_ERROR_FILE)

    print("Loading ACE-Step inpaint model...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(current_dir, "checkpoints")
    model = ACEStepPipeline(
        checkpoint_dir=checkpoint_path,
        dtype="bfloat16",
        torch_compile=False,
        cpu_offload=True,
        overlapped_decode=True,
    )
    if not model.loaded:
        model.load_checkpoint()
    print("✅ Model loaded.")
except Exception:
    with open(INIT_ERROR_FILE, "w") as f:
        f.write(traceback.format_exc())
    model = None


def handler(event):
    if os.path.exists(INIT_ERROR_FILE):
        with open(INIT_ERROR_FILE, "r") as f:
            return {"error": f"Model failed to load: {f.read()}"}

    job_input = event.get("input", {})
    audio_path = job_input.get("audio_path")
    start_time_sec = float(job_input.get("start_time", 30))
    end_time_sec = float(job_input.get("end_time", 60))
    prompt = job_input.get("prompt", "rock guitar with drums, full drum kit, stereo")

    if not audio_path:
        return {"error": "audio_path is required"}

    # locate file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        audio_path,
        os.path.join(current_dir, audio_path),
        os.path.join("/workspace", audio_path),
        os.path.join("/workspace", "outputs", os.path.basename(audio_path)),
    ]
    resolved = next((p for p in possible_paths if os.path.exists(p)), None)
    if resolved is None:
        return {"error": f"Audio file not found. Tried: {', '.join(possible_paths)}"}

    try:
        info = torchaudio.info(resolved)
        total_duration = info.num_frames / info.sample_rate
    except Exception:
        total_duration = 30.0

    # hardcoded params
    variance = 0.75
    infer_step = 60
    guidance_scale = 10.0
    scheduler_type = "euler"
    cfg_type = "apg"
    omega_scale = 10.0
    actual_seeds = [43]
    guidance_interval = 0.5
    guidance_interval_decay = 0.0
    min_guidance_scale = 3.0
    use_erg_tag = True
    use_erg_lyric = True
    use_erg_diffusion = True
    guidance_scale_text = 6.0
    guidance_scale_lyric = 0.0

    out_path = f"/tmp/inpainted_{uuid.uuid4().hex}.wav"
    start_t = time.time()
    try:
        model(
            format="wav",
            audio_duration=total_duration,
            prompt=prompt,
            lyrics="[inst]",
            infer_step=infer_step,
            guidance_scale=guidance_scale,
            scheduler_type=scheduler_type,
            cfg_type=cfg_type,
            omega_scale=omega_scale,
            manual_seeds=actual_seeds,
            guidance_interval=guidance_interval,
            guidance_interval_decay=guidance_interval_decay,
            min_guidance_scale=min_guidance_scale,
            use_erg_tag=use_erg_tag,
            use_erg_lyric=use_erg_lyric,
            use_erg_diffusion=use_erg_diffusion,
            oss_steps=None,
            guidance_scale_text=guidance_scale_text,
            guidance_scale_lyric=guidance_scale_lyric,
            save_path=out_path,
            task="repaint",
            repaint_start=int(start_time_sec),
            repaint_end=int(end_time_sec),
            retake_variance=variance,
            src_audio_path=resolved,
        )
        with open(out_path, "rb") as f:
            wav_bytes = f.read()
        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
        elapsed = time.time() - start_t
        print(f"[INFO] Inpainting took {elapsed:.2f}s")
        return {
            "status": "completed",
            "audio_base64": audio_b64,
            "format": "wav",
            "message": f"Inpainted {start_time_sec}-{end_time_sec}s in {elapsed:.2f}s",
        }
    except Exception:
        err = traceback.format_exc()
        print(err)
        return {"status": "failed", "error": err}


runpod.serverless.start({"handler": handler})
