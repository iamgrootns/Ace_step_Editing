import os
import torch
import torchaudio
import runpod
import base64
from io import BytesIO
import traceback
import requests
import urllib.parse
import uuid
import soundfile as sf
import time
from huggingface_hub import snapshot_download

# --- Global Variables & Model Loading ---
INIT_ERROR_FILE = "/tmp/init_error.log"
model_demo = None

try:
    if os.path.exists(INIT_ERROR_FILE):
        os.remove(INIT_ERROR_FILE)

    print("Loading ACEStep model for inpainting...")
    from acestep.pipeline_ace_step import ACEStepPipeline

    # ---------------------------------------------------------
    # FIXED CHECKPOINT PATH + AUTO DOWNLOAD
    # ---------------------------------------------------------
    checkpoint_path = os.environ.get("CHECKPOINT_PATH", "/runpod-volume/checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)

    # If model not present, download once
    has_ckpt = any(fname.endswith(".safetensors") for fname in os.listdir(checkpoint_path))

    if not has_ckpt:
        print(f"📥 Downloading ACEStep checkpoints to {checkpoint_path}...")
        snapshot_download(
            repo_id="ACE-Step/ACE-Step-v1-3.5B",
            local_dir=checkpoint_path,
            local_dir_use_symlinks=False
        )
        print("✅ Checkpoints downloaded")
    else:
        print(f"✅ Using cached checkpoints from {checkpoint_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_demo = ACEStepPipeline(
        checkpoint_dir=checkpoint_path,
        dtype="bfloat16",
        torch_compile=False,
        cpu_offload=True,
        overlapped_decode=True,
    )

    if not model_demo.loaded:
        model_demo.load_checkpoint()

    print("✅ ACEStep inpaint model loaded successfully")

except Exception as e:
    tb_str = traceback.format_exc()
    with open(INIT_ERROR_FILE, "w") as f:
        f.write(f"Failed to initialize ACEStep model: {tb_str}")
    model_demo = None


# --- Helper Functions ---
def download_audio_from_url(url, save_path="/tmp/input_audio.wav"):
    """Download audio file from URL"""
    try:
        print(f"📥 Downloading audio from: {url[:100]}...")
        response = requests.get(url, timeout=120)
        response.raise_for_status()

        with open(save_path, "wb") as f:
            f.write(response.content)

        print(f"✅ Audio downloaded: {save_path}")
        return save_path
    except Exception as e:
        print(f"❌ Download failed: {e}")
        raise


def upload_to_gcs(signed_url, audio_bytes, content_type="audio/wav"):
    """Upload audio to Google Cloud Storage"""
    try:
        response = requests.put(
            signed_url,
            data=audio_bytes,
            headers={"Content-Type": content_type},
            timeout=300
        )
        response.raise_for_status()
        print(f"✅ Uploaded to GCS: {signed_url[:100]}...")
        return True
    except Exception as e:
        print(f"❌ GCS upload failed: {e}")
        return False


def notify_backend(callback_url, status, error_message=None):
    """Send webhook notification"""
    try:
        parsed = urllib.parse.urlparse(callback_url)
        params = urllib.parse.parse_qs(parsed.query)
        params['status'] = [status]
        if error_message:
            params['error_message'] = [error_message]

        new_query = urllib.parse.urlencode(params, doseq=True)
        webhook_url = urllib.parse.urlunparse((
            parsed.scheme, parsed.netloc, parsed.path,
            parsed.params, new_query, parsed.fragment
        ))

        print(f"🔔 Calling webhook: {webhook_url}")
        response = requests.post(webhook_url, timeout=30)
        response.raise_for_status()
        print(f"✅ Backend notified: {status}")
        return True
    except Exception as e:
        print(f"❌ Webhook notification failed: {e}")
        return False


def patch_save_method(model):
    """Patch save method to capture output"""
    original_save = model.save_wav_file

    def patched_save(target_wav, idx, save_path=None, sample_rate=48000, format="wav"):
        if save_path is None:
            base_path = "/tmp/outputs"
            os.makedirs(base_path, exist_ok=True)
            output_path_wav = f"{base_path}/output_{time.strftime('%Y%m%d%H%M%S')}_{idx}.{format}"
        else:
            if os.path.isdir(save_path):
                output_path_wav = os.path.join(
                    save_path, f"output_{time.strftime('%Y%m%d%H%M%S')}_{idx}.{format}"
                )
            else:
                output_path_wav = save_path
            output_dir = os.path.dirname(os.path.abspath(output_path_wav))
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

        target_wav = target_wav.float().cpu()
        audio_data = target_wav.numpy() if target_wav.dim() == 1 else target_wav.T.numpy()

        sf.write(output_path_wav, audio_data, sample_rate)
        return output_path_wav

    return original_save, patched_save


# --- Runpod Handler ---
def handler(event):
    if os.path.exists(INIT_ERROR_FILE):
        with open(INIT_ERROR_FILE, "r") as f:
            error_msg = f"Worker initialization failed: {f.read()}"
        return {"error": error_msg, "status": "failed"}

    job_input = event.get("input", {})

    # Required parameters
    audio_url = job_input.get("audio_url")
    audio_base64 = job_input.get("audio_base64")
    start_time = job_input.get("start_time")
    end_time = job_input.get("end_time")
    prompt = job_input.get("prompt")

    if not prompt or start_time is None or end_time is None:
        error_msg = "Missing required parameters: prompt, start_time, end_time"
        return {"error": error_msg, "status": "failed"}

    if not audio_url and not audio_base64:
        return {"error": "Must provide either audio_url or audio_base64", "status": "failed"}

    callback_url = job_input.get("callback_url")
    upload_urls = job_input.get("upload_urls", {})

    try:
        # Download or decode input audio
        if audio_url:
            input_audio_path = download_audio_from_url(audio_url)
        else:
            input_audio_path = "/tmp/input_audio.wav"
            with open(input_audio_path, "wb") as f:
                f.write(base64.b64decode(audio_base64))

        # Read duration
        try:
            info = torchaudio.info(input_audio_path)
            total_duration = info.num_frames / info.sample_rate
        except:
            total_duration = 30.0

        output_path = f"/tmp/inpainted_{uuid.uuid4().hex}.wav"

        # Patch save method
        original_save, patched_save = patch_save_method(model_demo)
        model_demo.save_wav_file = patched_save

        start = time.time()

        # Run model
        model_demo(
            format="wav",
            audio_duration=total_duration,
            prompt=prompt,
            lyrics="[inst]",
            infer_step=job_input.get("infer_step", 60),
            guidance_scale=job_input.get("guidance_scale", 10.0),
            scheduler_type=job_input.get("scheduler_type", "euler"),
            cfg_type=job_input.get("cfg_type", "apg"),
            omega_scale=job_input.get("omega_scale", 10.0),
            manual_seeds=job_input.get("manual_seeds", [43]),
            guidance_interval=job_input.get("guidance_interval", 0.5),
            guidance_interval_decay=job_input.get("guidance_interval_decay", 0.0),
            min_guidance_scale=job_input.get("min_guidance_scale", 3.0),
            use_erg_tag=job_input.get("use_erg_tag", True),
            use_erg_lyric=job_input.get("use_erg_lyric", True),
            use_erg_diffusion=job_input.get("use_erg_diffusion", True),
            guidance_scale_text=job_input.get("guidance_scale_text", 6.0),
            guidance_scale_lyric=job_input.get("guidance_scale_lyric", 0.0),
            save_path=output_path,
            task="repaint",
            repaint_start=int(start_time),
            repaint_end=int(end_time),
            retake_variance=job_input.get("variance", 0.75),
            src_audio_path=input_audio_path,
        )

        model_demo.save_wav_file = original_save

        processing_time = time.time() - start
        with open(output_path, "rb") as f:
            audio_bytes = f.read()

        audio_base64_out = base64.b64encode(audio_bytes).decode()

        # Cleanup
        os.remove(output_path)
        os.remove(input_audio_path)

        return {
            "audio_base64": audio_base64_out,
            "sample_rate": 48000,
            "format": "wav",
            "inpainted_section": f"{start_time}-{end_time}",
            "processing_time": processing_time,
            "status": "completed",
        }

    except Exception as e:
        return {"error": traceback.format_exc(), "status": "failed"}


# Start worker
runpod.serverless.start({"handler": handler})
