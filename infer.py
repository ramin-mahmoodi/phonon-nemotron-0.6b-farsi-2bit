"""
Phonon 2.5 Inference Script
Command-line inference for Persian speech recognition
Usage:
    python infer.py --audio speech.wav [--device cuda:0]
"""

import argparse
import torch
import torchaudio
import torchaudio.functional as taf
from engine import load_phonon_model


def transcribe(audio_path: str, model, processor, device: str = "cuda:0") -> str:
    # 1. Load and resample audio
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = taf.resample(waveform, orig_freq=sr, new_freq=16000)
    audio_np = waveform.squeeze().numpy()

    # 2. Extract features
    inputs = processor(audio_np, sampling_rate=16000, return_tensors="pt", language="fa-IR")
    inputs_dev = {
        k: v.to(device, dtype=torch.float16) if v.is_floating_point() else v.to(device)
        for k, v in inputs.items()
        if torch.is_tensor(v)
    }

    # 3. Streamed Generation
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.float16) if "cuda" in device else torch.no_grad():
            out = model.generate(**inputs_dev, return_dict_in_generate=True)
            text = processor.batch_decode(out.sequences, skip_special_tokens=True)[0]
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Transcribe Persian audio using Phonon 2.5 (Sub-200MB 2-bit)")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file (.wav, .mp3, .flac)")
    parser.add_argument("--container", type=str, default=None, help="Path to local .bps.tar.zst container (optional)")
    parser.add_argument(
        "--repo_id",
        type=str,
        default="ramin-mahmoodi/phonon-nemotron-0.6b-farsi-2bit",
        help="Hugging Face repo ID",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Target device (cuda:0 or cpu)",
    )
    args = parser.parse_args()

    model, processor = load_phonon_model(container_path=args.container, repo_id=args.repo_id, device=args.device)

    print(f"\n🎙️ Transcribing: {args.audio}")
    transcription = transcribe(args.audio, model, processor, device=args.device)
    print("=" * 60)
    print(f"📝 متن رونویسی‌شده:\n{transcription}")
    print("=" * 60)


if __name__ == "__main__":
    main()
