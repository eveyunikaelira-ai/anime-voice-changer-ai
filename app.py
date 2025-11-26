import argparse
import queue
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sounddevice as sd
import torch
import torchaudio
from torch import nn


@dataclass
class EffectSettings:
    pitch_steps: float = 0.0
    formant_scale: float = 1.0
    reverb_seconds: float = 0.3
    reverb_decay: float = 0.4


class AnimeVoiceNet(nn.Module):
    """
    Lightweight placeholder network that colors the timbre toward an anime
    character-style voice.

    In a production system you would replace this with a larger pretrained
    model, but this network keeps the sample runnable while demonstrating
    PyTorch deployment details.
    """

    def __init__(self, channels: int = 1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=9, padding=4),
            nn.LeakyReLU(0.1),
            nn.Conv1d(32, 64, kernel_size=9, padding=4),
            nn.LeakyReLU(0.1),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.LeakyReLU(0.1),
            nn.Conv1d(32, channels, kernel_size=5, padding=2),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        return self.decoder(encoded)


def build_reverb_ir(sample_rate: int, seconds: float, decay: float) -> torch.Tensor:
    if seconds <= 0:
        return torch.tensor([1.0], dtype=torch.float32)

    num_samples = max(1, int(sample_rate * seconds))
    times = torch.linspace(0, seconds, steps=num_samples)
    envelope = torch.exp(-decay * times)
    noise = torch.randn(num_samples) * 0.1
    ir = envelope * noise
    ir[0] = 1.0
    return ir


def apply_effects(
    waveform: torch.Tensor,
    sample_rate: int,
    settings: EffectSettings,
    device: torch.device,
    reverb_ir: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply pitch shift, formant shaping, and reverb to a mono waveform."""

    if waveform.dim() != 2 or waveform.size(0) != 1:
        raise ValueError("Waveform must be mono with shape [1, num_samples].")

    with torch.no_grad():
        shifted = torchaudio.functional.pitch_shift(
            waveform, sample_rate=sample_rate, n_steps=settings.pitch_steps
        )

        formant = shifted
        if settings.formant_scale != 1.0:
            formant = formant.clone()
            base_formants = [300.0, 800.0, 1500.0, 2400.0]
            for f in base_formants:
                center = f * settings.formant_scale
                formant = torchaudio.functional.equalizer_biquad(
                    formant,
                    sample_rate=sample_rate,
                    center_freq=center,
                    q=4.0,
                    gain=3.0 if settings.formant_scale > 1 else -3.0,
                )

        reverb_input = formant
        if reverb_ir is not None:
            reverb_input = torchaudio.functional.fftconvolve(
                formant, reverb_ir.to(device).unsqueeze(0)
            )

    return reverb_input


class StreamingVoiceChanger:
    def __init__(
        self,
        model: nn.Module,
        settings: EffectSettings,
        sample_rate: int = 48000,
        block_size: int = 1024,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.model.eval()
        self.settings = settings
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)

        self.input_queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self.output_queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self.reverb_ir = build_reverb_ir(sample_rate, settings.reverb_seconds, settings.reverb_decay).to(
            self.device
        )
        self.running = False

    def _process_block(self, block: np.ndarray) -> np.ndarray:
        audio = torch.from_numpy(block).to(self.device).transpose(0, 1)
        with torch.no_grad():
            stylized = self.model(audio)
            effected = apply_effects(
                stylized, self.sample_rate, self.settings, self.device, reverb_ir=self.reverb_ir
            )
        processed = effected.squeeze(0).cpu().numpy().T
        return processed

    def _processing_loop(self) -> None:
        while self.running:
            try:
                block = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            processed = self._process_block(block)
            self.output_queue.put(processed)

    def _audio_callback(self, indata, outdata, frames, time, status):
        if status:
            print(status)
        self.input_queue.put(indata.copy())
        try:
            out_block = self.output_queue.get_nowait()
        except queue.Empty:
            out_block = np.zeros_like(indata)
        outdata[:] = out_block

    def run(self):
        self.running = True
        processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        processing_thread.start()

        with sd.Stream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
        ):
            print("Streaming... press Ctrl+C to stop.")
            try:
                while True:
                    sd.sleep(1000)
            except KeyboardInterrupt:
                print("Stopping...")
                self.running = False
                processing_thread.join()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Anime voice changer with PyTorch.")
    parser.add_argument("--pitch", type=float, default=4.0, help="Pitch shift in semitones.")
    parser.add_argument(
        "--formant", type=float, default=1.2, help="Formant scale multiplier (higher = brighter)."
    )
    parser.add_argument(
        "--reverb-seconds", type=float, default=0.35, help="Reverb tail length in seconds."
    )
    parser.add_argument(
        "--reverb-decay", type=float, default=0.5, help="Reverb decay multiplier (higher = faster decay)."
    )
    parser.add_argument("--block-size", type=int, default=1024, help="Audio block size for streaming.")
    parser.add_argument("--sample-rate", type=int, default=48000, help="Sampling rate for microphone.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device specifier (defaults to CUDA if available, else CPU).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    settings = EffectSettings(
        pitch_steps=args.pitch,
        formant_scale=args.formant,
        reverb_seconds=args.reverb_seconds,
        reverb_decay=args.reverb_decay,
    )

    model = AnimeVoiceNet(channels=1)
    engine = StreamingVoiceChanger(
        model,
        settings=settings,
        sample_rate=args.sample_rate,
        block_size=args.block_size,
        device=args.device,
    )
    engine.run()


if __name__ == "__main__":
    main()
