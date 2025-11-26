# anime-voice-changer-ai

Apply an anime-style voice model to microphone input with pitch, formant, and reverb controls. The sample includes a lightweight PyTorch network, real-time streaming via `sounddevice`, and DSP helpers powered by `torchaudio`.

## Requirements

Install the audio and ML dependencies:

```bash
pip install torch torchaudio sounddevice numpy
```

## Running the stream

Run the voice changer with default controls (pitch +4 semitones, brighter formants, and a short reverb):

```bash
python app.py
```

### Common options

- `--pitch`: semitone shift applied before formant shaping. Example: `--pitch 7` for a fifth.
- `--formant`: multiplier for formant centers; values >1 brighten the tone, values <1 darken it. Example: `--formant 0.9`.
- `--reverb-seconds`: tail duration for the generated impulse response.
- `--reverb-decay`: how quickly the reverb decays (higher values fade faster).
- `--sample-rate`: override the microphone rate if needed (default 48000 Hz).
- `--block-size`: adjust for latency vs. stability trade-offs.
- `--device`: set a PyTorch device like `cuda` or `cpu`.

Example for a softer, darker tone with a longer space:

```bash
python app.py --pitch 2 --formant 0.9 --reverb-seconds 0.6 --reverb-decay 0.35
```
