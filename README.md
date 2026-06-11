# pyannote-onnx-community

Pure-ONNX pyannote **community-1** speaker diarization (VBx + PLDA). Torch-free
inference, validated array-for-array against the official PyTorch pipeline. The
runtime needs only onnxruntime + a Kaldi fbank + scipy + pyannote.core + PyAV —
no `torch`, no `pyannote.audio` — yet every intermediate array (segmentation
probabilities, Kaldi fbank, PLDA projection, VBx clustering, L2 norm) matches
the upstream pyannote.audio PyTorch pipeline to cosine ~1.0.

## Install

```bash
pip install pyannote-onnx-community
```

The runtime is **torch-free**: onnxruntime + kaldi-native-fbank + scipy +
pyannote.core + PyAV. Models auto-download from the HuggingFace Hub on first
use:

- segmentation: `onnx-community/pyannote-segmentation-3.0`
- embedding: `onnx-community/wespeaker-voxceleb-resnet34-LM`
- community-1 PLDA artifacts: `pyannote/speaker-diarization-community-1` (gated —
  accept the conditions on the Hub and authenticate if downloading for the first
  time)

## Usage

Three public classes, each callable on a path/file object (decoded, resampled
and normalised via PyAV) or a waveform array. **A waveform array must already be
float32, normalised to `[-1, 1]`, at the target sample rate (16 kHz by
default)** — it is trusted as-is and never resampled or PCM-normalised. Pass it
mono 1-D, or 2-D `(channels, samples)` which is downmixed to mono. Integer
PCM (e.g. `scipy.io.wavfile` int16) or un-normalised arrays raise `ValueError`;
convert first (`arr.astype(np.float32) / 32768.0`) or pass a path to decode.

```python
from pyannote_onnx_community import ONNXSpeakerDiarization

dia = ONNXSpeakerDiarization(providers=["CPUExecutionProvider"])  # or CUDA / CoreML
out = dia("audio.wav")                         # -> DiarizeOutput (mirrors pyannote.audio)

# Overlap-aware diarization (pyannote.core.Annotation):
for turn, _, spk in out.speaker_diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}-{turn.end:.1f} {spk}")     # e.g. 0.5-3.2 SPEAKER_00

out.exclusive_speaker_diarization              # Annotation, no overlap (for ASR/transcription)
out.speaker_embeddings                         # (num_speakers, 256), in labels() order
out.serialize()                                # JSON-friendly dict

# Pin / bound the speaker count, and pass a progress hook:
out = dia("audio.wav", num_speakers=2)                          # exact count
out = dia("audio.wav", min_speakers=2, max_speakers=5)          # bounds
out = dia("audio.wav", hook=lambda step, x=None: print(step))   # segmentation/embeddings/...

from pyannote_onnx_community import ONNXVoiceActivityDetection

vad = ONNXVoiceActivityDetection()
speech = vad("audio.wav")                     # -> Annotation of SPEECH regions

from pyannote_onnx_community import ONNXSpeakerEmbedding

emb = ONNXSpeakerEmbedding()
vec = emb("audio.wav")                         # -> (256,) L2-normalised np.float32
```

- `providers` (all three classes) selects the ONNX Runtime execution provider —
  `["CPUExecutionProvider"]` (default), `["CUDAExecutionProvider"]`,
  `["CoreMLExecutionProvider"]`, etc.
- `ONNXSpeakerDiarization.__call__` returns a `DiarizeOutput` (overlap-aware
  `speaker_diarization`, non-overlapping `exclusive_speaker_diarization`,
  per-speaker `speaker_embeddings`, and `serialize()`) — mirroring upstream
  pyannote.audio. Need a bare `Annotation`? Use `out.speaker_diarization`.
- `num_speakers` / `min_speakers` / `max_speakers` constrain clustering: when the
  VBx auto-count falls outside the bounds (or `num_speakers` is pinned), the
  speaker count is forced via a KMeans re-cluster. `min`/`max` are ignored when
  `num_speakers` is given. Invalid values (`<= 0`, or `min_speakers >
  max_speakers`) raise `ValueError` rather than being silently clamped.

## Validation — per-stage array parity vs official pyannote.audio PyTorch

The reference is the official **pyannote.audio 4.0.4** PyTorch community-1
pipeline. Goldens are committed under `tests/goldens/`, and the parity suite runs
**torch-free** against those committed arrays.

| Stage | Metric | Result vs upstream PyTorch |
|-------|--------|----------------------------|
| `l2_norm` | max abs diff | 0.0 |
| `cluster_vbx` (gamma, pi) | max abs diff | 0.0 |
| PLDA projection | max abs diff | 0.0 |
| Kaldi fbank | cosine | 1.0000001 (max abs diff 2.5e-4) |
| segmentation probs | cosine | 1.0000000 (max abs diff 2.4e-7) |

Reproduce:

```bash
pytest tests/parity                  # torch-free, uses the committed goldens
python scripts/make_goldens.py       # regenerate goldens (in a dev venv with torch)
```

End-to-end: the output stage mirrors upstream's count-based reconstruction, so
on a 60s clip DER (ours vs upstream community-1, NIST-standard 0.25s collar) =
**0.149**, with the speaker count matching (2 vs 2). See
`tests/e2e/test_der_parity.py`.

## Speed

**Core advantage: fast on plain CPU, with zero `torch` / GPU / accelerator
dependency.** The benchmark below was measured on the current release benchmark
machine, an Apple M4 Pro Mac, using one 10s warmup followed by three measured
full-audio runs per backend; tables report the median. Full environment, raw
runs, private-input notes, and methodology are in `docs/benchmark_results.md`.

| clip | audio s | engine | device | median wall s | median RTFx | speakers |
|------|--------:|--------|--------|--------------:|------------:|---------:|
| clip.wav | 60.0 | ours | cpu | 4.1 | 14.73 | 2 |
| clip.wav | 60.0 | ours | coreml | 1.6 | 38.30 | 2 |
| clip.wav | 60.0 | pytorch | cpu | 36.8 | 1.63 | 2 |
| clip.wav | 60.0 | pytorch | mps | 2.7 | 22.14 | 2 |
| long-clip | 600.0 | ours | cpu | 53.4 | 11.24 | 2 |
| long-clip | 600.0 | ours | coreml | 17.8 | 33.67 | 2 |
| long-clip | 600.0 | pytorch | cpu | 434.0 | 1.38 | 2 |
| long-clip | 600.0 | pytorch | mps | 28.3 | 21.21 | 2 |

`long-clip` is a local/private 10-minute benchmark input. It is not committed or
distributed with this repository.

The portable headline is the ONNX CPU row: it needs no CUDA, no MPS, no CoreML,
and no `torch`, yet it processes the 10-minute private clip in 53.4s versus
434.0s for the official PyTorch CPU pipeline on the same Mac. Accelerator rows
map the Apple Silicon landscape. CoreML brings the ONNX path to 17.8s on the
10-minute clip; where the multi-GB PyTorch dependency and Apple GPU acceleration
are both acceptable, PyTorch-MPS reaches 28.3s. This repo's main win is the
deployment surface while staying fast on CPU.

## Known limitations

- **Long-form speaker under-count.** On long-form audio the pipeline can
  under-count speakers vs the full PyTorch community-1 pipeline. In a separate
  internal long-form evaluation input, a 10-min clip produced 5 vs 6 speakers;
  longer probes showed larger gaps. This is not the `long-clip` benchmark input
  above, where both pipelines report 2 speakers. The per-stage
  arrays match upstream exactly, so this is an **assembly /
  clustering-sensitivity** effect of the AHC-seed `clustering_threshold`
  (default `0.7`, tunable in `SDConfig`) — it is not a stage bug. If exact
  long-form speaker-count parity matters, this threshold needs per-corpus
  tuning, or pin the count explicitly with `num_speakers` / `min_speakers` /
  `max_speakers`.
- **`num_speakers` force-count is not array-exact vs upstream.** When the VBx
  auto-count is out of bounds the count is forced via a KMeans re-cluster using
  SciPy's `kmeans2` (sklearn-free), so the forced assignment can differ from
  upstream's `sklearn.KMeans`. The auto path (no constraints) is unaffected.

## Development

Install the dev extras in a **separate venv** — they pull
`pyannote.audio==4.0.4` + `torch`, which are needed only for golden generation
and benchmarking, never at runtime:

```bash
pip install -e '.[dev]'                # SEPARATE venv (pyannote.audio==4.0.4 + torch)
pytest tests/unit tests/parity         # torch-free
python scripts/make_goldens.py         # regenerate per-stage goldens (dev venv)
pytest tests/e2e                       # DER parity (dev venv, needs HF token for gated community-1)
python scripts/benchmark.py <audio>    # speed table (dev venv)
```

## Related projects

- [pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio) — the
  official upstream this project ports and validates against (the community-1
  pipeline).
- [samson6460/pyannote-onnx-extended](https://github.com/samson6460/pyannote-onnx-extended)
  — a sibling torch-free ONNX port that targets the older pyannote 3.1 / AHC
  pipeline with a librosa float-domain fbank. This project targets community-1
  (VBx + PLDA) with a Kaldi fbank (the front-end wespeaker was trained on),
  which keeps embeddings at cosine ~1.0 vs the upstream PyTorch pipeline. Both
  are valid ports of different pipelines.

## License + attribution

- MIT — see [LICENSE](LICENSE).
- `pyannote_onnx_community/_vbx.py` is vendored from pyannote-audio 4.0.4
  (Apache-2.0); `pyannote_onnx_community/_plda.py` is vendored from
  pyannote-audio 4.0.4 (MIT). The original attribution headers are preserved
  in those files.
- The segmentation and embedding ONNX models are from the `onnx-community` org
  on HuggingFace (`onnx-community/pyannote-segmentation-3.0`,
  `onnx-community/wespeaker-voxceleb-resnet34-LM`); the community-1 PLDA
  artifacts are from `pyannote/speaker-diarization-community-1`.
