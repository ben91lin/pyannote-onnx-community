# Speed: ONNX (this repo) vs pyannote.audio PyTorch

RTFx = audio seconds / wall-clock seconds (higher is faster).
Each successful cell reports the median of 3 timed runs after one 10.0s warmup.

## Current Mac Benchmark Environment

| field | value |
|---|---|
| benchmark date | 2026-06-10 |
| platform | macOS-15.4.1-arm64-arm-64bit-Mach-O |
| machine | arm64 |
| chip | Apple M4 Pro |
| python | 3.13.3 |
| onnxruntime | 1.26.0 |
| onnxruntime providers | CoreMLExecutionProvider, AzureExecutionProvider, CPUExecutionProvider |
| torch | 2.12.0 |
| torch MPS | available |
| pyannote.audio | 4.0.4 |

## Inputs

| clip | audio s | source | path |
|---|---:|---|---|
| clip.wav | 60.0 | public/committed | tests/goldens/clip.wav |
| long-clip | 600.0 | local/private | local/private |

Local/private clips are used only as benchmark inputs and are not distributed
with this repository.

## Median Results

| clip | audio s | engine | device | status | median wall s | median RTFx | speakers | providers |
|---|---:|---|---|---|---:|---:|---:|---|
| clip.wav | 60.0 | ours | cpu | ok | 4.1 | 14.73 | 2 | CPU |
| long-clip | 600.0 | ours | cpu | ok | 53.4 | 11.24 | 2 | CPU |
| clip.wav | 60.0 | ours | coreml | ok | 1.6 | 38.30 | 2 | CoreML, CPU |
| long-clip | 600.0 | ours | coreml | ok | 17.8 | 33.67 | 2 | CoreML, CPU |
| clip.wav | 60.0 | pytorch | cpu | ok | 36.8 | 1.63 | 2 | cpu |
| long-clip | 600.0 | pytorch | cpu | ok | 434.0 | 1.38 | 2 | cpu |
| clip.wav | 60.0 | pytorch | mps | ok | 2.7 | 22.14 | 2 | mps |
| long-clip | 600.0 | pytorch | mps | ok | 28.3 | 21.21 | 2 | mps |

## Raw Runs

| clip | engine | device | status | wall s | reason |
|---|---|---|---|---|---|
| clip.wav | ours | cpu | ok | 4.3, 4.0, 4.1 |  |
| long-clip | ours | cpu | ok | 49.8, 55.3, 53.4 |  |
| clip.wav | ours | coreml | ok | 1.8, 1.6, 1.6 |  |
| long-clip | ours | coreml | ok | 17.8, 17.6, 18.0 |  |
| clip.wav | pytorch | cpu | ok | 37.5, 36.8, 36.5 |  |
| long-clip | pytorch | cpu | ok | 434.0, 434.1, 432.7 |  |
| clip.wav | pytorch | mps | ok | 3.2, 2.7, 2.4 |  |
| long-clip | pytorch | mps | ok | 26.0, 28.3, 28.4 |  |

## Notes

- `ours` is this repo's torch-free ONNX Runtime diarization pipeline.
- `pytorch` is the official `pyannote/speaker-diarization-community-1` pipeline.
- CoreML runs with `CoreMLExecutionProvider, CPUExecutionProvider`; ONNX Runtime may fall back to CPU for unsupported nodes.
- MPS runs with `PYTORCH_ENABLE_MPS_FALLBACK=1` so unsupported PyTorch ops can fall back to CPU.
