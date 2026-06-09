"""pyannote-onnx-community: torch-free ONNX pyannote community-1 diarization."""

from pyannote_onnx_community.diarization import ONNXSpeakerDiarization
from pyannote_onnx_community.embedding import ONNXSpeakerEmbedding
from pyannote_onnx_community.vad import ONNXVoiceActivityDetection

__version__ = "0.1.0"

__all__ = [
    "ONNXSpeakerDiarization",
    "ONNXSpeakerEmbedding",
    "ONNXVoiceActivityDetection",
]
