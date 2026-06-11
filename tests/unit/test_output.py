import numpy as np
from pyannote.core import Annotation, Segment


def _annotation(spans):
    ann = Annotation()
    for start, end, label in spans:
        ann[Segment(start, end)] = label
    return ann


def test_diarize_output_fields():
    from pyannote_onnx_community.output import DiarizeOutput

    sd = _annotation([(0.0, 1.0, "SPEAKER_00")])
    ex = _annotation([(0.0, 1.0, "SPEAKER_00")])
    emb = np.zeros((1, 256), dtype=np.float32)
    out = DiarizeOutput(
        speaker_diarization=sd,
        exclusive_speaker_diarization=ex,
        speaker_embeddings=emb,
    )
    assert out.speaker_diarization is sd
    assert out.exclusive_speaker_diarization is ex
    assert out.speaker_embeddings.shape == (1, 256)


def test_diarize_output_speaker_embeddings_optional():
    from pyannote_onnx_community.output import DiarizeOutput

    out = DiarizeOutput(
        speaker_diarization=Annotation(),
        exclusive_speaker_diarization=Annotation(),
    )
    assert out.speaker_embeddings is None


def test_serialize_structure_and_rounding():
    from pyannote_onnx_community.output import DiarizeOutput

    sd = _annotation([(6.66449, 7.16551, "SPEAKER_00"), (8.0, 9.0, "SPEAKER_01")])
    ex = _annotation([(6.66449, 7.16551, "SPEAKER_00")])
    out = DiarizeOutput(speaker_diarization=sd, exclusive_speaker_diarization=ex)

    payload = out.serialize()

    assert set(payload) == {"diarization", "exclusive_diarization"}
    assert payload["diarization"] == [
        {"start": 6.664, "end": 7.166, "speaker": "SPEAKER_00"},
        {"start": 8.0, "end": 9.0, "speaker": "SPEAKER_01"},
    ]
    assert payload["exclusive_diarization"] == [
        {"start": 6.664, "end": 7.166, "speaker": "SPEAKER_00"},
    ]
