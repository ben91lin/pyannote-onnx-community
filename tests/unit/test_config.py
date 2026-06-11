def test_sd_config_defaults():
    from pyannote_onnx_community.config import SDConfig

    c = SDConfig()
    assert c.sample_rate == 16000
    assert c.clustering_threshold == 0.7
    assert c.vbx_fa == 0.07
    assert c.vbx_fb == 0.8
    assert c.plda_repo_id == "pyannote/speaker-diarization-community-1"
    assert c.embedding_exclude_overlap is True
    # Reconstruction min-duration filters — community-1 yaml defaults
    # (segmentation.min_duration_off=0.0, to_annotation min_duration_on=0.0).
    assert c.min_duration_on == 0.0
    assert c.min_duration_off == 0.0


def test_vad_config_defaults():
    from pyannote_onnx_community.config import VADConfig

    c = VADConfig()
    assert c.sample_rate == 16000
    assert c.onset == 0.5
    assert c.offset == 0.363
    assert c.min_duration_on == 2.0
    assert c.min_duration_off == 1.5
