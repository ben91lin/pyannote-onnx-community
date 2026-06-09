"""Vendored from pyannote-audio 4.0.4 (BSD-3-Clause).

Source: https://github.com/pyannote/pyannote-audio/blob/4.0.4/src/pyannote/audio/core/plda.py
Original copyright: 2024-2025 CNRS / pyannoteAI (MIT in upstream).
Modifications:
  - file header
  - replace ``pyannote.audio.utils.hf_hub.download_from_hf_hub`` with
    ``huggingface_hub.snapshot_download`` (offline-friendly snapshot resolve
    for ONNX repos)
  - import ``vbx_setup`` from the sibling ``_vbx`` module
"""

# MIT License
#
# Copyright (c) 2024-2025 CNRS
# Copyright (c) 2025- pyannoteAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

import os
from pathlib import Path
from typing import Optional

import numpy as np
from huggingface_hub import snapshot_download

from pyannote_onnx_community._vbx import vbx_setup


class PLDA:
    """PLDA"""

    def __init__(self, transform_npz: str | Path, plda_npz: str | Path, lda_dimension: int = 128):
        self._xvec_tf, self._plda_tf, self._plda_psi = vbx_setup(transform_npz, plda_npz)

        self.lda_dimension = lda_dimension

    @property
    def phi(self):
        """Between-class covariance in the PLDA space."""
        return self._plda_psi[: self.lda_dimension]

    def __call__(self, embeddings: np.ndarray):
        """

        Parameters
        ----------
        embeddings : (num_embeddings, embedding_dimension) ndarray
            Embeddings to be transformed into the PLDA space.

        Returns
        -------
        fea : (num_embeddings, lda_dimension) ndarray
            Embeddings transformed into the PLDA space.
        """
        return self._plda_tf(self._xvec_tf(embeddings), lda_dim=self.lda_dimension)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: Path | str,
        subfolder: str = "plda",
        revision: str | None = None,
        token: str | None = None,
        cache_dir: str | Path | None = None,
        **kwargs,
    ) -> Optional["PLDA"]:
        """Load PLDA from disk or Huggingface Hub.

        When ``checkpoint`` is a directory, look for ``xvec_transform.npz`` and
        ``plda.npz`` inside ``checkpoint/subfolder``. Otherwise treat
        ``checkpoint`` as a HF repo id and call
        ``huggingface_hub.snapshot_download`` (offline-friendly: when the
        bundle has been pre-staged into HF cache, this is a no-op resolve).
        """

        # Local checkpoint directory
        if os.path.isdir(checkpoint):
            if revision is not None:
                raise ValueError("Revisions cannot be used with local checkpoints.")
            path_to_transform = Path(checkpoint) / subfolder / "xvec_transform.npz"
            path_to_plda = Path(checkpoint) / subfolder / "plda.npz"
            return cls(path_to_transform, path_to_plda)

        # HF hub repo id
        checkpoint = str(checkpoint)
        if "@" in checkpoint:
            raise ValueError("Revisions must be passed with `revision` keyword argument.")

        # snapshot_download is offline-safe when the bundle is staged: it
        # resolves to the cached snapshot without hitting the network.
        snapshot_dir = snapshot_download(
            repo_id=checkpoint,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            allow_patterns=[f"{subfolder}/*.npz"],
        )

        path_to_transform = Path(snapshot_dir) / subfolder / "xvec_transform.npz"
        path_to_plda = Path(snapshot_dir) / subfolder / "plda.npz"

        if not path_to_transform.is_file() or not path_to_plda.is_file():
            return None

        return cls(path_to_transform, path_to_plda)
