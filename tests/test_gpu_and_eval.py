"""
Tests for SPADE-format eval runner + video sampler + keyed stages + audio...

Focus: verify the SPADE eval's pure math and path resolution without the heavy
CPU GeoCLIP loop; and the finetune script's correctness path (no GPU needed).
"""
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import importlib.util
import json


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_spade_haversine_known_distance():
    # Eiffel Tower to Arc de Triomphe ~1.7km
    m = _load("spade_eval", "scripts/spade_eval.py")
    d = m.haversine_km(48.8584, 2.2945, 48.8738, 2.2950)
    assert 1.5 < d < 2.0, f"expected ~1.7km, got {d}"


def test_spade_path_resolution_from_manifest():
    """Every sample's image path must resolve relative to the manifest dir."""
    MAN = "data/eval/wikipedia_landmarks_v1/manifest.json"
    assert os.path.exists(MAN)
    data = json.load(open(MAN))
    md = os.path.dirname(os.path.abspath(MAN))
    for s in data["samples"]:
        ip = s["image_path"]
        resolved = ip if os.path.isabs(ip) else os.path.join(md, ip)
        assert os.path.exists(resolved), f"not resolvable: {ip}"


def test_finetune_loss_pure():
    """The contrastive loss is a pure function; verify shape-validation + finite loss."""
    m = _load("finetune_geo", "scripts/finetune_geolocation.py")
    # mismatched shapes (batch or feature dim) must raise
    import torch
    try:
        m.contrastive_loss(torch.randn(3, 5), torch.randn(4, 256))
        raised = False
    except ValueError:
        raised = True
    assert raised
    # matched-shape forward is finite (embeddings should already be same latent dim)
    emb = torch.randn(4, 512)
    loss = m.contrastive_loss(emb, emb)
    assert torch.isfinite(loss) and loss.ndim == 0


def test_finetune_builds_geobins():
    m = _load("finetune_geo", "scripts/finetune_geolocation.py")
    assert m.gps_bin(48.8, 2.2, 5.0) == (9, 0)
    assert m.gps_bin(-33.8, 151.2, 5.0) == (-7, 30)