#!/usr/bin/env python3
"""
finetune_geolocation.py -- Geolocation contrastive fine-tuning script.

Implements:
  - contrastive_loss(emb1, emb2) : NT-Xent / InfoNCE style loss for same-location pairs
  - gps_bin(lat, lon, bin_size)  : discretise GPS into integer grid cells
  - GeoContrastiveDataset         : torch Dataset wrapper around a manifest
  - train_one_epoch()             : standard training loop step
  - main()                        : CLI fine-tuning entry point
"""

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Tuple

# ---------------------------------------------------------------------------
# GPS discretisation
# ---------------------------------------------------------------------------

def gps_bin(lat: float, lon: float, bin_size: float) -> Tuple[int, int]:
    """Map a GPS coordinate to an integer grid cell.

    Parameters
    ----------
    lat, lon  : float - latitude / longitude in decimal degrees
    bin_size  : float - cell edge length in degrees (e.g. 5.0 -> 5x5 degree cells)

    Returns
    -------
    (lat_bin, lon_bin) : Tuple[int, int]
        Signed integer bin indices computed via floor(coord / bin_size).

    Examples
    --------
    >>> gps_bin(48.8, 2.2, 5.0)
    (9, 0)
    >>> gps_bin(-33.8, 151.2, 5.0)
    (-7, 30)
    """
    return (int(math.floor(lat / bin_size)), int(math.floor(lon / bin_size)))


# ---------------------------------------------------------------------------
# Contrastive loss (NT-Xent / InfoNCE)
# ---------------------------------------------------------------------------

def contrastive_loss(emb1, emb2, temperature: float = 0.07):
    """Compute NT-Xent contrastive loss between two batches of embeddings.

    The two embedding tensors are treated as positive pairs: emb1[i] and
    emb2[i] represent two augmented views of the *same* location, while all
    cross-sample pairs are negatives.

    Parameters
    ----------
    emb1 : torch.Tensor - shape (N, D)
    emb2 : torch.Tensor - shape (N, D)
    temperature : float - InfoNCE temperature tau (default 0.07)

    Returns
    -------
    torch.Tensor - scalar loss value

    Raises
    ------
    ValueError
        If emb1 and emb2 have different batch sizes OR different feature dimensions.
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError("torch is required for contrastive_loss") from exc

    if emb1.shape[0] != emb2.shape[0]:
        raise ValueError(
            f"Batch size mismatch: emb1 has {emb1.shape[0]} rows, "
            f"emb2 has {emb2.shape[0]} rows."
        )
    if emb1.shape[1:] != emb2.shape[1:]:
        raise ValueError(
            f"Feature dimension mismatch: emb1 shape {tuple(emb1.shape)}, "
            f"emb2 shape {tuple(emb2.shape)}."
        )

    N = emb1.shape[0]

    # L2-normalise
    z1 = F.normalize(emb1.float(), dim=-1)
    z2 = F.normalize(emb2.float(), dim=-1)

    # Concatenate: [2N, D]
    z = torch.cat([z1, z2], dim=0)

    # Similarity matrix [2N, 2N] / temperature
    sim = torch.mm(z, z.t()) / temperature

    # Mask out self-similarities
    mask = torch.eye(2 * N, dtype=torch.bool, device=sim.device)
    sim = sim.masked_fill(mask, float("-inf"))

    # Positive indices: z1[i] pairs with z2[i] (offset by N), and vice-versa
    labels = torch.cat([torch.arange(N, 2 * N), torch.arange(N)]).to(sim.device)

    loss = F.cross_entropy(sim, labels)
    return loss


# ---------------------------------------------------------------------------
# Dataset (optional; only materialises when torch + PIL are available)
# ---------------------------------------------------------------------------

class GeoContrastiveDataset:
    """Minimal torch Dataset wrapping a SPADE-format manifest for fine-tuning."""

    def __init__(self, manifest_path: str, bin_size: float = 5.0):
        import json

        manifest_path = os.path.abspath(manifest_path)
        with open(manifest_path) as fh:
            manifest = json.load(fh)

        self.manifest_dir = os.path.dirname(manifest_path)
        self.samples = manifest.get("samples", [])
        self.bin_size = bin_size

        try:
            import torch  # noqa: F401
            import torchvision.transforms as T  # noqa: F401
            self._torch_ok = True
        except ImportError:
            self._torch_ok = False

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if not self._torch_ok:
            raise RuntimeError("torch/torchvision not installed; cannot iterate dataset.")

        import torch
        import torchvision.transforms as T
        from PIL import Image

        s = self.samples[idx]
        ip = s["image_path"]
        abs_path = ip if os.path.isabs(ip) else os.path.join(self.manifest_dir, ip)

        aug = T.Compose([
            T.Resize(256),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        img = Image.open(abs_path).convert("RGB")
        view1 = aug(img)
        view2 = aug(img)

        lat_bin, lon_bin = gps_bin(float(s["latitude"]), float(s["longitude"]), self.bin_size)
        return view1, view2, torch.tensor([lat_bin, lon_bin], dtype=torch.long)


# ---------------------------------------------------------------------------
# Training loop skeleton
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimiser, device="cpu", temperature=0.07):
    """Run one training epoch; returns average loss."""
    try:
        import torch
    except ImportError:
        raise

    model.train()
    total_loss = 0.0
    n_batches = 0

    for view1, view2, _ in loader:
        view1, view2 = view1.to(device), view2.to(device)
        emb1 = model(view1)
        emb2 = model(view2)
        loss = contrastive_loss(emb1, emb2, temperature=temperature)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches if n_batches else 0.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    parser = argparse.ArgumentParser(description="Geolocation contrastive fine-tuning.")
    parser.add_argument("--manifest", required=True, help="SPADE manifest JSON.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--bin-size", type=float, default=5.0,
                        help="GPS bin edge length in degrees.")
    parser.add_argument("--out", default="models/finetune_geo/",
                        help="Checkpoint output directory.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate setup but skip training loop.")
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
    except ImportError:
        print("ERROR: torch not installed. Install pytorch to run fine-tuning.", file=sys.stderr)
        sys.exit(1)

    dataset = GeoContrastiveDataset(args.manifest, bin_size=args.bin_size)
    print(f"[finetune] Dataset: {len(dataset)} samples, bin_size={args.bin_size} degrees")

    if args.dry_run:
        print("[finetune] --dry-run: skipping training loop. Setup OK.")
        return

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Minimal backbone stub (replace with a real ViT / ResNet encoder)
    model = nn.Sequential(
        nn.AdaptiveAvgPool2d((7, 7)),
        nn.Flatten(),
        nn.Linear(3 * 49, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.out, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, opt, device=device,
                               temperature=args.temperature)
        print(f"  Epoch {epoch:>4d}/{args.epochs}  loss={loss:.4f}")
        ckpt = os.path.join(args.out, f"geo_finetune_ep{epoch:04d}.pt")
        torch.save(model.state_dict(), ckpt)
        print(f"             saved -> {ckpt}")


if __name__ == "__main__":
    _cli()
