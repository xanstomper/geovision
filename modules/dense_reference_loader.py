#!/usr/bin/env python3
import gzip, json, os
from pathlib import Path
class DenseReferenceLoader:
    """Loads dense reference data for agent pipeline."""
    def load(self):
        refs = []
        path = Path(__file__).parent.parent / "data" / "dense_references_index.json"
        if path.exists():
            return json.loads(path.read_text())
        with gzip.open(Path(__file__).parent.parent / "data" / "dense_references.jsonl.gz", "rt") as f:
            for line in f:
                refs.append(json.loads(line))
        return {"count": len(refs), "loaded": True}
