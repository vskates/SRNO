#!/usr/bin/env python3
"""Create a same-root manifest and active index for a gate ablation."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from srno.data.schema import DatasetManifest
from srno.data.tools import build_active_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--delta-gate-m", type=float, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-index", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    source = DatasetManifest.load(args.manifest)
    if args.output_manifest.resolve().parent != source.root:
        raise ValueError("variant manifest must stay beside the source manifest")
    variant = dataclasses.replace(source, delta_gate_m=args.delta_gate_m)
    variant.validate()
    variant.save(args.output_manifest)
    active = build_active_index(variant, args.output_index, device=args.device)
    print(
        json.dumps(
            {
                "manifest": str(args.output_manifest),
                "manifest_sha256": variant.sha256(),
                "delta_gate_m": variant.delta_gate_m,
                "active_index": str(args.output_index),
                "active_transitions": int(len(active.step_index)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
