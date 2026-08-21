#!/usr/bin/env python3
"""Audit authored USD mass/inertia against the SRNO dataset input contract.

Run with the Isaac Sim Python launcher because ``pxr`` is provided there, e.g.
``/path/to/isaacsim/python.sh scripts/audit_object_mass_contract.py ...``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from pxr import Usd, UsdPhysics


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _components(value: Any) -> list[float]:
    if hasattr(value, "GetReal") and hasattr(value, "GetImaginary"):
        imaginary = value.GetImaginary()
        return [float(value.GetReal()), *(float(component) for component in imaginary)]
    return [float(component) for component in value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text())
    manifest = json.loads(args.manifest.read_text())
    records = []
    for record in catalog["objects"]:
        stage = Usd.Stage.Open(str(args.catalog.parent / record["usd"]))
        authored = []
        for prim in stage.Traverse():
            api = UsdPhysics.MassAPI(prim)
            attributes = {
                "mass_kg": api.GetMassAttr(),
                "density": api.GetDensityAttr(),
                "center_of_mass": api.GetCenterOfMassAttr(),
                "diagonal_inertia": api.GetDiagonalInertiaAttr(),
                "principal_axes": api.GetPrincipalAxesAttr(),
            }
            if any(value.HasAuthoredValueOpinion() for value in attributes.values()):
                authored.append(
                    {
                        "prim": str(prim.GetPath()),
                        "mass_kg": float(attributes["mass_kg"].Get()),
                        "density": float(attributes["density"].Get()),
                        "center_of_mass": _components(attributes["center_of_mass"].Get()),
                        "diagonal_inertia": _components(
                            attributes["diagonal_inertia"].Get()
                        ),
                        "principal_axes": _components(attributes["principal_axes"].Get()),
                        "authored": {
                            name: value.HasAuthoredValueOpinion()
                            for name, value in attributes.items()
                        },
                    }
                )
        catalog_mass = float(record["mass_kg"])
        authored_masses = [value["mass_kg"] for value in authored]
        records.append(
            {
                "object_id": record["id"],
                "catalog_mass_kg": catalog_mass,
                "authored_mass_entries": authored,
                "has_authored_mass": bool(authored_masses),
                "catalog_mass_matches_authored": bool(authored_masses)
                and min(abs(value - catalog_mass) for value in authored_masses) < 1e-6,
            }
        )
    masses = [value["catalog_mass_kg"] for value in records]
    manifest_text = json.dumps(manifest).lower()
    result = {
        "catalog": str(args.catalog.resolve()),
        "manifest": str(args.manifest.resolve()),
        "object_count": len(records),
        "mass_range_kg": [min(masses), max(masses)],
        "all_objects_have_authored_mass": all(
            value["has_authored_mass"] for value in records
        ),
        "all_catalog_masses_match_authored": all(
            value["catalog_mass_matches_authored"] for value in records
        ),
        "manifest_contains_mass_or_inertia_field": (
            "mass" in manifest_text or "inertia" in manifest_text
        ),
        "records": records,
    }
    _write_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
