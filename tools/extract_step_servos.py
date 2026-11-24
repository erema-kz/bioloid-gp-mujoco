"""Extract AX-12 pivot frames directly from the exact STEP assembly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import FreeCAD  # type: ignore
import Import  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    Import.open(str(args.source.resolve()))
    document = FreeCAD.ActiveDocument
    top = document.getObject("BioloidGP")
    if top is None:
        raise ValueError("The STEP file has no BioloidGP assembly")

    servos = []
    for assembly in top.OutList:
        if not assembly.Name.startswith("Full_AX"):
            continue
        horn = next(child for child in assembly.OutList if "horn" in child.Label)
        frame = assembly.Placement.multiply(horn.Placement)
        axis = frame.Rotation.multVec(FreeCAD.Vector(0, 0, 1))
        servos.append(
            {
                "name": assembly.Name,
                "label": assembly.Label,
                "pivot_mm": [frame.Base.x, frame.Base.y, frame.Base.z],
                "axis": [axis.x, axis.y, axis.z],
                "quaternion_xyzw": [
                    frame.Rotation.Q[0],
                    frame.Rotation.Q[1],
                    frame.Rotation.Q[2],
                    frame.Rotation.Q[3],
                ],
            }
        )

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(json.dumps({"servos": servos}, indent=2), encoding="utf-8")
    print(f"Extracted {len(servos)} servo frames")
    for servo in servos:
        print(servo["name"], servo["pivot_mm"], servo["axis"])


if __name__ == "__main__":
    main()
