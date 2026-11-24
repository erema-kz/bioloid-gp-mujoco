"""Export a STEP assembly to a tessellated STL with FreeCAD's Python."""

from __future__ import annotations

import argparse
from pathlib import Path

import FreeCAD  # type: ignore
import MeshPart  # type: ignore
import Part  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--linear-deflection-mm", type=float, default=0.25)
    parser.add_argument("--angular-deflection-rad", type=float, default=0.35)
    args = parser.parse_args()

    shape = Part.read(str(args.source.resolve()))
    if shape.isNull() or not shape.Solids:
        raise ValueError(f"STEP assembly has no solids: {args.source}")

    mesh = MeshPart.meshFromShape(
        Shape=shape,
        LinearDeflection=args.linear_deflection_mm,
        AngularDeflection=args.angular_deflection_rad,
        Relative=False,
    )
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    mesh.write(str(args.destination.resolve()))

    bounds = shape.BoundBox
    print(
        f"Exported {len(shape.Solids)} solids and {mesh.CountFacets} triangles; "
        f"bounds={bounds.XLength:.1f} x {bounds.YLength:.1f} x {bounds.ZLength:.1f} mm"
    )


if __name__ == "__main__":
    main()
