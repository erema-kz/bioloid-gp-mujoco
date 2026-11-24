"""Recover rigid links from the exact Bioloid GP STEP assembly.

Run with FreeCAD's Python. Parts that touch form a rigid link, except for each
AX-12 body/horn pair: that interface becomes a revolute joint.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
import json
from pathlib import Path

import FreeCAD  # type: ignore
import Import  # type: ignore
import MeshPart  # type: ignore
import Part  # type: ignore


@dataclass
class Leaf:
    name: str
    label: str
    shape: object
    servo: str | None
    servo_part: str | None


def transformed_shape(obj, parent_placement):
    shape = obj.Shape.copy()
    shape.Placement = parent_placement.multiply(obj.Shape.Placement)
    return shape


def collect_leaves(obj, parent_placement, servo=None, servo_part=None):
    placement = parent_placement.multiply(obj.Placement)
    if obj.Name.startswith("Full_AX"):
        servo = obj.Name
    children = [child for child in obj.OutList if hasattr(child, "Shape")]
    solid_children = [child for child in children if not child.Shape.isNull()]
    if obj.Shape.ShapeType == "Solid" and not solid_children:
        part_kind = servo_part
        if servo is not None:
            part_kind = "horn" if "horn" in obj.Label else "body"
        shape = obj.Shape.copy()
        # The leaf's Shape already contains its own Placement. Only ancestor
        # placements need to be prepended here.
        shape.Placement = parent_placement.multiply(obj.Shape.Placement)
        return [Leaf(obj.Name, obj.Label, shape, servo, part_kind)]

    leaves = []
    for child in children:
        leaves.extend(collect_leaves(child, placement, servo, servo_part))
    return leaves


def aabb_gap(a, b):
    return max(
        0.0,
        a.XMin - b.XMax,
        b.XMin - a.XMax,
        a.YMin - b.YMax,
        b.YMin - a.YMax,
        a.ZMin - b.ZMax,
        b.ZMin - a.ZMax,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tolerance-mm", type=float, default=0.25)
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()

    Import.open(str(args.source.resolve()))
    document = FreeCAD.ActiveDocument
    top = document.getObject("BioloidGP")
    if top is None:
        raise ValueError("The STEP file has no BioloidGP assembly")

    leaves = collect_leaves(top, FreeCAD.Placement())
    graph = defaultdict(set)
    contact_edges = []
    for i, first in enumerate(leaves):
        graph[i]
        for j in range(i + 1, len(leaves)):
            second = leaves[j]
            same_servo_pair = (
                first.servo is not None
                and first.servo == second.servo
                and {first.servo_part, second.servo_part} == {"body", "horn"}
            )
            if same_servo_pair or aabb_gap(first.shape.BoundBox, second.shape.BoundBox) > args.tolerance_mm:
                continue
            distance = first.shape.distToShape(second.shape)[0]
            if distance <= args.tolerance_mm:
                graph[i].add(j)
                graph[j].add(i)
                contact_edges.append((i, j, distance))

    components = []
    unseen = set(range(len(leaves)))
    while unseen:
        start = unseen.pop()
        component = {start}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in graph[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        components.append(sorted(component))
    components.sort(key=lambda indices: (-len(indices), indices[0]))

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"source": str(args.source), "tolerance_mm": args.tolerance_mm, "links": []}
    for number, indices in enumerate(components):
        shapes = [leaves[index].shape for index in indices]
        compound = Part.makeCompound(shapes)
        link_name = f"link_{number:02d}"
        entry = {
            "name": link_name,
            "parts": [
                {
                    "name": leaves[index].name,
                    "label": leaves[index].label,
                    "servo": leaves[index].servo,
                    "servo_part": leaves[index].servo_part,
                }
                for index in indices
            ],
            "bounds_mm": {
                "min": [compound.BoundBox.XMin, compound.BoundBox.YMin, compound.BoundBox.ZMin],
                "max": [compound.BoundBox.XMax, compound.BoundBox.YMax, compound.BoundBox.ZMax],
            },
        }
        manifest["links"].append(entry)
        print(link_name, len(indices), ", ".join(leaves[index].label for index in indices))
        if args.export:
            mesh = MeshPart.meshFromShape(
                Shape=compound,
                LinearDeflection=0.25,
                AngularDeflection=0.35,
                Relative=False,
            )
            mesh.write(str((args.output / f"{link_name}.stl").resolve()))

    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Recovered {len(components)} components from {len(leaves)} solids")


if __name__ == "__main__":
    main()
