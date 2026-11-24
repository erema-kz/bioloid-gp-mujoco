"""Export the exact STEP assembly as 17 articulated rigid-link meshes.

The grouping follows the physical AX-12 body/horn mounting scheme visible in
the Autodesk assembly. No geometry from the reference ROS repositories is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import FreeCAD  # type: ignore
import Import  # type: ignore
import MeshPart  # type: ignore
import Part  # type: ignore

from segment_step_assembly import collect_leaves


LINKS = {
    "base": {
        "parent": None,
        "servo": None,
        "parts": [
            "Part__Feature", "Part__Feature001", "Part__Feature002", "Part__Feature004",
            "Part__Feature006", "Part__Feature007", "Part__Feature008", "Part__Feature009",
            "Part__Feature010", "Part__Feature011", "Part__Feature012", "Part__Feature027",
            "Part__Feature029",
        ],
    },
    "right_arm_1": {"parent": "base", "servo": "Full_AX_12", "parts": ["Part__Feature003", "Part__Feature016", "Part__Feature024"]},
    "right_arm_2": {"parent": "right_arm_1", "servo": "Full_AX_003", "parts": ["Part__Feature023", "Part__Feature018", "Part__Feature019", "Part__Feature031"]},
    "right_arm_3": {"parent": "right_arm_2", "servo": "Full_AX_007", "parts": ["Part__Feature032", "Part__Feature014"]},
    "left_arm_1": {"parent": "base", "servo": "Full_AX_001", "parts": ["Part__Feature005", "Part__Feature015", "Part__Feature022"]},
    "left_arm_2": {"parent": "left_arm_1", "servo": "Full_AX_002", "parts": ["Part__Feature021", "Part__Feature017", "Part__Feature020", "Part__Feature025"]},
    "left_arm_3": {"parent": "left_arm_2", "servo": "Full_AX_004", "parts": ["Part__Feature026", "Part__Feature013"]},
    "right_leg_1": {"parent": "base", "servo": "Full_AX_006", "parts": ["Part__Feature030", "Part__Feature038", "Part__Feature046"]},
    "right_leg_2": {"parent": "right_leg_1", "servo": "Full_AX_009", "parts": ["Part__Feature045", "Part__Feature040", "Part__Feature049"]},
    "right_leg_3": {"parent": "right_leg_2", "servo": "Full_AX_011", "parts": ["Part__Feature050", "Part__Feature036", "Part__Feature054"]},
    "right_leg_4": {"parent": "right_leg_3", "servo": "Full_AX_013", "parts": ["Part__Feature053", "Part__Feature041", "Part__Feature057"]},
    "right_leg_5": {"parent": "right_leg_4", "servo": "Full_AX_015", "parts": ["Part__Feature058", "Part__Feature060", "Part__Feature062", "Part__Feature034"]},
    "left_leg_1": {"parent": "base", "servo": "Full_AX_005", "parts": ["Part__Feature028", "Part__Feature037", "Part__Feature044"]},
    "left_leg_2": {"parent": "left_leg_1", "servo": "Full_AX_008", "parts": ["Part__Feature043", "Part__Feature039", "Part__Feature047"]},
    "left_leg_3": {"parent": "left_leg_2", "servo": "Full_AX_010", "parts": ["Part__Feature048", "Part__Feature035", "Part__Feature052"]},
    "left_leg_4": {"parent": "left_leg_3", "servo": "Full_AX_012", "parts": ["Part__Feature051", "Part__Feature042", "Part__Feature055"]},
    "left_leg_5": {"parent": "left_leg_4", "servo": "Full_AX_014", "parts": ["Part__Feature056", "Part__Feature059", "Part__Feature061", "Part__Feature033"]},
}


def placement_record(placement):
    qx, qy, qz, qw = placement.Rotation.Q
    return {
        "pos_mm": [placement.Base.x, placement.Base.y, placement.Base.z],
        "quat_xyzw": [qx, qy, qz, qw],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    Import.open(str(args.source.resolve()))
    document = FreeCAD.ActiveDocument
    top = document.getObject("BioloidGP")
    if top is None:
        raise ValueError("The STEP file has no BioloidGP assembly")
    leaves = {leaf.name: leaf for leaf in collect_leaves(top, FreeCAD.Placement())}

    assigned = [name for spec in LINKS.values() for name in spec["parts"]]
    if len(assigned) != len(set(assigned)):
        raise ValueError("A STEP part is assigned to more than one rigid link")
    missing = set(leaves) - set(assigned)
    unknown = set(assigned) - set(leaves)
    if missing or unknown:
        raise ValueError(f"Invalid link map: missing={sorted(missing)}, unknown={sorted(unknown)}")

    servo_frames = {}
    for assembly in top.OutList:
        if not assembly.Name.startswith("Full_AX"):
            continue
        horn = next(child for child in assembly.OutList if "horn" in child.Label)
        servo_frames[assembly.Name] = assembly.Placement.multiply(horn.Placement)

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"source": str(args.source.resolve()), "density_kg_m3": 1850.0, "links": []}
    for link_name, spec in LINKS.items():
        frame = FreeCAD.Placement() if spec["servo"] is None else servo_frames[spec["servo"]]
        inverse = frame.inverse()
        local_shapes = []
        labels = []
        for part_name in spec["parts"]:
            leaf = leaves[part_name]
            shape = leaf.shape.copy()
            shape.Placement = inverse.multiply(shape.Placement)
            local_shapes.append(shape)
            labels.append(leaf.label)
        compound = Part.makeCompound(local_shapes)
        mesh = MeshPart.meshFromShape(
            Shape=compound,
            LinearDeflection=0.25,
            AngularDeflection=0.35,
            Relative=False,
        )
        mesh.write(str((args.output / f"{link_name}.stl").resolve()))

        parent_name = spec["parent"]
        parent_frame = FreeCAD.Placement() if parent_name is None else (
            FreeCAD.Placement()
            if LINKS[parent_name]["servo"] is None
            else servo_frames[LINKS[parent_name]["servo"]]
        )
        relative = parent_frame.inverse().multiply(frame)
        volume_mm3 = sum(shape.Volume for shape in local_shapes)
        manifest["links"].append(
            {
                "name": link_name,
                "parent": parent_name,
                "servo": spec["servo"],
                "parts": labels,
                "frame": placement_record(frame),
                "relative_frame": placement_record(relative),
                "volume_mm3": volume_mm3,
                "mass_kg": volume_mm3 * 1e-9 * manifest["density_kg_m3"],
                "center_of_mass_mm": [
                    compound.CenterOfGravity.x,
                    compound.CenterOfGravity.y,
                    compound.CenterOfGravity.z,
                ],
                "bounds_size_mm": [
                    compound.BoundBox.XLength,
                    compound.BoundBox.YLength,
                    compound.BoundBox.ZLength,
                ],
                "mesh": f"{link_name}.stl",
                "triangles": mesh.CountFacets,
            }
        )
        print(link_name, len(local_shapes), f"mass={volume_mm3 * 1e-9 * 1850.0:.3f}kg", ", ".join(labels))

    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Exported {len(LINKS)} exact links from {len(leaves)} STEP solids")


if __name__ == "__main__":
    main()
