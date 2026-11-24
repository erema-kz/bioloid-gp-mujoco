"""Build an articulated MuJoCo model from the exact STEP link manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET


ARM_LINKS = {"right_arm_1", "right_arm_2", "right_arm_3", "left_arm_1", "left_arm_2", "left_arm_3"}
JOINT_RANGES = {
    "right_arm_1": (-1.5, 1.5),
    "right_arm_2": (-1.3, 1.3),
    "right_arm_3": (-1.8, 1.8),
    "left_arm_1": (-1.5, 1.5),
    "left_arm_2": (-1.3, 1.3),
    "left_arm_3": (-1.8, 1.8),
    "right_leg_1": (-0.8, 0.8),
    "right_leg_2": (-1.2, 1.2),
    "right_leg_3": (-1.5, 1.5),
    "right_leg_4": (-1.2, 1.2),
    "right_leg_5": (-0.8, 0.8),
    "left_leg_1": (-0.8, 0.8),
    "left_leg_2": (-1.2, 1.2),
    "left_leg_3": (-1.5, 1.5),
    "left_leg_4": (-1.2, 1.2),
    "left_leg_5": (-0.8, 0.8),
}


def vector(values, scale=1.0):
    return " ".join(f"{value * scale:.9g}" for value in values)


def quat_wxyz(quat_xyzw):
    x, y, z, w = quat_xyzw
    return vector((w, x, y, z))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    links = {link["name"]: link for link in manifest["links"]}
    children = {name: [] for name in links}
    for link in manifest["links"]:
        if link["parent"] is not None:
            children[link["parent"]].append(link["name"])

    root = ET.Element("mujoco", {"model": "bioloid_gp_exact_step_16dof"})
    ET.SubElement(
        root,
        "compiler",
        {"angle": "radian", "autolimits": "true", "balanceinertia": "true", "boundmass": "0.005", "boundinertia": "0.000001"},
    )
    ET.SubElement(
        root,
        "option",
        {"timestep": "0.001", "integrator": "implicitfast", "solver": "Newton", "iterations": "100", "gravity": "0 0 -9.81"},
    )
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", {"offwidth": "960", "offheight": "720"})

    default = ET.SubElement(root, "default")
    ET.SubElement(default, "joint", {"damping": "0.15", "armature": "0.01", "frictionloss": "0.02", "limited": "true"})
    ET.SubElement(default, "geom", {"condim": "4", "friction": "1.0 0.02 0.002", "solref": "0.015 1", "solimp": "0.85 0.95 0.005"})

    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", {"name": "grid", "type": "2d", "builtin": "checker", "rgb1": "0.16 0.18 0.20", "rgb2": "0.22 0.24 0.27", "width": "512", "height": "512"})
    ET.SubElement(asset, "material", {"name": "floor", "texture": "grid", "texrepeat": "8 8", "reflectance": "0.15"})
    ET.SubElement(asset, "material", {"name": "cad", "rgba": "0.78 0.81 0.88 1"})
    for link in manifest["links"]:
        ET.SubElement(
            asset,
            "mesh",
            {"name": f"mesh_{link['name']}", "file": f"step_links/{link['mesh']}", "scale": "0.001 0.001 0.001"},
        )

    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", {"pos": "1 -1 2", "dir": "-1 1 -2", "diffuse": "0.9 0.9 0.9"})
    ET.SubElement(world, "geom", {"name": "floor", "type": "plane", "size": "4 4 0.1", "material": "floor", "contype": "1", "conaffinity": "2"})
    ET.SubElement(world, "camera", {"name": "tracking", "mode": "trackcom", "pos": "0.57 -0.68 0.39", "xyaxes": "0.75 0.66 0 -0.25 0.29 0.92"})

    bodies = {}

    def add_link(link_name, parent_element):
        link = links[link_name]
        attributes = {"name": link_name}
        if link_name == "base":
            attributes["pos"] = "0 0 0.295034"
            attributes["quat"] = "0.707106781 0.707106781 0 0"
        elif link["parent"] is not None:
            attributes["pos"] = vector(link["relative_frame"]["pos_mm"], 0.001)
            attributes["quat"] = quat_wxyz(link["relative_frame"]["quat_xyzw"])
        body = ET.SubElement(parent_element, "body", attributes)
        bodies[link_name] = body
        mass = link["mass_kg"]
        dx, dy, dz = (value * 0.001 for value in link["bounds_size_mm"])
        inertia = (
            mass * (dy * dy + dz * dz) / 12.0,
            mass * (dx * dx + dz * dz) / 12.0,
            mass * (dx * dx + dy * dy) / 12.0,
        )
        ET.SubElement(
            body,
            "inertial",
            {
                "pos": vector(link["center_of_mass_mm"], 0.001),
                "mass": f"{mass:.9g}",
                "diaginertia": vector(inertia),
            },
        )
        if link_name == "base":
            ET.SubElement(body, "freejoint", {"name": "root"})
            ET.SubElement(body, "site", {"name": "imu", "size": "0.004", "rgba": "0 1 0 1"})
        else:
            lower, upper = JOINT_RANGES[link_name]
            ET.SubElement(
                body,
                "joint",
                {"name": f"{link_name}_joint", "axis": "0 0 1", "range": f"{lower} {upper}"},
            )
        ET.SubElement(
            body,
            "geom",
            {"name": f"{link_name}_visual", "type": "mesh", "mesh": f"mesh_{link_name}", "material": "cad", "contype": "0", "conaffinity": "0", "group": "2", "density": "0"},
        )
        ET.SubElement(
            body,
            "geom",
            {"name": f"{link_name}_collision", "type": "mesh", "mesh": f"mesh_{link_name}", "rgba": "0 0 0 0", "group": "3", "density": "0", "contype": "2", "conaffinity": "1"},
        )
        for child_name in children[link_name]:
            add_link(child_name, body)

    add_link("base", world)

    actuator = ET.SubElement(root, "actuator")
    ordered_joints = []

    def add_actuators(link_name):
        for child_name in children[link_name]:
            lower, upper = JOINT_RANGES[child_name]
            ordered_joints.append(child_name)
            arm = child_name in ARM_LINKS
            ET.SubElement(
                actuator,
                "position",
                {
                    "name": f"{child_name}_act",
                    "joint": f"{child_name}_joint",
                    "ctrlrange": f"{lower} {upper}",
                    "kp": "8" if arm else "12",
                    "kv": "0.5" if arm else "0.8",
                    "forcerange": "-1.5 1.5",
                },
            )
            add_actuators(child_name)

    add_actuators("base")

    sensor = ET.SubElement(root, "sensor")
    ET.SubElement(sensor, "framequat", {"name": "base_orientation", "objtype": "body", "objname": "base"})
    ET.SubElement(sensor, "gyro", {"name": "base_gyro", "site": "imu"})
    ET.SubElement(sensor, "accelerometer", {"name": "base_accel", "site": "imu"})

    keyframe = ET.SubElement(root, "keyframe")
    zeros = " ".join("0" for _ in ordered_joints)
    ET.SubElement(
        keyframe,
        "key",
        {"name": "home", "qpos": f"0 0 0.295034 0.707106781 0.707106781 0 0 {zeros}", "ctrl": zeros},
    )

    ET.indent(root, space="  ")
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(args.destination, encoding="utf-8", xml_declaration=False)
    print(f"Built {args.destination} with {len(ordered_joints)} joints")


if __name__ == "__main__":
    main()
