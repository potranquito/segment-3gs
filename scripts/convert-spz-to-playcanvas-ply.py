#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import spz


def write_playcanvas_ply(input_path: Path, output_path: Path, max_splats: int | None) -> None:
    cloud = spz.load_spz(str(input_path))
    n = int(cloud.num_points)
    if n <= 0:
        raise SystemExit(f"no splats loaded from {input_path}")

    positions = cloud.positions.reshape(n, 3).astype(np.float32)
    colors = cloud.colors.reshape(n, 3).astype(np.float32)
    alphas = cloud.alphas.astype(np.float32)
    scales = cloud.scales.reshape(n, 3).astype(np.float32)
    rotations = cloud.rotations.reshape(n, 4).astype(np.float32)

    selected = np.arange(n)
    if max_splats is not None and 0 < max_splats < n:
        selected = np.argpartition(alphas, -max_splats)[-max_splats:]
        selected = selected[np.argsort(alphas[selected])[::-1]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = int(len(selected))
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {count}",
            "property float x",
            "property float y",
            "property float z",
            "property float f_dc_0",
            "property float f_dc_1",
            "property float f_dc_2",
            "property float opacity",
            "property float scale_0",
            "property float scale_1",
            "property float scale_2",
            "property float rot_0",
            "property float rot_1",
            "property float rot_2",
            "property float rot_3",
            "end_header",
            "",
        ]
    ).encode("ascii")

    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("f_dc_0", "<f4"),
            ("f_dc_1", "<f4"),
            ("f_dc_2", "<f4"),
            ("opacity", "<f4"),
            ("scale_0", "<f4"),
            ("scale_1", "<f4"),
            ("scale_2", "<f4"),
            ("rot_0", "<f4"),
            ("rot_1", "<f4"),
            ("rot_2", "<f4"),
            ("rot_3", "<f4"),
        ]
    )
    rows = np.empty(count, dtype=dtype)
    rows["x"] = positions[selected, 0]
    rows["y"] = positions[selected, 1]
    rows["z"] = positions[selected, 2]
    rows["f_dc_0"] = colors[selected, 0]
    rows["f_dc_1"] = colors[selected, 1]
    rows["f_dc_2"] = colors[selected, 2]
    rows["opacity"] = alphas[selected]
    rows["scale_0"] = scales[selected, 0]
    rows["scale_1"] = scales[selected, 1]
    rows["scale_2"] = scales[selected, 2]
    # SPZ exposes quaternions as x, y, z, w. PlayCanvas PLY expects w, x, y, z.
    rows["rot_0"] = rotations[selected, 3]
    rows["rot_1"] = rotations[selected, 0]
    rows["rot_2"] = rotations[selected, 1]
    rows["rot_3"] = rotations[selected, 2]

    with output_path.open("wb") as file:
        file.write(header)
        rows.tofile(file)

    size_mb = output_path.stat().st_size / 1_000_000
    if count != n:
        print(f"wrote {count:,} of {n:,} splats to {output_path} ({size_mb:.1f} MB)")
    else:
        print(f"wrote all {count:,} splats to {output_path} ({size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-splats", type=int, default=0)
    args = parser.parse_args()
    write_playcanvas_ply(args.input, args.output, args.max_splats)


if __name__ == "__main__":
    main()
