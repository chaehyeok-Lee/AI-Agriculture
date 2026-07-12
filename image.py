"""
Split ENVI multispectral cubes into band image files.

Usage:
    # Process the current folder if it contains cube.hdr/cube.raw
    python image.py

    # Process every child folder under a parent folder
    python image.py --root C:/Users/chaem/Desktop/dataset/train/ms

For each cube folder, this creates:
    band_images/band_01_713nm.png
    band_images/band_02_736nm.png
    ...
"""

from __future__ import annotations

from pathlib import Path
import argparse
import re

import numpy as np
from PIL import Image


ENVI_DATA_TYPES = {
    1: np.uint8,
    2: np.int16,
    3: np.int32,
    4: np.float32,
    5: np.float64,
    12: np.uint16,
    13: np.uint32,
    14: np.int64,
    15: np.uint64,
}


def read_header(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    header: dict[str, str] = {}
    current_key = None
    current_value: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.upper() == "ENVI":
            continue

        if current_key is not None:
            current_value.append(line)
            if "}" in line:
                header[current_key] = " ".join(current_value)
                current_key = None
                current_value = []
            continue

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip().lower()
        value = value.strip()

        if value.startswith("{") and "}" not in value:
            current_key = key
            current_value = [value]
        else:
            header[key] = value

    return header


def clean(value: str) -> str:
    return str(value).replace("{", "").replace("}", "").strip()


def get_int(header: dict[str, str], key: str) -> int:
    if key not in header:
        raise KeyError(f"Missing required HDR key: {key}")
    return int(float(clean(header[key])))


def get_wavelengths(header: dict[str, str], band_count: int) -> list[str]:
    if "wavelength" not in header:
        return [f"band_{i + 1:02d}" for i in range(band_count)]

    values = [value.strip() for value in clean(header["wavelength"]).split(",") if value.strip()]
    if len(values) != band_count:
        return [f"band_{i + 1:02d}" for i in range(band_count)]
    return values


def find_raw_file(hdr_path: Path) -> Path:
    folder = hdr_path.parent
    stem = hdr_path.stem
    candidates = [
        folder / f"{stem}.raw",
        folder / f"{stem}.img",
        folder / f"{stem}.dat",
        folder / f"{stem}.bin",
        folder / stem,
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    raise FileNotFoundError(f"Could not find RAW file next to {hdr_path}")


def load_cube(hdr_path: Path) -> tuple[np.ndarray, list[str]]:
    header = read_header(hdr_path)
    samples = get_int(header, "samples")
    lines = get_int(header, "lines")
    bands = get_int(header, "bands")
    data_type = get_int(header, "data type")
    interleave = clean(header.get("interleave", "bsq")).lower()
    header_offset = int(float(clean(header.get("header offset", "0"))))
    byte_order = int(float(clean(header.get("byte order", "0"))))

    if data_type not in ENVI_DATA_TYPES:
        raise ValueError(f"Unsupported ENVI data type: {data_type}")
    if interleave not in {"bsq", "bil", "bip"}:
        raise ValueError(f"Unsupported interleave: {interleave}")

    dtype = np.dtype(ENVI_DATA_TYPES[data_type])
    dtype = dtype.newbyteorder("<" if byte_order == 0 else ">")
    raw_path = find_raw_file(hdr_path)
    expected_count = samples * lines * bands

    with raw_path.open("rb") as raw_file:
        raw_file.seek(header_offset)
        raw = np.fromfile(raw_file, dtype=dtype, count=expected_count)

    if raw.size != expected_count:
        raise ValueError(
            f"RAW size mismatch: expected {expected_count:,} values, got {raw.size:,}"
        )

    if interleave == "bsq":
        cube = raw.reshape(bands, lines, samples).transpose(1, 2, 0)
    elif interleave == "bil":
        cube = raw.reshape(lines, bands, samples).transpose(0, 2, 1)
    else:
        cube = raw.reshape(lines, samples, bands)

    wavelengths = get_wavelengths(header, bands)
    return cube, wavelengths


def find_cube_headers(root: Path) -> list[Path]:
    if root.is_file():
        return [root]

    direct_hdr = root / "cube.hdr"
    if direct_hdr.exists():
        return [direct_hdr]

    return sorted(root.rglob("cube.hdr"))


def safe_wavelength_name(wavelength: str, index: int) -> str:
    match = re.search(r"[-+]?\d*\.?\d+", wavelength)
    if not match:
        return f"band{index + 1:02d}"

    value = float(match.group(0))
    if value.is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def normalize_to_uint8(band: np.ndarray) -> np.ndarray:
    band_float = band.astype(np.float32)
    low = float(np.percentile(band_float, 1))
    high = float(np.percentile(band_float, 99))

    if high <= low:
        return np.zeros(band.shape, dtype=np.uint8)

    normalized = (band_float - low) / (high - low)
    normalized = np.clip(normalized, 0, 1)
    return (normalized * 255).astype(np.uint8)


def process_cube(hdr_path: Path, output_name: str, overwrite: bool) -> None:
    cube, wavelengths = load_cube(hdr_path)
    output_dir = hdr_path.parent / output_name
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print(f"folder: {hdr_path.parent}")
    print(f"cube shape: {cube.shape}")
    print(f"output: {output_dir}")
    print()

    for index, wavelength in enumerate(wavelengths):
        band = cube[:, :, index]
        wavelength_name = safe_wavelength_name(wavelength, index)
        output_path = output_dir / f"band_{index + 1:02d}_{wavelength_name}nm.png"

        if output_path.exists() and not overwrite:
            print(f"skip: {output_path.name}")
            continue

        image = Image.fromarray(normalize_to_uint8(band), mode="L")
        image.save(output_path)
        print(
            f"saved: {output_path.name}, shape={band.shape}, dtype={band.dtype}, "
            f"min={band.min()}, max={band.max()}, mean={band.mean():.4f}"
        )
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split all cube.hdr/cube.raw files under a root folder into band PNG images."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Folder to search recursively. Defaults to the current folder.",
    )
    parser.add_argument(
        "--output-name",
        default="band_images",
        help="Output folder name created inside each cube folder.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing band PNG files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    hdr_paths = find_cube_headers(root)

    if not hdr_paths:
        raise FileNotFoundError(f"No cube.hdr files found under: {root}")

    print(f"root: {root}")
    print(f"cube folders found: {len(hdr_paths)}")
    print()

    for hdr_path in hdr_paths:
        process_cube(hdr_path, args.output_name, args.overwrite)


if __name__ == "__main__":
    main()
