import argparse
import re
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


DEFAULT_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".m4v", ".mkv")


def natural_sort_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert an image sequence in a folder to a video."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Folder containing the image sequence.",
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Output video path, e.g. output.mp4 or output.gif.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=24,
        help="Playback speed in frames per second. Default: 24.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=DEFAULT_EXTENSIONS,
        help=(
            "Image file extensions to include. Default: "
            + " ".join(DEFAULT_EXTENSIONS)
        ),
    )
    return parser.parse_args()


def collect_image_paths(input_dir, extensions):
    normalized_extensions = {
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in extensions
    }
    return sorted(
        (
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in normalized_extensions
        ),
        key=natural_sort_key,
    )


def import_cv2():
    try:
        import cv2
    except ImportError:
        return None
    return cv2


def rgb_to_bgr(frame, cv2):
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    raise ValueError(f"Unsupported frame shape: {frame.shape}")


def write_video(image_paths, output_path, fps):
    cv2 = import_cv2()
    if cv2 is None:
        try:
            write_imageio_video(image_paths, output_path, fps)
            return
        except Exception as error:
            raise RuntimeError(
                "Writing video containers such as MP4, AVI, MOV, and MKV "
                "requires opencv-python or imageio-ffmpeg."
            ) from error

    first_frame = np.asarray(imageio.imread(image_paths[0]))
    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for: {output_path}")

    try:
        for image_path in image_paths:
            frame = np.asarray(imageio.imread(image_path))
            if frame.shape[:2] != (height, width):
                raise ValueError(
                    "All frames must have the same dimensions. "
                    f"Expected {(height, width)}, got {frame.shape[:2]} for {image_path}"
                )
            writer.write(rgb_to_bgr(frame, cv2))
    finally:
        writer.release()


def write_imageio_video(image_paths, output_path, fps):
    with imageio.get_writer(str(output_path), fps=fps) as writer:
        for image_path in image_paths:
            writer.append_data(imageio.imread(image_path))


def write_imageio_sequence(image_paths, output_path, fps):
    duration = 1000 / fps
    with imageio.get_writer(str(output_path), duration=duration) as writer:
        for image_path in image_paths:
            writer.append_data(imageio.imread(image_path))


def image_sequence_to_video(input_dir, output_path, fps, extensions):
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    if fps <= 0:
        raise ValueError("FPS must be greater than zero.")

    image_paths = collect_image_paths(input_dir, extensions)
    if not image_paths:
        raise ValueError(f"No matching image files found in: {input_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() in VIDEO_EXTENSIONS:
        write_video(image_paths, output_path, fps)
    else:
        write_imageio_sequence(image_paths, output_path, fps)

    return len(image_paths)


def main():
    args = parse_args()
    try:
        frame_count = image_sequence_to_video(
            args.input_dir,
            args.output_path,
            args.fps,
            args.extensions,
        )
    except (RuntimeError, ValueError) as error:
        raise SystemExit(f"Error: {error}")
    print(f"Wrote {frame_count} frames to {args.output_path}")


if __name__ == "__main__":
    main()
