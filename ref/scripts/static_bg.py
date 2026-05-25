"""Lock low-motion pixels to the temporal median, keep moving subjects.

Kills the inter-frame wall flicker that LongLive 2.0 / Wan 2.2 produce in
unspecified background regions. Per-frame |frame - median| seeds a soft mask
(threshold -> dilate -> Gaussian blur); the mask composites original over
median. Typical: 2-15% of frame stays "subject", the rest locks static.

Usage:
    python scripts/static_bg.py input.mp4 output.mp4 [--threshold 25]

--threshold N (0-255, default 25): motion detection cutoff. Lower = more
              pixels treated as foreground. 15-35 is the useful range.
"""
import argparse
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np


def static_bg(input_path: Path, output_path: Path, threshold: float = 25.0,
              dilate: int = 11, blur: int = 21) -> None:
    cap = cv2.VideoCapture(str(input_path))
    fps: float = cap.get(cv2.CAP_PROP_FPS)
    width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame.astype(np.float32))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {input_path}")

    stack: np.ndarray = np.stack(frames, axis=0)
    median: np.ndarray = np.median(stack, axis=0)

    fg_pct_acc: float = 0.0
    out_frames: List[np.ndarray] = []
    for frame in frames:
        diff = np.abs(frame - median).mean(axis=2)
        mask = (diff > threshold).astype(np.uint8) * 255
        if dilate > 0:
            mask = cv2.dilate(mask, np.ones((dilate, dilate), np.uint8))
        if blur > 0:
            mask = cv2.GaussianBlur(mask.astype(np.float32), (blur, blur), blur / 3.0)
            mask = mask / 255.0
        else:
            mask = mask.astype(np.float32) / 255.0
        mask = mask[..., None]
        composed = mask * frame + (1 - mask) * median
        out_frames.append(np.clip(composed, 0, 255).astype(np.uint8))
        fg_pct_acc += float(mask.mean())

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp: str = str(output_path.with_suffix(".tmp.mp4"))
    writer = cv2.VideoWriter(tmp, fourcc, fps, (width, height))
    for frame in out_frames:
        writer.write(frame)
    writer.release()

    # Re-encode to h264 via ffmpeg for clean playback in any player.
    import subprocess
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp,
         "-c:v", "libx264", "-preset", "slow", "-crf", "16",
         "-pix_fmt", "yuv420p", str(output_path)],
        check=True,
    )
    Path(tmp).unlink(missing_ok=True)

    avg_fg = fg_pct_acc / len(frames) * 100.0
    print(f"  frames={len(frames)} fps={fps:.1f} size={width}x{height}")
    print(f"  avg foreground area: {avg_fg:.1f}%  (rest locked to temporal median)")
    print(f"  wrote: {output_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--threshold", type=float, default=25.0)
    p.add_argument("--dilate", type=int, default=11)
    p.add_argument("--blur", type=int, default=21)
    args = p.parse_args()
    static_bg(args.input, args.output, args.threshold, args.dilate, args.blur)
    return 0


if __name__ == "__main__":
    sys.exit(main())
