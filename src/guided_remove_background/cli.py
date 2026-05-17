"""CLI entry point for guided background removal."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from . import MODES
from .clients.http_utils import SUPPORTED_EXTS
from .remove_bg import remove_bg


def main() -> None:
    load_dotenv(override=True)

    parser = argparse.ArgumentParser(
        description="Guided background removal — tell RMBG what to keep",
    )
    parser.add_argument("--image", required=True, type=Path,
                        help="Input image path")
    parser.add_argument("--prompts", nargs="+", required=True,
                        help="Foreground guidance: describe what to keep")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output PNG path")
    parser.add_argument("--mode", default="guided", choices=MODES,
                        help="guided = user-directed, rmbg-only = unguided baseline")
    parser.add_argument("--edge-band", type=int, default=8)
    parser.add_argument("--dilation", type=int, default=5)
    parser.add_argument("--blur", type=float, default=1.0)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip judge verification loop")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.image.exists():
        sys.exit(f"Image not found: {args.image}")
    if args.image.suffix.lower() not in SUPPORTED_EXTS:
        sys.exit(f"Unsupported format '{args.image.suffix}'. Use: {sorted(SUPPORTED_EXTS)}")

    result = remove_bg(
        image=args.image,
        prompts=[p.strip() for p in args.prompts if p.strip()],
        output=args.output,
        mode=args.mode,
        edge_band_px=args.edge_band,
        dilation=args.dilation,
        blur=args.blur,
        min_score=args.min_score,
        verify=not args.no_verify,
    )
    print(f"Result: {result.output_path}")
    print(f"Preview: {result.preview_path}")
    print(f"Elapsed: {result.elapsed_s:.1f}s")


if __name__ == "__main__":
    main()
