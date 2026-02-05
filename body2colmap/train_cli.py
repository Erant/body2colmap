"""
CLI entry point for body2colmap-train.

Usage:
    body2colmap-train <colmap_dir> -o <output_dir> [--max-steps N] [--sh-degree N]
"""

import argparse
import sys
from typing import Optional


def main(argv: Optional[list] = None) -> int:
    """
    Main entry point for the training CLI.

    Args:
        argv: Command-line arguments (None = sys.argv)

    Returns:
        Exit code (0 = success, non-zero = error)
    """
    parser = argparse.ArgumentParser(
        prog="body2colmap-train",
        description="Train 3D Gaussian Splats from a COLMAP dataset produced by body2colmap",
    )

    parser.add_argument(
        "colmap_dir",
        help="Path to COLMAP dataset directory (contains sparse/0/ and images/)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="Output directory for trained PLY file",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30000,
        help="Number of training iterations (default: 30000)",
    )
    parser.add_argument(
        "--sh-degree",
        type=int,
        default=3,
        help="Maximum spherical harmonics degree (default: 3)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device (default: cuda)",
    )
    parser.add_argument(
        "--bg-color",
        type=str,
        default="0,0,0",
        metavar="R,G,B",
        help="Background color as R,G,B floats 0-1 (default: 0,0,0 black)",
    )
    parser.add_argument(
        "--ssim-lambda",
        type=float,
        default=0.2,
        help="SSIM loss weight; L1 weight = 1 - this (default: 0.2)",
    )
    parser.add_argument(
        "--scale-reg",
        type=float,
        default=0.01,
        help="Scale regularization weight to penalize oversized Gaussians (default: 0.01)",
    )
    parser.add_argument(
        "--strategy",
        choices=["default", "mcmc"],
        default="default",
        help="Densification strategy: 'default' (gradient-based) or 'mcmc' (stochastic) (default: default)",
    )
    parser.add_argument(
        "--cap-max",
        type=int,
        default=1_000_000,
        help="Max number of Gaussians (mcmc strategy only, default: 1000000)",
    )
    parser.add_argument(
        "--noise-lr",
        type=float,
        default=5e5,
        help="Noise learning rate for MCMC position perturbation (mcmc only, default: 5e5)",
    )
    parser.add_argument(
        "--export-rgb",
        metavar="DIR",
        help="Export per-view RGB renders to DIR",
    )
    parser.add_argument(
        "--export-depth",
        metavar="DIR",
        help="Export per-view depth renders to DIR (grayscale, near=dark far=bright)",
    )
    parser.add_argument(
        "--export-confidence",
        metavar="DIR",
        help="Export per-view confidence maps to DIR (bright=high grad norm=uncertain)",
    )
    parser.add_argument(
        "--export-scale",
        metavar="DIR",
        help="Export per-view scale magnitude renders to DIR (grayscale, bright=large)",
    )
    parser.add_argument(
        "--export-error",
        metavar="DIR",
        help="Export per-view reconstruction error maps to DIR (grayscale, bright=high error)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args(argv)

    # Parse bg_color
    try:
        bg_color = tuple(float(x) for x in args.bg_color.split(","))
        if len(bg_color) != 3:
            raise ValueError
    except ValueError:
        print(f"Error: invalid --bg-color '{args.bg_color}'. Use R,G,B (e.g. 0,0,0)", file=sys.stderr)
        return 1

    try:
        from .trainer import train

        result = train(
            colmap_dir=args.colmap_dir,
            output_dir=args.output_dir,
            max_steps=args.max_steps,
            sh_degree=args.sh_degree,
            device=args.device,
            bg_color=bg_color,
            ssim_lambda=args.ssim_lambda,
            scale_reg_lambda=args.scale_reg,
            strategy_type=args.strategy,
            cap_max=args.cap_max,
            noise_lr=args.noise_lr,
            export_rgb_dir=args.export_rgb,
            export_depth_dir=args.export_depth,
            export_confidence_dir=args.export_confidence,
            export_scale_dir=args.export_scale,
            export_error_dir=args.export_error,
            verbose=not args.quiet,
        )

        if not args.quiet:
            print(f"\nTraining complete:")
            print(f"  Final loss: {result.final_loss:.6f}")
            print(f"  Gaussians:  {result.n_gaussians}")
            print(f"  SH degree:  {result.sh_degree}")
            print(f"  Output:     {args.output_dir}/point_cloud.ply")

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ImportError as e:
        print(f"Error: Missing dependency: {e}", file=sys.stderr)
        print(
            "Install training dependencies with: pip install body2colmap[train]",
            file=sys.stderr,
        )
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
