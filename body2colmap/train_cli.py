"""
CLI entry point for training 3D Gaussian Splats from COLMAP data.

Usage::

    body2colmap-train ./my_colmap_output -o ./trained_splat
    body2colmap-train ./dataset --max-steps 15000 --sh-degree 2

The input directory should contain the standard COLMAP layout::

    data_dir/
    ├── sparse/0/
    │   ├── cameras.txt
    │   ├── images.txt
    │   └── points3D.txt
    └── images/
        └── *.png
"""

import argparse
import sys
from pathlib import Path
from typing import Optional


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="body2colmap-train",
        description="Train 3D Gaussian Splats from a COLMAP dataset using gsplat.",
    )

    # Positional
    p.add_argument(
        "data_dir",
        type=str,
        help="Path to COLMAP dataset (contains sparse/0/ and images/).",
    )

    # Output
    p.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory for trained splat.  Defaults to <data_dir>/trained/.",
    )
    p.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Explicit image directory override (auto-detected by default).",
    )

    # Training
    p.add_argument("--max-steps", type=int, default=30_000, help="Training iterations.")
    p.add_argument("--batch-size", type=int, default=1, help="Images per step.")
    p.add_argument("--sh-degree", type=int, default=3, help="Spherical harmonics degree.")
    p.add_argument("--ssim-lambda", type=float, default=0.2, help="SSIM loss weight.")
    p.add_argument(
        "--strategy",
        choices=["default", "mcmc"],
        default="default",
        help="Densification strategy.",
    )
    p.add_argument("--device", type=str, default="cuda", help="Device (cuda / cpu).")
    p.add_argument(
        "--no-absgrad",
        action="store_true",
        help="Disable absolute-value gradient accumulation.",
    )

    # Densification
    p.add_argument("--refine-start", type=int, default=None)
    p.add_argument("--refine-stop", type=int, default=None)
    p.add_argument("--refine-every", type=int, default=None)
    p.add_argument("--grow-grad2d", type=float, default=None)

    # Checkpointing
    p.add_argument(
        "--save-steps",
        type=int,
        nargs="+",
        default=None,
        help="Steps at which to save intermediate checkpoints.",
    )

    # Stat maps
    p.add_argument(
        "--render-stats",
        action="store_true",
        help="Render per-Gaussian statistic maps (grad norm, view count) "
             "for the first training view after training completes.",
    )

    # Debugging
    p.add_argument(
        "--no-alpha",
        action="store_true",
        help="Ignore alpha channel in images (treat as fully opaque). "
             "Useful for debugging training issues.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information about cameras and point cloud.",
    )

    return p


def _print_progress(
    step: int, max_steps: int, loss: float,
    *, it_per_sec: float = 0.0, n_gaussians: int = 0,
) -> None:
    pct = step / max_steps * 100
    print(
        f"  [{step:>6d}/{max_steps}] ({pct:5.1f}%%)"
        f"  loss={loss:.5f}"
        f"  {it_per_sec:.1f} it/s"
        f"  {n_gaussians:,} gaussians",
        flush=True,
    )


def main(argv: Optional[list] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Resolve output dir
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: data directory does not exist: {data_dir}", file=sys.stderr)
        return 1

    output_dir = Path(args.output) if args.output else data_dir / "trained"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Lazy-import heavy deps so --help stays fast
    try:
        import torch  # noqa: F401
    except ImportError:
        print("Error: PyTorch is required.  pip install torch", file=sys.stderr)
        return 1

    try:
        import gsplat  # noqa: F401
    except ImportError:
        print("Error: gsplat is required.  pip install gsplat", file=sys.stderr)
        return 1

    from .trainer import SplatTrainer, TrainConfig

    # Build config
    cfg = TrainConfig(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        sh_degree=args.sh_degree,
        ssim_lambda=args.ssim_lambda,
        strategy=args.strategy,
        absgrad=not args.no_absgrad,
        device=args.device,
        ignore_alpha=args.no_alpha,
    )

    # Apply optional densification overrides
    if args.refine_start is not None:
        cfg.refine_start = args.refine_start
    if args.refine_stop is not None:
        cfg.refine_stop = args.refine_stop
    if args.refine_every is not None:
        cfg.refine_every = args.refine_every
    if args.grow_grad2d is not None:
        cfg.grow_grad2d = args.grow_grad2d
    if args.save_steps is not None:
        cfg.save_steps = args.save_steps

    # Print banner
    print("=" * 60)
    print("body2colmap-train :: Gaussian Splat Trainer (gsplat)")
    print("=" * 60)
    print(f"  Data:       {data_dir}")
    print(f"  Output:     {output_dir}")
    print(f"  Steps:      {cfg.max_steps}")
    print(f"  SH degree:  {cfg.sh_degree}")
    print(f"  Strategy:   {cfg.strategy}")
    print(f"  Device:     {cfg.device}")
    print("=" * 60)

    try:
        trainer = SplatTrainer(str(data_dir), config=cfg, images_dir=args.images_dir)

        n_images = len(trainer.dataset.image_paths)
        n_points = len(trainer.dataset.points)
        print(f"\n  Loaded {n_images} images, {n_points} initial points")
        print(f"  Scene scale: {trainer.dataset.scene_scale:.3f}")
        if args.no_alpha:
            print(f"  Alpha:       DISABLED (--no-alpha)")
        print()

        # Debug output
        if args.debug:
            import numpy as np
            print("-" * 60)
            print("DEBUG: Camera and Point Cloud Information")
            print("-" * 60)

            # Point cloud stats
            pts = trainer.dataset.points
            pts_min = pts.min(axis=0)
            pts_max = pts.max(axis=0)
            pts_center = (pts_min + pts_max) / 2
            print(f"  Point cloud bounds:")
            print(f"    min:    [{pts_min[0]:8.3f}, {pts_min[1]:8.3f}, {pts_min[2]:8.3f}]")
            print(f"    max:    [{pts_max[0]:8.3f}, {pts_max[1]:8.3f}, {pts_max[2]:8.3f}]")
            print(f"    center: [{pts_center[0]:8.3f}, {pts_center[1]:8.3f}, {pts_center[2]:8.3f}]")

            # Camera positions (first 5 and last)
            c2ws = trainer.dataset.camtoworlds
            cam_positions = c2ws[:, :3, 3]  # (N, 3)
            print(f"\n  Camera positions (first 5):")
            for i in range(min(5, len(cam_positions))):
                pos = cam_positions[i]
                print(f"    [{i:3d}]: [{pos[0]:8.3f}, {pos[1]:8.3f}, {pos[2]:8.3f}]")
            if len(cam_positions) > 5:
                pos = cam_positions[-1]
                print(f"    [{len(cam_positions)-1:3d}]: [{pos[0]:8.3f}, {pos[1]:8.3f}, {pos[2]:8.3f}]")

            # Check if cameras are looking at points
            cam_center = cam_positions.mean(axis=0)
            print(f"\n  Camera centroid: [{cam_center[0]:.3f}, {cam_center[1]:.3f}, {cam_center[2]:.3f}]")

            # Distance from camera center to point cloud center
            dist_to_pts = np.linalg.norm(cam_center - pts_center)
            print(f"  Distance (cam center -> point center): {dist_to_pts:.3f}")

            # Average camera-to-point distance
            avg_cam_dist = np.linalg.norm(cam_positions - pts_center, axis=1).mean()
            print(f"  Average camera distance to point center: {avg_cam_dist:.3f}")

            # Check camera forward directions
            print(f"\n  Camera forward directions (first 3):")
            for i in range(min(3, len(c2ws))):
                # Forward is -Z in camera space, which is -column 2 of rotation
                fwd = -c2ws[i, :3, 2]
                fwd = fwd / np.linalg.norm(fwd)
                print(f"    [{i:3d}]: [{fwd[0]:6.3f}, {fwd[1]:6.3f}, {fwd[2]:6.3f}]")

            # Check if first camera is looking toward point cloud
            c0_pos = cam_positions[0]
            c0_fwd = -c2ws[0, :3, 2]
            c0_fwd = c0_fwd / np.linalg.norm(c0_fwd)
            to_pts = pts_center - c0_pos
            to_pts = to_pts / np.linalg.norm(to_pts)
            dot = np.dot(c0_fwd, to_pts)
            print(f"\n  Camera[0] forward · direction_to_points = {dot:.3f}")
            if dot < 0:
                print("  WARNING: Camera[0] is looking AWAY from point cloud!")
            elif dot < 0.5:
                print("  WARNING: Camera[0] is not directly facing point cloud")

            print("-" * 60)
            print()

        # Train
        result = trainer.train(progress_cb=_print_progress)

        # Summary
        print()
        print("-" * 60)
        print(f"  Training complete: {result.n_gaussians} Gaussians")
        print(f"  Final loss:        {result.final_loss:.6f}")
        print(f"  Visible (count>0): {(result.view_count > 0).sum().item()}")
        avg = result.avg_grad_norm
        print(f"  Avg grad norm:     min={avg.min():.6f}  max={avg.max():.6f}"
              f"  mean={avg[avg > 0].mean():.6f}")
        print("-" * 60)

        # Save PLY
        ply_path = output_dir / "point_cloud.ply"
        result.save_ply(str(ply_path))
        print(f"\n  PLY  -> {ply_path}")

        # Save checkpoint (includes statistics)
        ckpt_path = output_dir / "checkpoint.pt"
        result.save_checkpoint(str(ckpt_path))
        print(f"  CKPT -> {ckpt_path}")

        # Render statistic maps
        if args.render_stats:
            import cv2  # type: ignore

            # Use the first training view
            c2w = trainer.dataset.camtoworlds[0]
            K = trainer.dataset.Ks[0]
            w, h = int(trainer.dataset.image_sizes[0][0]), int(trainer.dataset.image_sizes[0][1])

            stats_dir = output_dir / "stats"
            stats_dir.mkdir(parents=True, exist_ok=True)

            # Gradient norm map (log-scaled for visibility)
            grad_map = trainer.render_stat_map(
                result.avg_grad_norm, c2w, K, w, h, log_scale=True
            )
            grad_path = stats_dir / "grad_norm.png"
            cv2.imwrite(str(grad_path), (grad_map * 255).clip(0, 255).astype("uint8"))
            print(f"  STAT -> {grad_path}")

            # View count map
            count_map = trainer.render_stat_map(
                result.view_count, c2w, K, w, h, log_scale=False
            )
            count_path = stats_dir / "view_count.png"
            cv2.imwrite(str(count_path), (count_map * 255).clip(0, 255).astype("uint8"))
            print(f"  STAT -> {count_path}")

        print(f"\nDone.")
        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
