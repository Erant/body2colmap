"""Export node for Body2COLMAP - exports COLMAP sparse reconstruction format."""

from pathlib import Path
import cv2
from body2colmap.exporter import ColmapExporter
from ..core.comfy_utils import comfy_to_cv2


class Body2COLMAP_ExportCOLMAP:
    """Export COLMAP sparse reconstruction format for 3DGS training."""

    CATEGORY = "Body2COLMAP"
    FUNCTION = "export"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    OUTPUT_NODE = True  # Terminal node - produces file output
    OUTPUT_TOOLTIPS = ("Path to the output directory containing COLMAP files",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "render_data": ("B2C_RENDER_DATA",),
                "output_directory": ("STRING", {
                    "default": "output/colmap",
                    "tooltip": "Directory for COLMAP files (creates sparse/0/ subdirectory)"
                }),
                "image_name": ("STRING", {
                    "default": "frame",
                    "tooltip": "Base name for images (follows ComfyUI convention: <name>_%05d.png)"
                }),
            },
            "optional": {
                "images": ("IMAGE",),
                "pointcloud_samples": ("INT", {
                    "default": 10000,
                    "min": 1000,
                    "max": 500000,
                    "step": 1000,
                    "tooltip": "Number of points to sample from mesh surface"
                }),
            }
        }

    def export(self, render_data, output_directory, image_name,
               images=None, pointcloud_samples=10000):
        """
        Export COLMAP format files.

        Creates:
            output_directory/
            ├── images/           (if images provided)
            │   ├── frame_00001.png
            │   └── ...
            └── sparse/0/
                ├── cameras.txt   (camera intrinsics)
                ├── images.txt    (camera extrinsics per image)
                └── points3D.txt  (initial point cloud)
        """
        # Extract data
        cameras = render_data["cameras"]
        scene = render_data["scene"]
        width, height = render_data["resolution"]
        focal_length = render_data["focal_length"]

        # Create output directories
        output_path = Path(output_directory)
        sparse_path = output_path / "sparse" / "0"
        sparse_path.mkdir(parents=True, exist_ok=True)

        # Generate image filenames using ComfyUI convention
        # Format: <image_name>_%05d.png with 1-based indexing
        image_names = [f"{image_name}_{i+1:05d}.png" for i in range(len(cameras))]

        # Save images if provided
        if images is not None:
            images_path = output_path / "images"
            images_path.mkdir(parents=True, exist_ok=True)

            # Convert ComfyUI images to CV2 format and save
            cv2_images = comfy_to_cv2(images)

            for img, filename in zip(cv2_images, image_names):
                img_path = images_path / filename
                cv2.imwrite(str(img_path), img)

        # Create COLMAP exporter using classmethod
        exporter = ColmapExporter.from_scene_and_cameras(
            scene=scene,
            cameras=cameras,
            image_names=image_names,
            n_pointcloud_samples=pointcloud_samples
        )

        # Export COLMAP files
        exporter.export(output_dir=sparse_path)

        print(f"[Body2COLMAP] Exported COLMAP files to: {sparse_path}")
        print(f"[Body2COLMAP] - cameras.txt: {len(cameras)} cameras")
        print(f"[Body2COLMAP] - images.txt: {len(cameras)} images")
        print(f"[Body2COLMAP] - points3D.txt: {pointcloud_samples} points")

        if images is not None:
            print(f"[Body2COLMAP] - images/: {len(cv2_images)} image files")

        return (str(output_path.absolute()),)
