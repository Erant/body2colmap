"""Export node for Body2COLMAP - exports COLMAP sparse reconstruction format."""

from pathlib import Path
import cv2
from body2colmap.exporter import ColmapExporter, ImageExporter
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
                "images": ("IMAGE",),
                "output_directory": ("STRING", {
                    "default": "output/colmap",
                    "tooltip": "Directory for COLMAP files (creates sparse/0/ subdirectory)"
                }),
            },
            "optional": {
                "filename_pattern": ("STRING", {
                    "default": "frame_{:04d}.png",
                    "tooltip": "Filename pattern for images.txt (must match SaveImage output)"
                }),
                "pointcloud_samples": ("INT", {
                    "default": 50000,
                    "min": 1000,
                    "max": 500000,
                    "step": 1000,
                    "tooltip": "Number of points to sample from mesh surface"
                }),
                "save_images": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Also save images to output_directory/images/"
                }),
            }
        }

    def export(self, render_data, images, output_directory,
               filename_pattern="frame_{:04d}.png",
               pointcloud_samples=50000,
               save_images=True):
        """
        Export COLMAP format files.

        Creates:
            output_directory/
            ├── images/           (if save_images=True)
            │   ├── frame_0000.png
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

        # Generate image filenames for COLMAP
        image_names = ImageExporter.generate_filenames(
            n_frames=len(cameras),
            pattern=filename_pattern,
            start_index=0
        )

        if save_images:
            images_path = output_path / "images"
            images_path.mkdir(parents=True, exist_ok=True)

            # Convert ComfyUI images to CV2 format and save
            cv2_images = comfy_to_cv2(images)

            for i, (img, filename) in enumerate(zip(cv2_images, image_names)):
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

        if save_images:
            print(f"[Body2COLMAP] - images/: {len(cv2_images)} image files")

        return (str(output_path.absolute()),)
