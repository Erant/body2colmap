"""
Tests for ColmapExporter's camera table: one CAMERA_ID per distinct lens.
"""

import numpy as np

from body2colmap.camera import Camera
from body2colmap.exporter import ColmapExporter


def _camera(fx, x=0.0):
    cam = Camera(focal_length=(fx, fx), image_size=(720, 1280),
                 position=np.array([x, 0.0, 3.0], dtype=np.float32))
    cam.look_at(np.zeros(3, dtype=np.float32))
    return cam


def _rows(path):
    return [line.split() for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")]


def test_one_lens_exports_one_camera(tmp_path):
    cams = [_camera(1183.0, x) for x in (0.0, 0.5, 1.0)]
    ColmapExporter(cams, ["a.png", "b.png", "c.png"]).export(tmp_path)

    cameras = _rows(tmp_path / "cameras.txt")
    assert len(cameras) == 1
    assert cameras[0][:4] == ["1", "PINHOLE", "720", "1280"]
    assert "# Number of cameras: 1" in (tmp_path / "cameras.txt").read_text()
    assert [row[8] for row in _rows(tmp_path / "images.txt")] == ["1", "1", "1"]


def test_mixed_lenses_get_their_own_cameras(tmp_path):
    cams = [_camera(3076.0), _camera(1183.0, 0.5), _camera(3076.0, 1.0), _camera(2000.0, 1.5)]
    ColmapExporter(cams, ["a.png", "b.png", "c.png", "d.png"]).export(tmp_path)

    cameras = _rows(tmp_path / "cameras.txt")
    assert [(row[0], float(row[4])) for row in cameras] == [
        ("1", 3076.0), ("2", 1183.0), ("3", 2000.0)]
    images = _rows(tmp_path / "images.txt")
    assert [(row[8], row[9]) for row in images] == [
        ("1", "a.png"), ("2", "b.png"), ("1", "c.png"), ("3", "d.png")]


def test_lenses_equal_to_the_written_precision_share_a_camera(tmp_path):
    cams = [_camera(1183.0), _camera(1183.0 + 1e-9, 0.5)]
    ColmapExporter(cams, ["a.png", "b.png"]).export(tmp_path)
    assert len(_rows(tmp_path / "cameras.txt")) == 1
