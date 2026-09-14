import base64
import io
import json
import struct
import zipfile
import os
import subprocess
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient
from PIL import Image
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from app import app
from color_3mf import CORE, MATERIAL, ConversionError, convert_glb_to_color_3mf


def sample_glb():
    """Closed cubes with red/blue texture samples and a green PBR material.

    A repeated, mirrored node also checks scene transforms and winding.
    """
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    pixels[:4] = [255, 0, 0]
    pixels[4:] = [0, 0, 255]
    material = PBRMaterial(baseColorTexture=Image.fromarray(pixels), baseColorFactor=[1., 1., 1., 1.])
    scene = trimesh.Scene()
    for i, v in enumerate((0.75, 0.25)):
        box = trimesh.creation.box(extents=[0.01, 0.01, 0.01])
        box.visual = TextureVisuals(uv=np.tile([0.5, v], (len(box.vertices), 1)), material=material)
        scene.add_geometry(box, node_name=f"texture-{i}", geom_name=f"texture-{i}",
                           transform=trimesh.transformations.translation_matrix([i * 0.01, 0, 0]))
    green = trimesh.creation.box(extents=[0.01, 0.01, 0.01])
    green.visual = TextureVisuals(material=PBRMaterial(baseColorFactor=[0., 1., 0., 1.]))
    scene.add_geometry(green, node_name="green", geom_name="green",
                       transform=trimesh.transformations.translation_matrix([0.02, 0, 0]))
    transform = trimesh.transformations.translation_matrix([0.03, 0, 0])
    transform[0, 0] = -1
    scene.graph.update(frame_to="green-mirrored", matrix=transform, geometry="green")
    return scene.export(file_type="glb")


def unpack(data):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert archive.testzip() is None
        root = ET.fromstring(archive.read("3D/3dmodel.model"))
        settings = json.loads(archive.read("Metadata/project_settings.config"))
    vertices = np.array([[float(v.get(c)) for c in "xyz"] for v in root.findall(f".//{{{CORE}}}vertex")])
    triangles = root.findall(f".//{{{CORE}}}triangle")
    faces = np.array([[int(t.get(c)) for c in ("v1", "v2", "v3")] for t in triangles])
    return root, vertices, faces, triangles, settings


def test_texture_orientation_materials_instances_and_units():
    result = convert_glb_to_color_3mf(sample_glb(), color_count=4, refinement=0)
    root, vertices, faces, triangles, settings = unpack(result.data)
    assert result.palette == ["#0000FF", "#00FF00", "#FF0000"]
    assert settings["filament_colour"] == result.palette
    np.testing.assert_allclose(np.ptp(vertices, axis=0), [40, 10, 10], atol=1e-5)
    assert len(faces) == 48  # Four instances, not three unique geometries.
    centroids = vertices[faces].mean(axis=1)
    for triangle, centroid in zip(triangles, centroids):
        colour = result.palette[int(triangle.get("p1"))]
        assert colour == ("#FF0000" if centroid[0] < 9.99 else "#0000FF" if centroid[0] < 19.99 else "#00FF00") or abs(centroid[0] - 10) < 1e-4 or abs(centroid[0] - 20) < 1e-4
    # Mirroring must preserve positive signed volume for all closed parts.
    for i in range(4):
        part_faces = faces[i * 12:(i + 1) * 12]
        mesh = trimesh.Trimesh(vertices, part_faces, process=False)
        assert mesh.volume > 0
    assert len(root.findall(f".//{{{MATERIAL}}}color")) == 3
    assert {t.get("paint_color") for t in triangles} == {"4", "8", "0C"}


def test_refinement_palette_limit_size_and_determinism():
    glb = sample_glb()
    first = convert_glb_to_color_3mf(glb, color_count=2, refinement=2, size_mm=12)
    second = convert_glb_to_color_3mf(glb, color_count=2, refinement=2, size_mm=12)
    assert len(first.palette) == 2
    assert first.face_count > 48
    np.testing.assert_allclose(first.size_mm, [12, 3, 3], atol=1e-5)
    assert first.palette == second.palette
    with zipfile.ZipFile(io.BytesIO(first.data)) as a, zipfile.ZipFile(io.BytesIO(second.data)) as b:
        assert a.read("3D/3dmodel.model") == b.read("3D/3dmodel.model")


def test_missing_colours_reports_single_colour():
    glb = trimesh.creation.box().export(file_type="glb")
    result = convert_glb_to_color_3mf(glb, size_mm=10)
    assert len(result.palette) == 1
    assert any("一种颜色" in warning for warning in result.warnings)


def test_linear_material_factor_is_converted_to_srgb():
    mesh = trimesh.creation.box()
    mesh.visual = TextureVisuals(material=PBRMaterial(baseColorFactor=[0.5, 0.5, 0.5, 1.]))
    result = convert_glb_to_color_3mf(mesh.export(file_type="glb"), size_mm=10)
    channels = [int(result.palette[0][i:i+2], 16) for i in (1, 3, 5)]
    assert all(186 <= c <= 190 for c in channels)


def test_vertex_colours_survive():
    mesh = trimesh.creation.box()
    mesh.visual.vertex_colors = np.tile([255, 0, 0, 255], (len(mesh.vertices), 1))
    result = convert_glb_to_color_3mf(mesh.export(file_type="glb"), size_mm=10)
    assert result.palette == ["#FF0000"]


def test_vertex_colour_with_material_factor():
    mesh = trimesh.creation.box()
    mesh.visual = TextureVisuals(material=PBRMaterial(baseColorFactor=[0.5, 1., 1., 1.]))
    mesh.visual.vertex_attributes["color"] = np.tile(np.array([255, 0, 0, 255], dtype=np.uint8), (len(mesh.vertices), 1))
    result = convert_glb_to_color_3mf(mesh.export(file_type="glb"), size_mm=10)
    colour = result.palette[0]
    assert 186 <= int(colour[1:3], 16) <= 190
    assert int(colour[3:], 16) == 0


@pytest.mark.parametrize("count", [1, 3, 16])
@pytest.mark.skipif(not os.environ.get("BAMBU_STUDIO_BIN"), reason="Optional Bambu Studio compatibility test")
def test_bambu_roundtrip_preserves_palette_and_face_assignments(tmp_path, count):
    # Optional test-only executable. The runtime converter never invokes Studio.
    scene = trimesh.Scene()
    for i in range(count):
        mesh = trimesh.creation.box(extents=[0.004, 0.004, 0.004])
        # Distinct saturated RGB values, expressed as glTF linear factors.
        rgb = [float((i >> bit) & 1) for bit in range(3)]
        if i >= 8:
            rgb = [0.25 + 0.5 * value for value in rgb]
        mesh.visual = TextureVisuals(material=PBRMaterial(baseColorFactor=rgb + [1.]))
        scene.add_geometry(mesh, transform=trimesh.transformations.translation_matrix([i * 0.005, 0, 0]))
    result = convert_glb_to_color_3mf(scene.export(file_type="glb"), color_count=count, size_mm=40)
    source, target = tmp_path / "input.3mf", tmp_path / "roundtrip.3mf"
    source.write_bytes(result.data)
    command = [os.environ["BAMBU_STUDIO_BIN"], "--debug", "2", "--info", "--export-3mf", str(target), str(source)]
    process = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=90)
    assert process.returncode == 0, process.stdout + process.stderr
    with zipfile.ZipFile(target) as archive:
        config = json.loads(archive.read("Metadata/project_settings.config"))
        assert config["filament_colour"] == result.palette
        actual_paints = Counter()
        for name in archive.namelist():
            if name.endswith(".model"):
                actual_paints.update(t.get("paint_color") for t in ET.fromstring(archive.read(name)).iter()
                                     if t.tag == f"{{{CORE}}}triangle")
    _, _, _, triangles, _ = unpack(result.data)
    assert actual_paints == Counter(t.get("paint_color") for t in triangles)
    loaded = trimesh.load_scene(target).to_mesh()
    np.testing.assert_allclose(loaded.extents, result.size_mm, atol=1e-4)


@pytest.mark.parametrize("options", [{"color_count": 0}, {"color_count": 17}, {"color_count": True},
    {"color_count": "4"}, {"refinement": 3}, {"size_mm": 0}, {"size_mm": float("nan")}, {"size_mm": "10"}])
def test_invalid_options(options):
    with pytest.raises(ConversionError):
        convert_glb_to_color_3mf(sample_glb(), **options)


def glb_header_only(header):
    payload = json.dumps(header).encode()
    payload += b" " * (-len(payload) % 4)
    return struct.pack("<4sIIII", b"glTF", 2, 20 + len(payload), len(payload), 0x4E4F534A) + payload


@pytest.mark.parametrize("header", [
    {"images": [{"uri": "file:///etc/passwd"}]},
    {"buffers": [{"uri": "https://example.com/model.bin"}]},
    {"extensionsRequired": ["KHR_draco_mesh_compression"]},
    {"samplers": [{"wrapS": 33071}]},
    {"materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"texCoord": 1}}}]},
])
def test_unsupported_inputs_rejected_before_loading(header):
    with pytest.raises(ConversionError):
        convert_glb_to_color_3mf(glb_header_only(header))


def test_api_download_and_auth():
    client = TestClient(app)
    body = {"format": "3mf", "glb_base64": base64.b64encode(sample_glb()).decode(),
            "color_count": 4, "size_mm": 10}
    assert client.post("/api/convert-format", json=body).status_code == 401
    headers = {"Decrypted-Userinfo": '{"name":"local"}'}
    response = client.post("/api/convert-format", json=body, headers=headers)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "model/3mf"
    assert "accessory-color.3mf" in response.headers["content-disposition"]
    assert len(response.headers["x-color-palette"].split(",")) == 3
    unpack(response.content)
    body["color_count"] = 100
    assert client.post("/api/convert-format", json=body, headers=headers).status_code == 400
    assert client.post("/api/convert-format", json={"format": "3mf", "glb_base64": "%%%"}, headers=headers).status_code == 400
    assert client.post("/api/convert-format", content="{", headers=headers).status_code == 400


def test_stl_download_preserves_instances():
    client = TestClient(app)
    response = client.post("/api/convert-format", headers={"Decrypted-Userinfo": '{"name":"local"}'},
                           json={"format": "stl", "glb_base64": base64.b64encode(sample_glb()).decode()})
    assert response.status_code == 200
    mesh = trimesh.load(io.BytesIO(response.content), file_type="stl")
    assert len(mesh.faces) == 48
    np.testing.assert_allclose(mesh.extents, [0.04, 0.01, 0.01], atol=1e-6)


if __name__ == "__main__":
    # Generate inspectable fixtures; no AI calls and no user model is required.
    from pathlib import Path
    directory = Path("artifacts/color-3mf")
    directory.mkdir(parents=True, exist_ok=True)
    glb = sample_glb()
    directory.joinpath("reference.glb").write_bytes(glb)
    result = convert_glb_to_color_3mf(glb, color_count=4, size_mm=20)
    directory.joinpath("reference.3mf").write_bytes(result.data)
    print(json.dumps({"palette": result.palette, "size_mm": result.size_mm, "faces": result.face_count}))
