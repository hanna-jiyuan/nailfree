"""GLB base colours -> a limited palette -> Bambu-compatible painted 3MF.

Independent Python implementation; no Bambu Studio algorithm code is vendored.
The 3MF paint encoding is an interoperability format, not an RGB attribute.
See docs/color-3mf.md for supported inputs and limitations.
"""
from __future__ import annotations

import io
import json
import struct
import zipfile
from dataclasses import dataclass
from xml.etree import ElementTree as ET

import cv2
import numpy as np
import trimesh

MAX_GLB_BYTES = 50 * 1024 * 1024
MAX_FACES = 200_000
CORE = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
MATERIAL = "http://schemas.microsoft.com/3dmanufacturing/material/2015/02"
BAMBU = "http://schemas.bambulab.com/package/2021"
ET.register_namespace("", CORE)
ET.register_namespace("m", MATERIAL)


class ConversionError(ValueError):
    """An input that cannot be converted without silently losing information."""


@dataclass
class Color3MF:
    data: bytes
    palette: list[str]
    face_count: int
    size_mm: list[float]
    warnings: list[str]


def _validate_glb(data: bytes) -> dict:
    if len(data) > MAX_GLB_BYTES:
        raise ConversionError("GLB 文件不能超过 50MB")
    if len(data) < 20:
        raise ConversionError("不是有效的 GLB 文件")
    magic, version, length, chunk_length, chunk_type = struct.unpack_from("<4sIIII", data)
    if magic != b"glTF" or version != 2 or length != len(data) or chunk_type != 0x4E4F534A:
        raise ConversionError("仅支持 GLB 2.0 文件")
    if chunk_length > len(data) - 20:
        raise ConversionError("GLB JSON 数据不完整")
    try:
        header = json.loads(data[20:20 + chunk_length])
        for item in header.get("buffers", []) + header.get("images", []):
            if "uri" in item and not item["uri"].startswith("data:"):
                raise ConversionError("请导出内嵌纹理的 GLB；不支持外部文件或纹理网址")
        unsupported = set(header.get("extensionsRequired", [])) - {"KHR_materials_unlit"}
        if unsupported:
            raise ConversionError("请先导出不依赖扩展的 GLB：" + ", ".join(sorted(unsupported)))
        for mesh in header.get("meshes", []):
            for primitive in mesh.get("primitives", []):
                if primitive.get("targets"):
                    raise ConversionError("请先将变形目标烘焙为静态网格")
                if primitive.get("mode", 4) != 4:
                    raise ConversionError("仅支持三角形网格")
                attrs = primitive.get("attributes", {})
                if "COLOR_0" in attrs and "TEXCOORD_0" in attrs:
                    raise ConversionError("暂不支持同时叠加顶点颜色与纹理，请先烘焙为基础颜色贴图")
        for material in header.get("materials", []):
            tex = material.get("pbrMetallicRoughness", {}).get("baseColorTexture", {})
            if tex.get("texCoord", 0) != 0 or tex.get("extensions"):
                raise ConversionError("请将基础颜色贴图烘焙到 UV0，不支持纹理坐标变换扩展")
        # trimesh does not expose texture sampler wrap modes on its material.
        for sampler in header.get("samplers", []):
            if sampler.get("wrapS", 10497) != 10497 or sampler.get("wrapT", 10497) != 10497:
                raise ConversionError("暂仅支持重复平铺纹理；请将 clamp/mirror 纹理烘焙后导出")
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        if isinstance(exc, ConversionError):
            raise
        raise ConversionError("GLB 结构无效") from exc
    return header


def _srgb_to_linear(rgb):
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(rgb):
    rgb = np.clip(rgb, 0, 1)
    return np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * rgb ** (1 / 2.4) - 0.055)


def _subdivide(vertices, faces, attributes):
    """Split all faces together; share midpoints and interpolate UV/vertex data."""
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    unique, inverse = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True)
    a, b, c = (inverse + len(vertices)).reshape(3, -1)
    new_faces = np.vstack((np.column_stack((faces[:, 0], a, c)),
                           np.column_stack((a, faces[:, 1], b)),
                           np.column_stack((c, b, faces[:, 2])),
                           np.column_stack((a, b, c))))
    new_vertices = np.vstack((vertices, vertices[unique].mean(axis=1)))
    attrs = {key: np.vstack((value, value[unique].mean(axis=1))) for key, value in attributes.items()}
    return new_vertices, new_faces, attrs


def _texture_samples(image, uv):
    """Bilinear sampling. trimesh uses bottom-left UVs after loading glTF."""
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    height, width = pixels.shape[:2]
    # GL sampling uses texel centres and wraps the neighbours across seams.
    x = (uv[..., 0] % 1) * width - 0.5
    y = ((1 - uv[..., 1]) % 1) * height - 0.5
    ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - ix)[..., None], (y - iy)[..., None]
    linear = _srgb_to_linear(pixels)
    top = linear[iy % height, ix % width] * (1 - fx) + linear[iy % height, (ix + 1) % width] * fx
    bottom = linear[(iy + 1) % height, ix % width] * (1 - fx) + linear[(iy + 1) % height, (ix + 1) % width] * fx
    return top * (1 - fy) + bottom * fy


def _mesh_colours(mesh, transform, refinement):
    vertices = trimesh.transform_points(mesh.vertices, transform)
    faces = np.array(mesh.faces, copy=True)
    attrs = {}
    visual = mesh.visual
    material = getattr(visual, "material", None)
    image = getattr(material, "baseColorTexture", None)
    if image is None:
        image = getattr(material, "image", None)
    factor = getattr(material, "baseColorFactor", None)
    factor = np.ones(3) if factor is None else np.asarray(factor[:3], dtype=float) / 255
    if image is not None:
        uv = getattr(visual, "uv", None)
        if uv is None or len(uv) != len(vertices) or not np.isfinite(uv).all():
            raise ConversionError("纹理缺少有效的 UV 坐标")
        if image.width * image.height > 16_777_216:
            raise ConversionError("纹理不能超过 1600 万像素")
        attrs["uv"] = np.asarray(uv)
    elif visual.kind == "vertex":
        attrs["rgb"] = np.asarray(visual.vertex_colors[:, :3], dtype=float) / 255
    elif "color" in getattr(visual, "vertex_attributes", {}):
        colour = np.asarray(visual.vertex_attributes["color"])
        attrs["rgb"] = colour[:, :3].astype(float)
        if colour.dtype.kind in "ui":
            attrs["rgb"] /= np.iinfo(colour.dtype).max

    face_rgb = None
    if visual.kind == "face":
        face_rgb = np.asarray(visual.face_colors[:, :3], dtype=float) / 255
    for _ in range(refinement if attrs else 0):
        vertices, faces, attrs = _subdivide(vertices, faces, attrs)
        if face_rgb is not None:
            face_rgb = np.tile(face_rgb, (4, 1))

    if "uv" in attrs:
        # Four subtriangle centres capture details that a single centroid misses.
        bary = np.array([[2/3, 1/6, 1/6], [1/6, 2/3, 1/6], [1/6, 1/6, 2/3], [1/3, 1/3, 1/3]])
        uv = np.einsum("si,fij->fsj", bary, attrs["uv"][faces])
        rgb = _linear_to_srgb(_texture_samples(image, uv).mean(axis=1) * factor)
    elif "rgb" in attrs:
        # glTF vertex colours and material factors are linear, unlike textures.
        rgb = _linear_to_srgb(attrs["rgb"][faces].mean(axis=1) * factor)
    elif face_rgb is not None:
        rgb = face_rgb
    else:
        rgb = np.tile(_linear_to_srgb(factor), (len(faces), 1))
    if np.linalg.det(transform[:3, :3]) < 0:
        faces = faces[:, ::-1]
    return vertices, faces, rgb


def _palette(rgb, areas, count):
    """Deterministic, area-weighted k-means in perceptual Lab colour space."""
    # Bin to 5-bit sRGB for bounded work, preserving the weighted true means.
    keys = np.floor(np.clip(rgb, 0, 1) * 31).astype(np.int32)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    weight = np.bincount(inv, weights=areas)
    samples = np.column_stack([np.bincount(inv, weights=areas * rgb[:, c]) / weight for c in range(3)])
    lab = cv2.cvtColor(samples.astype(np.float32)[None], cv2.COLOR_RGB2LAB)[0]
    count = min(count, len(samples))
    centres = [lab[np.argmax(weight)]]
    for _ in range(1, count):
        dist = np.min(np.sum((lab[:, None] - np.array(centres)[None]) ** 2, axis=2), axis=1)
        centres.append(lab[np.argmax(dist * weight)])
    centres = np.asarray(centres)
    for _ in range(24):
        labels = np.argmin(np.sum((lab[:, None] - centres[None]) ** 2, axis=2), axis=1)
        new = centres.copy()
        for i in range(count):
            selected = labels == i
            if selected.any():
                new[i] = np.average(lab[selected], axis=0, weights=weight[selected])
        if np.allclose(new, centres, atol=0.01):
            centres = new
            break
        centres = new
    labels = np.argmin(np.sum((lab[:, None] - centres[None]) ** 2, axis=2), axis=1)
    used, remap = np.unique(labels, return_inverse=True)
    palette = cv2.cvtColor(centres[used][None].astype(np.float32), cv2.COLOR_LAB2RGB)[0]
    palette = np.rint(np.clip(palette, 0, 1) * 255).astype(np.uint8)
    # Rounding can collapse nearby centres; coalesce their filament slots too.
    palette, unique_labels = np.unique(palette, axis=0, return_inverse=True)
    return palette, unique_labels[remap][inv]


def _paint(slot):
    """Bambu leaf encoding for a 1-based filament slot (limited to 16)."""
    if slot < 1 or slot > 16:
        raise ConversionError("颜色数量必须在 1 到 16 之间")
    return ("4", "8")[slot - 1] if slot <= 2 else f"{slot - 3:X}C"


def _xml(element):
    return ET.tostring(element, encoding="utf-8", xml_declaration=True)


def _write_3mf(vertices, faces, palette, labels, metadata):
    def sub(parent, tag, **attrs):
        return ET.SubElement(parent, f"{{{CORE}}}{tag}", {k: str(v) for k, v in attrs.items()})

    model = ET.Element(f"{{{CORE}}}model", {"unit": "millimeter", "xml:lang": "en-US", "xmlns:BambuStudio": BAMBU})
    # Bambu identifies projects by this compatibility string, not the 3MF
    # version metadata. Keep real generator/provenance explicit alongside it.
    for key, value in {"Application": "BambuStudio-02.00.00.00", "Title": "NailFree accessory",
                       "Description": "Generated by NailFree; Bambu-compatible colour project; slice before printing.",
                       "NailFree:Generator": "NailFree color-3mf/1", "BambuStudio:3mfVersion": "1",
                       "BambuStudio:MmPaintingVersion": "0"}.items():
        sub(model, "metadata", name=key).text = value
    resources = sub(model, "resources")
    group = ET.SubElement(resources, f"{{{MATERIAL}}}colorgroup", {"id": "1"})
    hexes = ["#" + "".join(f"{int(c):02X}" for c in colour) for colour in palette]
    for colour in hexes:
        ET.SubElement(group, f"{{{MATERIAL}}}color", {"color": colour + "FF"})
    obj = sub(resources, "object", id=2, type="model", name="NailFree accessory", pid=1, pindex=0)
    mesh = sub(obj, "mesh")
    vertex_el = sub(mesh, "vertices")
    for x, y, z in vertices:
        sub(vertex_el, "vertex", x=f"{x:.9g}", y=f"{y:.9g}", z=f"{z:.9g}")
    triangles = sub(mesh, "triangles")
    for face, label in zip(faces, labels):
        sub(triangles, "triangle", v1=face[0], v2=face[1], v3=face[2], pid=1,
            p1=label, p2=label, p3=label, paint_color=_paint(int(label) + 1))
    sub(sub(model, "build"), "item", objectid=2)

    config = ET.Element("config")
    config_obj = ET.SubElement(config, "object", {"id": "2"})
    ET.SubElement(config_obj, "metadata", {"key": "name", "value": "NailFree accessory"})
    ET.SubElement(config_obj, "metadata", {"key": "extruder", "value": "1"})
    part = ET.SubElement(config_obj, "part", {"id": "0", "subtype": "normal_part"})
    ET.SubElement(part, "metadata", {"key": "name", "value": "NailFree accessory"})
    ET.SubElement(part, "metadata", {"key": "extruder", "value": "1"})
    ET.SubElement(part, "mesh", {"firstid": "0", "lastid": str(len(faces) - 1)})
    plate = ET.SubElement(config, "plate")
    for key, value in {"plater_id": "1", "plater_name": "NailFree", "locked": "false"}.items():
        ET.SubElement(plate, "metadata", {"key": key, "value": value})
    instance = ET.SubElement(plate, "model_instance")
    for key, value in {"object_id": "2", "instance_id": "0", "identify_id": "1"}.items():
        ET.SubElement(instance, "metadata", {"key": key, "value": value})
    settings = {"filament_colour": hexes, "filament_type": ["PLA"] * len(hexes),
                "filament_diameter": ["1.75"] * len(hexes),
                "filament_is_support": ["0"] * len(hexes),
                # Unbound preset IDs: users select their actual printer and
                # materials before slicing. Required even for CLI model import.
                "printer_settings_id": "", "print_settings_id": "",
                "filament_settings_id": [""] * len(hexes), "printer_model": "",
                "nozzle_diameter": ["0.4"], "printable_height": "256",
                "printable_area": ["0x0", "256x0", "256x256", "0x256"]}
    types = b'''<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
<Default Extension="config" ContentType="application/octet-stream"/>
<Default Extension="json" ContentType="application/json"/>
</Types>'''
    rels = b'''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>'''
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", types)
        archive.writestr("_rels/.rels", rels)
        archive.writestr("3D/3dmodel.model", _xml(model))
        archive.writestr("Metadata/model_settings.config", _xml(config))
        archive.writestr("Metadata/project_settings.config", json.dumps(settings))
        archive.writestr("Metadata/nailfree.json", json.dumps(metadata, ensure_ascii=False))
    return output.getvalue(), hexes


def convert_glb_to_color_3mf(data: bytes, color_count=4, refinement=1, size_mm=None) -> Color3MF:
    if type(color_count) is not int or not 1 <= color_count <= 16:
        raise ConversionError("颜色数量必须是 1 到 16 的整数")
    if type(refinement) is not int or not 0 <= refinement <= 2:
        raise ConversionError("细节等级必须是 0、1 或 2")
    if size_mm is not None and (isinstance(size_mm, bool) or not isinstance(size_mm, (float, int))
                                or not np.isfinite(size_mm) or not 1 <= size_mm <= 300):
        raise ConversionError("模型最长边必须在 1 到 300 毫米之间")
    header = _validate_glb(data)
    try:
        scene = trimesh.load_scene(io.BytesIO(data), file_type="glb", process=False)
    except Exception as exc:
        raise ConversionError("GLB 解析失败，请确认模型包含有效的网格和内嵌纹理") from exc
    nodes = list(scene.graph.nodes_geometry)
    total_faces = sum(len(scene.geometry[scene.graph[node][1]].faces) for node in nodes)
    if not total_faces or total_faces > MAX_FACES:
        raise ConversionError(f"模型必须包含 1 到 {MAX_FACES} 个三角面")
    warnings = ["颜色为有限耗材色的近似；透明、金属、粗糙度及法线效果不会转成打印材质。"]
    if header.get("animations") or header.get("skins"):
        raise ConversionError("请先将动画或骨骼模型烘焙为静态网格")
    actual_refinement = refinement
    while total_faces * 4 ** actual_refinement > MAX_FACES:
        actual_refinement -= 1
    if actual_refinement < refinement:
        warnings.append("为限制模型大小，已自动降低细节等级。")
    vertices, faces, colours = [], [], []
    offset = 0
    for node in nodes:
        transform, name = scene.graph[node]
        mesh = scene.geometry[name]
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
            continue
        v, f, rgb = _mesh_colours(mesh, transform, actual_refinement)
        if not np.isfinite(v).all() or not np.isfinite(rgb).all():
            raise ConversionError("模型包含无效坐标或颜色")
        vertices.append(v)
        faces.append(f + offset)
        colours.append(rgb)
        offset += len(v)
    v, f, rgb = np.vstack(vertices), np.vstack(faces), np.vstack(colours)
    extent = np.ptp(v, axis=0)
    longest = float(extent.max())
    if longest <= 0:
        raise ConversionError("模型尺寸为零")
    # glTF uses metres; explicit longest-edge sizing is useful for AI accessories.
    v = (v - v.min(axis=0)) * (float(size_mm) / longest if size_mm is not None else 1000)
    areas = np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1) / 2
    valid = areas > max(float(areas.max()) * 1e-12, 1e-16)
    if not valid.any():
        raise ConversionError("模型没有有效的三角面")
    if not valid.all():
        warnings.append("已移除退化三角面。")
        f, rgb, areas = f[valid], rgb[valid], areas[valid]
    palette, labels = _palette(rgb, areas, color_count)
    if len(palette) == 1:
        warnings.append("当前模型仅提取到一种颜色；无法从无色几何恢复原始配色。")
    dimensions = np.ptp(v, axis=0).tolist()
    if max(dimensions) > 300:
        warnings.append("模型最长边超过 300mm，请在导出时设置打印尺寸。")
    metadata = {"generator": "NailFree", "version": 1, "requested_colors": color_count,
                "actual_colors": len(palette), "refinement": actual_refinement,
                "size_mm": dimensions, "warnings": warnings, "requires_slicing": True}
    archive, hexes = _write_3mf(v, f, palette, labels, metadata)
    return Color3MF(archive, hexes, len(f), dimensions, warnings)
