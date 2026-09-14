"""NailFree - AI 美甲试穿 Demo"""

from __future__ import annotations

import base64
import io
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Optional

import logging

import cv2
import httpx
import numpy as np
import trimesh
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
from starlette.concurrency import run_in_threadpool

from color_3mf import ConversionError, MAX_GLB_BYTES, convert_glb_to_color_3mf

logger = logging.getLogger(__name__)
_color_conversion_slots = threading.BoundedSemaphore(2)


def _parse_sso_user(decrypted_userinfo: Optional[str]) -> Optional[dict]:
    """从 Decrypted-Userinfo header 解 SSO 用户（latin-1 → JSON 两步）。"""
    if not decrypted_userinfo:
        return None
    try:
        fixed = decrypted_userinfo.encode("latin-1").decode("utf-8")
        data = json.loads(fixed)
    except Exception:
        return None
    return {
        "email": data.get("email") or data.get("workEmail"),
        "name": data.get("name") or data.get("displayName"),
        "userId": data.get("userId") or data.get("id"),
        "raw": data,
    }


def _require_user(decrypted_userinfo: Optional[str]) -> dict:
    """拿不到用户 → 401，Cowork Guard 自动跳 SSO 登录页。"""
    user = _parse_sso_user(decrypted_userinfo)
    if not user:
        raise HTTPException(status_code=401, detail="unauthenticated")
    return user


def _load_rai_api_key() -> str:
    """读取 RAI API Key。优先环境变量，其次文件。"""
    # 优先环境变量
    key = os.environ.get("RAI_API_KEY", "").strip()
    if key:
        return key
    # 其次：项目目录下的 .rai-api-key 文件
    key_file = Path(__file__).resolve().parent / ".rai-api-key"
    if not key_file.exists():
        raise RuntimeError("RAI API key not found: neither RAI_API_KEY env nor .rai-api-key file")
    return key_file.read_text().strip()


app = FastAPI(title="NailFree - AI 美甲试穿")
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))


@app.get("/", response_class=HTMLResponse)
def index(
    request: Request,
    decrypted_userinfo: Optional[str] = Header(None, alias="Decrypted-Userinfo"),
):
    """主页：上传手部照片 + 输入美甲风格 → 生成效果图"""
    user = _require_user(decrypted_userinfo)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"user": user},
    )


@app.post("/api/generate")
async def generate_nail(
    image: UploadFile = File(...),
    style: str = Form(...),
    decrypted_userinfo: Optional[str] = Header(None, alias="Decrypted-Userinfo"),
):
    """调用 gpt-image-2 生成美甲效果图"""
    user = _require_user(decrypted_userinfo)

    # 校验图片
    if image.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(400, "仅支持 JPEG/PNG/WebP 图片")
    image_bytes = await image.read()
    if len(image_bytes) > 5 * 1024 * 1024:
        raise HTTPException(400, "图片大小不能超过 5MB")

    # base64 编码
    b64 = base64.b64encode(image_bytes).decode("ascii")
    mime = image.content_type or "image/jpeg"

    # 构造 prompt：把用户的风格描述 + 上下文说明
    prompt = f"在这张手部照片上生成美甲效果。美甲风格要求：{style}。请保持手部姿势不变，只在指甲上应用描述的美甲设计，效果要自然逼真。"

    # 调用 gpt-image-2 图片编辑 API
    api_key = _load_rai_api_key()
    api_url = "https://maas.devops.rednote.life/openai/images/edits"

    # /images/edits 用 multipart/form-data 上传
    async with httpx.AsyncClient(timeout=180.0) as client:
        try:
            resp = await client.post(
                api_url,
                headers={"api-key": api_key},
                data={
                    "model": "gpt-image-2",
                    "prompt": prompt,
                    "n": "1",
                    "size": "1024x1024",
                    "quality": "medium",
                },
                files={
                    "image": ("hand.jpg", image_bytes, mime),
                },
            )
        except httpx.RequestError as e:
            raise HTTPException(502, f"调用 AI 服务失败：{e}")

    if resp.status_code != 200:
        # 如果 edits 端点不可用，fallback 到 generations（纯文生图，不传图片）
        try:
            fallback_url = "https://maas.devops.rednote.life/openai/images/generations"
            async with httpx.AsyncClient(timeout=180.0) as client2:
                resp = await client2.post(
                    fallback_url,
                    headers={
                        "Content-Type": "application/json",
                        "api-key": api_key,
                    },
                    json={
                        "model": "gpt-image-2",
                        "prompt": f"{prompt} 请生成一张逼真的手部美甲效果图。",
                        "n": 1,
                        "size": "1024x1024",
                        "quality": "medium",
                        "output_format": "jpeg",
                    },
                )
        except httpx.RequestError as e:
            raise HTTPException(502, f"调用 AI 服务失败：{e}")
        if resp.status_code != 200:
            raise HTTPException(502, f"AI 服务返回错误：{resp.status_code} - {resp.text[:500]}")

    data = resp.json()

    # 解析响应：可能返回 b64_json 或 url
    try:
        item = data["data"][0]
        if "b64_json" in item:
            result_b64 = item["b64_json"]
            result_image = f"data:image/jpeg;base64,{result_b64}"
        elif "url" in item:
            result_image = item["url"]
        else:
            raise HTTPException(502, "AI 响应格式异常：缺少图像数据")
    except (KeyError, IndexError, TypeError) as e:
        raise HTTPException(502, f"解析 AI 响应失败：{e} - 原始响应：{str(data)[:300]}")

    return JSONResponse({"image": result_image})


def cutout_accessory(image_bytes: bytes) -> bytes:
    """使用 GrabCut 算法抠出配饰主体，返回带透明通道的 PNG bytes。

    算法策略：
    - 图片边缘一圈（border 像素）标记为"确定背景"
    - 内部区域标记为"可能前景"
    - 跑 GrabCut 5 次迭代得到前景 mask
    - 对 mask 做形态学操作去噪 + 填空洞
    - 背景设为透明，输出 RGBA PNG

    如果抠图失败（异常 / 前景面积异常），返回 None 让调用方 fallback。
    """
    try:
        # 1. 加载图片
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            logger.warning("cutout: cv2.imdecode failed, image_bytes len=%d", len(image_bytes))
            return None

        h, w = img_bgr.shape[:2]
        if w < 20 or h < 20:
            logger.warning("cutout: image too small %dx%d", w, h)
            return None

        # 2. 初始化 GrabCut mask
        mask = np.full((h, w), cv2.GC_PR_FGD, dtype=np.uint8)  # 默认：可能前景

        # 边缘一圈作为"确定背景"
        border = max(2, min(h, w) // 10)  # 动态边框，至少 2px
        mask[:border, :] = cv2.GC_BGD       # 上
        mask[h - border:, :] = cv2.GC_BGD   # 下
        mask[:, :border] = cv2.GC_BGD       # 左
        mask[:, w - border:] = cv2.GC_BGD   # 右

        # 确保矩形区域合法（GrabCut 要求 mask 内有足够的前景/背景样本）
        rect = (border, border, w - 2 * border, h - 2 * border)
        if rect[2] < 5 or rect[3] < 5:
            logger.warning("cutout: rect too small after border: %s", rect)
            return None

        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)

        # 3. 跑 GrabCut
        cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model, iterCount=5, mode=cv2.GC_INIT_WITH_MASK)

        # 4. 提取前景 mask（GC_FGD=1, GC_PR_FGD=3 都算前景）
        fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

        # 5. 形态学操作：去噪 + 填空洞
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_med = np.ones((5, 5), np.uint8)
        # 开运算去小噪点
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        # 闭运算填小空洞
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_med, iterations=2)

        # 6. 检查前景面积占比
        total_pixels = h * w
        fg_pixels = np.count_nonzero(fg_mask)
        fg_ratio = fg_pixels / total_pixels

        if fg_ratio < 0.03 or fg_ratio > 0.97:
            logger.warning("cutout: foreground ratio abnormal: %.2f%% (threshold 3%%-97%%)", fg_ratio * 100)
            return None

        # 7. 构建 RGBA 图片
        img_rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
        img_rgba[:, :, 3] = fg_mask  # alpha 通道

        # 8. 编码 PNG
        success, buf = cv2.imencode('.png', img_rgba)
        if not success:
            logger.warning("cutout: cv2.imencode failed")
            return None

        return buf.tobytes()

    except Exception as e:
        logger.exception("cutout: GrabCut failed: %s", e)
        return None


@app.post("/api/generate-3d")
async def generate_3d(
    request: Request,
    decrypted_userinfo: Optional[str] = Header(None, alias="Decrypted-Userinfo"),
):
    """提交 HY-3D-3.1 图生3D 任务（框选配饰图片 base64）

    流程：解码 base64 → GrabCut 抠图去背景 → 重新编码 → 送 HY-3D
    抠图失败则 fallback 使用原图，不阻塞请求。
    """
    user = _require_user(decrypted_userinfo)

    body = await request.json()
    image_data = body.get("image", "")
    if not image_data:
        raise HTTPException(400, "缺少配饰图片数据")

    # 解析 base64 data URI → 纯 base64 → 原始 bytes
    if image_data.startswith("data:"):
        b64_str = image_data.split(",", 1)[1]
    else:
        b64_str = image_data

    raw_bytes = base64.b64decode(b64_str)

    # GrabCut 抠图：去背景（手指/皮肤），只保留配饰主体
    cutout_bytes = cutout_accessory(raw_bytes)
    if cutout_bytes is not None:
        logger.info("generate-3d: cutout succeeded, using cutout image")
        final_bytes = cutout_bytes
    else:
        logger.warning("generate-3d: cutout failed, fallback to original image")
        final_bytes = raw_bytes

    # 重新编码为 base64
    final_b64 = base64.b64encode(final_bytes).decode("ascii")

    api_key = _load_rai_api_key()
    api_url = "http://maas.devops.xiaohongshu.com/gateway-v2/v1/tasks"

    payload = {
        "model": "hy-3d-3.1",
        "input": {"image_base64": final_b64},
        "parameters": {
            "generate_type": "Normal",
            "enable_pbr": True,
            "face_count": 50000,
            "result_format": "GLB",
        },
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(
                api_url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                json=payload,
            )
        except httpx.RequestError as e:
            raise HTTPException(502, f"调用 3D 服务失败：{e}")

    if resp.status_code != 200:
        raise HTTPException(502, f"3D 服务返回错误：{resp.status_code} - {resp.text[:500]}")

    data = resp.json()
    task_id = data.get("id", "")
    if not task_id:
        raise HTTPException(502, f"3D 服务未返回任务ID：{resp.text[:300]}")

    return JSONResponse({"task_id": task_id, "status": data.get("status", "queued")})


@app.get("/api/check-3d")
async def check_3d_status(
    task_id: str = "",
    decrypted_userinfo: Optional[str] = Header(None, alias="Decrypted-Userinfo"),
):
    """轮询 HY-3D-3.1 任务状态"""
    if not task_id:
        return JSONResponse({"error": "missing task_id"}, status_code=400)

    try:
        api_key = _load_rai_api_key()
        api_url = f"http://maas.devops.xiaohongshu.com/gateway-v2/v1/tasks/{task_id}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(api_url, headers={"Authorization": f"Bearer {api_key}"})

        if resp.status_code != 200:
            return JSONResponse({"error": f"RAI API {resp.status_code}: {resp.text[:300]}"}, status_code=502)

        data = resp.json()
        status = data.get("status", "unknown")
        result = {"status": status, "task_id": task_id}

        if status == "completed":
            task_result = data.get("result", {})
            items = task_result.get("data", [])
            if not isinstance(items, list):
                items = [items] if items else []

            glb_url = ""
            preview_url = ""
            for item in items:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "glb":
                    glb_url = item.get("url", "")
                    if not preview_url:
                        preview_url = item.get("preview_image_url", "")
                elif not preview_url:
                    preview_url = item.get("preview_image_url", "")

            result["glb_url"] = glb_url
            result["preview_url"] = preview_url

        elif status == "failed":
            err = data.get("error", {})
            if isinstance(err, dict):
                result["error"] = err.get("message", "生成失败")
            else:
                result["error"] = str(err) or "生成失败"

        return JSONResponse(result)

    except Exception as e:
        import traceback
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()}, status_code=500)


@app.post("/api/convert-format")
async def convert_format(
    request: Request,
    decrypted_userinfo: Optional[str] = Header(None, alias="Decrypted-Userinfo"),
):
    """将 GLB 模型转换为 STL 或 3MF 格式用于 3D 打印"""
    user = _require_user(decrypted_userinfo)

    # Bound encoded model size before parsing a potentially large JSON body.
    raw_body = bytearray()
    async for chunk in request.stream():
        raw_body.extend(chunk)
        if len(raw_body) > MAX_GLB_BYTES * 4 // 3 + 8192:
            raise HTTPException(413, "GLB 文件不能超过 50MB")
    try:
        body = json.loads(raw_body)
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(400, "请求必须是有效的 JSON")
    if not isinstance(body, dict) or not isinstance(body.get("format"), str):
        raise HTTPException(400, "缺少有效的 format 参数")
    fmt = body["format"].lower()
    glb_base64 = body.get("glb_base64", "")
    glb_url = body.get("glb_url", "")

    if fmt not in ("stl", "3mf"):
        raise HTTPException(400, "format 参数必须为 stl 或 3mf")

    glb_bytes: bytes
    if not isinstance(glb_base64, str) or not isinstance(glb_url, str):
        raise HTTPException(400, "模型数据必须是字符串")
    if glb_base64:
        # 浏览器先下 GLB，再 base64 传给后端（因为 Cowork Pod 访问不了腾讯 CDN）
        try:
            if "," in glb_base64:
                glb_base64 = glb_base64.split(",", 1)[1]
            glb_bytes = base64.b64decode(glb_base64, validate=True)
        except Exception as e:
            raise HTTPException(400, f"glb_base64 解码失败：{e}")
    elif glb_url and fmt == "3mf":
        raise HTTPException(400, "彩色 3MF 转换请上传 glb_base64 数据")
    elif glb_url:
        # 兜底：后端直接下载（多数情况会失败，仅内网 URL 可用）
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.get(glb_url)
                if resp.status_code != 200:
                    raise HTTPException(502, f"下载 GLB 文件失败：HTTP {resp.status_code}")
                glb_bytes = resp.content
        except httpx.RequestError as e:
            raise HTTPException(502, f"后端下载 GLB 失败（Cowork Pod 无法访问外网 CDN），请前端先下载后 base64 传入：{e}")
    else:
        raise HTTPException(400, "必须提供 glb_base64 或 glb_url")

    if len(glb_bytes) > MAX_GLB_BYTES:
        raise HTTPException(413, "GLB 文件不能超过 50MB")

    if fmt == "3mf":
        if not _color_conversion_slots.acquire(blocking=False):
            raise HTTPException(429, "彩色转换繁忙，请稍后重试")
        try:
            result = await run_in_threadpool(
                convert_glb_to_color_3mf, glb_bytes,
                color_count=body.get("color_count", 4),
                refinement=body.get("refinement", 1),
                size_mm=body.get("size_mm"),
            )
        except ConversionError as exc:
            raise HTTPException(400, str(exc))
        except Exception:
            logger.exception("彩色 3MF 转换失败")
            raise HTTPException(500, "彩色 3MF 转换失败，请检查模型或降低细节等级")
        finally:
            _color_conversion_slots.release()
        return StreamingResponse(
            io.BytesIO(result.data), media_type="model/3mf",
            headers={
                "Content-Disposition": 'attachment; filename="accessory-color.3mf"',
                "X-Color-Palette": ",".join(result.palette),
                "X-Model-Size-Mm": ",".join(f"{v:.2f}" for v in result.size_mm),
                "X-Conversion-Warnings": json.dumps(result.warnings, ensure_ascii=True),
            },
        )

    # Convert using trimesh
    try:
        mesh = trimesh.load(io.BytesIO(glb_bytes), file_type="glb")
    except Exception as e:
        raise HTTPException(500, f"加载 GLB 模型失败：{e}")

    # Apply scene transforms and retain repeated instances before STL export.
    if isinstance(mesh, trimesh.Scene):
        geometries = list(mesh.geometry.values())
        if not geometries:
            raise HTTPException(500, "GLB 模型中没有找到几何体")
        mesh = mesh.to_mesh()

    # Export to requested format
    try:
        output = io.BytesIO()
        mesh.export(output, file_type="stl")
        mime = "application/sla"
        filename = "accessory-3d.stl"
        output.seek(0)
    except Exception as e:
        raise HTTPException(500, f"转换格式失败：{e}")

    return StreamingResponse(
        output,
        media_type=mime,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/whoami")
def whoami(
    decrypted_userinfo: Optional[str] = Header(None, alias="Decrypted-Userinfo"),
) -> JSONResponse:
    user = _require_user(decrypted_userinfo)
    return JSONResponse({"email": user["email"], "name": user["name"], "userId": user["userId"]})




@app.get("/api/debug/env")
async def debug_env():
    import sys
    result = {"python": sys.version}
    for mod in ("trimesh", "numpy", "manifold3d", "lxml", "cv2", "PIL"):
        try:
            m = __import__(mod)
            result[mod] = getattr(m, "__version__", "installed")
        except ImportError as e:
            result[mod] = f"MISSING: {e}"
    return result


@app.get("/api/debug/cv")
async def debug_cv():
    """验证 cv2 (OpenCV) 是否能正常 import 及版本信息。"""
    try:
        import cv2
        return {
            "cv2_version": cv2.__version__,
            "numpy_version": np.__version__,
            "ok": True,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/api/debug/fetch")
async def debug_fetch(url: str):
    """测试从 Cowork Pod 能不能下载腾讯 CDN 的 GLB。"""
    import os
    result = {"url": url, "https_proxy": os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")}
    # 直连
    try:
        async with httpx.AsyncClient(timeout=30.0, trust_env=False) as c:
            r = await c.get(url)
            result["direct"] = {"status": r.status_code, "content_length": len(r.content), "content_type": r.headers.get("content-type")}
    except Exception as e:
        result["direct"] = f"ERR: {type(e).__name__}: {e}"
    # 走 env 代理
    try:
        async with httpx.AsyncClient(timeout=30.0) as c:
            r = await c.get(url)
            result["env_proxy"] = {"status": r.status_code, "content_length": len(r.content), "content_type": r.headers.get("content-type")}
    except Exception as e:
        result["env_proxy"] = f"ERR: {type(e).__name__}: {e}"
    return result


@app.get("/api/debug/net-probe")
async def net_probe(url: str):
    import os
    result = {"env_proxies": {k: v for k, v in os.environ.items() if "proxy" in k.lower()}}
    proxies_to_try = [
        None,
        "http://127.0.0.1:3128",
        "http://sock-proxy.devops.xiaohongshu.com:3128",
        "http://proxy.devops.xiaohongshu.com:3128",
    ]
    for p in proxies_to_try:
        key = str(p)
        try:
            kwargs = {"timeout": 15.0, "trust_env": False}
            if p:
                kwargs["proxy"] = p
            async with httpx.AsyncClient(**kwargs) as c:
                r = await c.get(url)
                result[key] = {"status": r.status_code, "size": len(r.content)}
        except Exception as e:
            result[key] = f"ERR {type(e).__name__}: {str(e)[:100]}"
    return result
