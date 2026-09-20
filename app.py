"""NailFree - AI 美甲试穿 Demo"""

from __future__ import annotations

import base64
import io
import json
import os
import tempfile
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

logger = logging.getLogger(__name__)


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

# ===== 文字需求转设计描述（Prompt 优化）=====

RAI_CHAT_URL = "https://maas.devops.xiaohongshu.com/openai/v1/chat/completions"
CHAT_MODEL = "qwen3.7-plus"

_OPTIMIZE_BASE = (
    "你是一位资深美甲设计师，擅长把用户的想法转化为专业、具体、可直接用于 AI 图像生成的"
    "美甲设计描述。你的输出将被用作图像生成模型的提示词，在用户的手部照片上渲染美甲效果。"
)

# 四种输入情况的 system prompt（有/无参考图 × 有/无描述词）
OPTIMIZE_SYSTEM_PROMPTS = {
    # 有参考图 + 有描述词：分析参考图 + 应用用户修改
    "ref+desc": (
        _OPTIMIZE_BASE
        + "\n本次任务：用户会上传一张美甲参考图，并给出修改要求。请：\n"
        "1. 仔细观察参考图，提取关键设计信息：底色与配色、图案主题、装饰元素（花朵/线条/钻饰/亮片等）、"
        "图案在指甲上的布局、甲型、质感光泽（哑光/亮面/猫眼/渐变/立体）。\n"
        "2. 将用户的修改要求精确应用到该设计上：用户要求改动的细节必须改，用户未提及的部分必须保持参考图原样。\n"
        "3. 严禁凭空添加参考图与用户要求中都不存在的元素。\n"
        "输出要求：一段 80~180 字的中文设计描述，涵盖整体风格、颜色、图案元素、质感与装饰细节；"
        "只输出描述正文，不要标题、编号、引号或 markdown 格式。"
    ),
    # 有参考图 + 无描述词：忠实还原参考图
    "ref-only": (
        _OPTIMIZE_BASE
        + "\n本次任务：用户上传了一张美甲参考图，未附加任何说明，希望复刻该设计。"
        "请忠实还原参考图，输出一段 80~180 字的中文设计描述，具体到 AI 图像生成模型可以直接据其还原该款式："
        "底色与配色、图案主题、装饰元素、布局、甲型、质感光泽。"
        "只输出描述正文，不要标题、编号、引号或 markdown 格式。"
    ),
    # 无参考图 + 有描述词：扩写为专业设计描述
    "desc-only": (
        _OPTIMIZE_BASE
        + "\n本次任务：用户没有参考图，只有一句简短的美甲想法。请把它扩写成专业的设计描述：\n"
        "1. 用户明确指定的元素（风格、颜色、图案、材质等）必须全部保留，不得替换或删除；\n"
        "2. 合理补充细节：配色方案、图案在指甲上的布局、质感（哑光/亮面/渐变/猫眼/闪粉）、"
        "装饰元素（钻饰/金属线/珍珠等）、整体氛围与适用场景；\n"
        "3. 补充要克制，与用户原意风格一致，不喧宾夺主。\n"
        "输出要求：一段 80~180 字的中文描述；只输出描述正文，不要标题、编号、引号或 markdown 格式。"
    ),
}


def _validate_image(upload: UploadFile, image_bytes: bytes) -> str:
    """校验图片类型与大小，返回 mime。"""
    if upload.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(400, "仅支持 JPEG/PNG/WebP 图片")
    if len(image_bytes) > 5 * 1024 * 1024:
        raise HTTPException(400, "图片大小不能超过 5MB")
    return upload.content_type or "image/jpeg"


async def _call_chat_llm(system_prompt: str, user_content) -> str:
    """调用 RAI 文本模型（qwen3.7-plus，支持视觉输入）。

    user_content: 纯字符串，或 OpenAI 多模态 content 数组（text + image_url）。
    """
    api_key = _load_rai_api_key()
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": 3000,  # 推理模型，需为 reasoning 留足空间
        "temperature": 0.5,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(RAI_CHAT_URL, headers={"api-key": api_key}, json=payload)
        except httpx.RequestError as e:
            raise HTTPException(502, f"调用 AI 优化服务失败：{e}")
    if resp.status_code != 200:
        raise HTTPException(502, f"AI 优化服务返回错误：{resp.status_code} - {resp.text[:300]}")
    try:
        content = (resp.json()["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError) as e:
        raise HTTPException(502, f"解析 AI 优化响应失败：{e}")
    if not content:
        raise HTTPException(502, "AI 优化返回了空结果，请重试")
    return content


@app.post("/api/optimize-prompt")
async def optimize_prompt(
    style: Optional[str] = Form(None),
    reference: Optional[UploadFile] = File(None),
    decrypted_userinfo: Optional[str] = Header(None, alias="Decrypted-Userinfo"),
):
    """一键优化：把用户的文字需求 / 参考图转化为专业设计描述。

    按输入内容分四种情况：
    - 有参考图 + 有描述词 → 分析参考图 + 应用用户修改要求
    - 有参考图 + 无描述词 → 忠实还原参考图
    - 无参考图 + 有描述词 → 扩写为专业设计描述
    - 无参考图 + 无描述词 → 400（无可优化内容）
    """
    _require_user(decrypted_userinfo)

    style_text = (style or "").strip()[:2000]
    has_desc = bool(style_text)

    ref_bytes, ref_mime = None, None
    if reference is not None and reference.filename:
        ref_bytes = await reference.read()
        ref_mime = _validate_image(reference, ref_bytes)
    has_ref = ref_bytes is not None

    if not has_ref and not has_desc:
        raise HTTPException(400, "请先输入美甲描述或上传参考图，再使用一键优化")

    if has_ref:
        ref_b64 = base64.b64encode(ref_bytes).decode("ascii")
        image_part = {
            "type": "image_url",
            "image_url": {"url": f"data:{ref_mime};base64,{ref_b64}"},
        }
        if has_desc:
            case = "ref+desc"
            user_content = [
                {"type": "text", "text": f"参考图美甲设计如上。用户修改要求：{style_text}"},
                image_part,
            ]
        else:
            case = "ref-only"
            user_content = [
                {"type": "text", "text": "请复刻参考图中的美甲设计。"},
                image_part,
            ]
    else:
        case = "desc-only"
        user_content = f"用户的美甲想法：{style_text}"

    optimized = await _call_chat_llm(OPTIMIZE_SYSTEM_PROMPTS[case], user_content)
    return JSONResponse({"prompt": optimized, "case": case})


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
    style: Optional[str] = Form(None),
    reference: Optional[UploadFile] = File(None),
    decrypted_userinfo: Optional[str] = Header(None, alias="Decrypted-Userinfo"),
):
    """调用 gpt-image-2 生成美甲效果图（支持可选参考图）"""
    user = _require_user(decrypted_userinfo)

    # 校验手部图片
    image_bytes = await image.read()
    mime = _validate_image(image, image_bytes)

    # 校验参考图（可选）
    style_text = (style or "").strip()[:2000]
    ref_bytes, ref_mime = None, None
    if reference is not None and reference.filename:
        ref_bytes = await reference.read()
        ref_mime = _validate_image(reference, ref_bytes)

    if not style_text and ref_bytes is None:
        raise HTTPException(400, "请填写风格描述或上传参考图")

    # 构造 prompt + 输入图片：有参考图时走多图编辑模式
    if ref_bytes is not None:
        prompt = (
            "第一张图是用户的手部照片，第二张图是美甲设计参考图。"
            "请将参考图中的美甲设计精准应用到第一张图所有指甲上："
            "保持手部姿势、手指形态、皮肤质感与背景完全不变，只改变指甲表面的美甲图案。"
            + (f"设计要求：{style_text}。" if style_text else "")
            + "效果自然逼真，美甲图案清晰精致。"
        )
        files = [
            ("image[]", ("hand.jpg", image_bytes, mime)),
            ("image[]", ("reference.jpg", ref_bytes, ref_mime)),
        ]
    else:
        prompt = (
            f"在这张手部照片上生成美甲效果。美甲风格要求：{style_text}。"
            "请保持手部姿势不变，只在指甲上应用描述的美甲设计，效果要自然逼真。"
        )
        files = {"image": ("hand.jpg", image_bytes, mime)}

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
                files=files,
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

    body = await request.json()
    fmt = body.get("format", "").lower()
    glb_base64 = body.get("glb_base64", "")
    glb_url = body.get("glb_url", "")

    if fmt not in ("stl", "3mf"):
        raise HTTPException(400, "format 参数必须为 stl 或 3mf")

    glb_bytes: bytes
    if glb_base64:
        # 浏览器先下 GLB，再 base64 传给后端（因为 Cowork Pod 访问不了腾讯 CDN）
        try:
            if "," in glb_base64:
                glb_base64 = glb_base64.split(",", 1)[1]
            glb_bytes = base64.b64decode(glb_base64)
        except Exception as e:
            raise HTTPException(400, f"glb_base64 解码失败：{e}")
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

    # Convert using trimesh
    try:
        mesh = trimesh.load(io.BytesIO(glb_bytes), file_type="glb")
    except Exception as e:
        raise HTTPException(500, f"加载 GLB 模型失败：{e}")

    # If it's a Scene, merge all meshes
    if isinstance(mesh, trimesh.Scene):
        geometries = list(mesh.geometry.values())
        if not geometries:
            raise HTTPException(500, "GLB 模型中没有找到几何体")
        mesh = trimesh.util.concatenate(geometries)

    # Export to requested format
    try:
        output = io.BytesIO()
        if fmt == "stl":
            mesh.export(output, file_type="stl")
            mime = "application/sla"
            filename = "accessory-3d.stl"
        else:
            mesh.export(output, file_type="3mf")
            mime = "model/3mf-binary"
            filename = "accessory-3d.3mf"
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
