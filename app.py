"""NailFree - AI 美甲试穿 Demo"""

from __future__ import annotations

import base64
import io
import json
import os
import tempfile
import asyncio
import time
import uuid
import threading
from pathlib import Path
from typing import Optional

import logging

import cv2
import httpx
import numpy as np
import trimesh
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
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


# ===== 文生 3D：描述 → TRELLIS.2 英文 3D prompt =====

_OPTIMIZE_3D_BASE = (
    "你是一位资深 3D 配饰设计师，擅长把用户的想法转化为英文 3D 生成提示词（用于 TRELLIS 文生3D 模型）。"
)

OPTIMIZE_3D_SYSTEM_PROMPTS = {
    # 有描述：中→英专业 3D prompt
    "desc-only": (
        _OPTIMIZE_3D_BASE
        + "\n本次任务：把用户的美甲配饰/立体饰品描述转化为一段专业的英文 3D 提示词。要求：\n"
        "1. 保留用户指定的全部元素（造型、颜色、材质、风格），不得替换或删除；\n"
        "2. 补充 3D 物体所需细节：整体形状与比例、表面纹理、光泽/哑光/金属感、装饰细节；\n"
        "3. 描述一个独立的立体实物（如蝴蝶结、花朵、宝石、立体摆件），不要出现指甲、手、佩戴等场景词；\n"
        "输出要求：纯英文，30~80 词，一句话或短段落；不要标题、引号或 markdown 格式。"
    ),
    # 有参考图 + 有描述：看图提取设计 + 应用修改
    "ref+desc": (
        _OPTIMIZE_3D_BASE
        + "\n本次任务：用户会上传一张美甲/配饰参考图，并给出修改要求。请：\n"
        "1. 观察参考图，把其中最有代表性的立体设计元素（花朵、蝴蝶结、钻饰、链条等）转化为一个可独立生成的 3D 物体；\n"
        "2. 将用户的修改要求精确应用到该设计上；未提及的部分保持参考图原样；\n"
        "3. 描述一个独立的立体实物，不要出现指甲、手、佩戴等场景词；\n"
        "输出要求：纯英文，30~80 词，一句话或短段落；不要标题、引号或 markdown 格式。"
    ),
    # 有参考图 + 无描述
    "ref-only": (
        _OPTIMIZE_3D_BASE
        + "\n本次任务：用户上传一张美甲/配饰参考图，未附加说明。请把图中最有代表性的立体设计元素"
        "转化为一个可独立生成的 3D 实物描述（形状、颜色、材质、纹理、装饰细节）；"
        "不要出现指甲、手、佩戴等场景词。输出纯英文，30~80 词；不要标题、引号或 markdown 格式。"
    ),
}


@app.post("/api/optimize-prompt")
async def optimize_prompt(
    style: Optional[str] = Form(None),
    target: Optional[str] = Form("2d"),
    reference: Optional[UploadFile] = File(None),
    decrypted_userinfo: Optional[str] = Header(None, alias="Decrypted-Userinfo"),
):
    """一键优化：把用户的文字需求 / 参考图转化为专业描述。

    target=2d（默认）：美甲效果图设计描述（中文，给 gpt-image-2 用）
    target=3d：独立 3D 配饰实物描述（英文，给 TRELLIS.2 文生3D 用）

    按输入内容分四种情况：
    - 有参考图 + 有描述词 → 分析参考图 + 应用用户修改要求
    - 有参考图 + 无描述词 → 忠实还原参考图
    - 无参考图 + 有描述词 → 扩写为专业描述
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
        raise HTTPException(400, "请先输入描述或上传参考图，再使用一键优化")

    is_3d = (target or "2d").strip().lower() in ("3d", "t3d", "text-to-3d")
    prompts = OPTIMIZE_3D_SYSTEM_PROMPTS if is_3d else OPTIMIZE_SYSTEM_PROMPTS

    if has_ref:
        ref_b64 = base64.b64encode(ref_bytes).decode("ascii")
        image_part = {
            "type": "image_url",
            "image_url": {"url": f"data:{ref_mime};base64,{ref_b64}"},
        }
        if has_desc:
            case = "ref+desc"
            if is_3d:
                text = f"参考图设计如上。用户修改要求：{style_text}"
            else:
                text = f"参考图美甲设计如上。用户修改要求：{style_text}"
            user_content = [{"type": "text", "text": text}, image_part]
        else:
            case = "ref-only"
            user_content = [
                {"type": "text", "text": "请转化为可独立生成的 3D 实物描述。" if is_3d else "请复刻参考图中的美甲设计。"},
                image_part,
            ]
    else:
        case = "desc-only"
        user_content = f"用户的想法：{style_text}"

    optimized = await _call_chat_llm(prompts[case], user_content)
    return JSONResponse({"prompt": optimized, "case": case, "target": "3d" if is_3d else "2d"})


# ============================================================
# TRELLIS.2 图生3D 服务（自建 8xH20 多卡网关）集成
# 网关同步返回 GLB；这里用后台任务 + 内存任务表，对接前端「提交+轮询」流程。
# ============================================================
TRELLIS2_BASE_URL = os.environ.get("TRELLIS2_BASE_URL", "http://10.142.3.181:8080").rstrip("/")
TRELLIS2_PIPELINE_TYPE = os.environ.get("TRELLIS2_PIPELINE_TYPE", "1024_cascade")  # 1024 质量显著更好(~28s)，512 更快但糊
TRELLIS2_TIMEOUT = float(os.environ.get("TRELLIS2_TIMEOUT", "900"))
_TRELLIS_DIR = Path(tempfile.gettempdir()) / "nailfree_trellis"
_TRELLIS_DIR.mkdir(parents=True, exist_ok=True)
_trellis_tasks: dict = {}          # task_id -> {status, glb_url, preview_url, error, ...}
_trellis_bg: dict = {}             # task_id -> asyncio.Task（持有引用防 GC）


def _prune_trellis_tasks(keep: int = 30):
    """限制内存任务数：删除最旧任务及其 GLB 临时文件。"""
    while len(_trellis_tasks) > keep:
        old = next(iter(_trellis_tasks))
        _trellis_tasks.pop(old, None)
        _trellis_bg.pop(old, None)
        try:
            (_TRELLIS_DIR / f"{old}.glb").unlink(missing_ok=True)
        except Exception:
            pass


async def _run_trellis_generations(
    task_id: str,
    text: str,
    image_uri: str,
    params: dict = None,
    mode: str = "image",
    preview_url: Optional[str] = None,
):
    """统一走 TRELLIS.2 /v1/images/generations（text+image），带重试并下载 GLB。"""
    params = params or {}
    label = "文生" if mode == "text" else "图生"
    data = {
        "text": text,
        "image": image_uri,
        "pipeline_type": str(params.get("pipeline_type") or TRELLIS2_PIPELINE_TYPE),
    }
    for _k in ("steps", "texture_size", "decimation_target", "seed"):
        _v = params.get(_k)
        if _v is not None and _v != "":
            data[_k] = str(_v)

    resp = None
    last_error = ""
    backoff = [3, 5, 8, 12, 20]
    for attempt in range(6):
        _trellis_tasks[task_id]["stage"] = (
            f"TRELLIS.2 {label} 3D 生成中..." if attempt == 0 else f"TRELLIS.2 重试中（{attempt + 1}/6）..."
        )
        try:
            async with httpx.AsyncClient(timeout=TRELLIS2_TIMEOUT, trust_env=False) as client:
                resp = await client.post(f"{TRELLIS2_BASE_URL}/v1/images/generations", json=data)
        except Exception as e:  # noqa: BLE001
            last_error = f"{type(e).__name__}: {e}"
            logger.warning("trellis generations attempt %s failed: %s", attempt + 1, last_error)
            resp = None
        else:
            if resp.status_code == 200:
                break
            last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            logger.warning("trellis generations attempt %s returned %s", attempt + 1, last_error)
        if attempt < len(backoff):
            await asyncio.sleep(backoff[attempt])

    if resp is None or resp.status_code != 200:
        _trellis_tasks[task_id] = {
            "status": "failed",
            "error": f"TRELLIS {label}3D 错误：{last_error or '服务不可用'}",
        }
        return

    try:
        info = (resp.json().get("data") or [{}])[0]
    except Exception as e:  # noqa: BLE001
        _trellis_tasks[task_id] = {"status": "failed", "error": f"TRELLIS 响应解析失败：{e}"}
        return
    glb_url = info.get("url")
    if not glb_url:
        _trellis_tasks[task_id] = {"status": "failed", "error": "TRELLIS 未返回 GLB 地址"}
        return

    async with httpx.AsyncClient(timeout=180.0, trust_env=False) as client:
        glb_resp = await client.get(glb_url)
    if glb_resp.status_code != 200:
        _trellis_tasks[task_id] = {"status": "failed", "error": f"GLB 下载失败 {glb_resp.status_code}"}
        return

    (_TRELLIS_DIR / f"{task_id}.glb").write_bytes(glb_resp.content)
    _trellis_tasks[task_id] = {
        "status": "completed",
        "glb_url": f"/api/trellis-glb/{task_id}",
        "preview_url": preview_url,
        "num_vertices": info.get("num_vertices"),
        "num_faces": info.get("num_faces"),
        "engine": "trellis2",
        "mode": mode,
    }


async def _run_trellis(task_id: str, image_bytes: bytes, preview_uri: str, params: dict = None):
    """图生3D：参考图 → TRELLIS.2 generations（网关 /to3d 目前不稳定，统一走 generations）。"""
    del image_bytes  # generations 直接吃 data URI；保留参数兼容调用方
    text = (params or {}).get("text") or (
        "Recreate the object in the reference image as a single high-quality 3D accessory, "
        "preserving its shape, colors, materials and fine details."
    )
    await _run_trellis_generations(task_id, text, preview_uri, params, mode="image", preview_url=None)


async def _run_trellis_text(task_id: str, prompt: str, params: dict = None):
    """文生3D 管线：文字描述 → gpt-image-2 概念图 → TRELLIS.2 generations（text+image）→ GLB。"""
    params = params or {}
    try:
        # 1) 文字 → 白底概念图（gpt-image-2）
        _trellis_tasks[task_id]["stage"] = "正在生成概念图（gpt-image-2）..."
        img_prompt = (
            f"Product render of: {prompt}. "
            "A single centered 3D object on a pure white background, studio lighting, "
            "high detail, no hands, no text, no watermark."
        )
        api_key = _load_rai_api_key()
        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(
                "https://maas.devops.rednote.life/openai/images/generations",
                headers={"api-key": api_key, "Content-Type": "application/json"},
                json={
                    "model": "gpt-image-2",
                    "prompt": img_prompt,
                    "n": 1,
                    "size": "1024x1024",
                    "quality": "medium",
                    "output_format": "jpeg",
                },
            )
        if resp.status_code != 200:
            _trellis_tasks[task_id] = {
                "status": "failed",
                "error": f"概念图生成失败 {resp.status_code}: {resp.text[:200]}",
            }
            return
        item = resp.json()["data"][0]
        if "b64_json" in item and item["b64_json"]:
            img_bytes = base64.b64decode(item["b64_json"])
        elif "url" in item and item["url"]:
            async with httpx.AsyncClient(timeout=60.0) as c2:
                img_bytes = (await c2.get(item["url"])).content
        else:
            _trellis_tasks[task_id] = {"status": "failed", "error": "概念图响应缺少图像数据"}
            return
        preview_uri = "data:image/jpeg;base64," + base64.b64encode(img_bytes).decode("ascii")

        # 2) 概念图 + 英文 prompt → TRELLIS.2 generations
        await _run_trellis_generations(
            task_id,
            prompt,
            preview_uri,
            params,
            mode="text",
            preview_url=preview_uri,
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("trellis text task %s failed", task_id)
        _trellis_tasks[task_id] = {"status": "failed", "error": f"调用 TRELLIS 文生3D 失败：{e}"}


async def _finish_trellis_task(task_id: str, resp):
    """处理网关同步返回的 GLB 响应，写入任务表。"""
    if resp.status_code != 200:
        _trellis_tasks[task_id] = {
            "status": "failed",
            "error": f"TRELLIS 服务错误 {resp.status_code}: {resp.text[:300]}",
        }
        return
    (_TRELLIS_DIR / f"{task_id}.glb").write_bytes(resp.content)
    _trellis_tasks[task_id] = {
        "status": "completed",
        "glb_url": f"/api/trellis-glb/{task_id}",
        "preview_url": None,
        "num_vertices": resp.headers.get("X-Num-Vertices"),
        "num_faces": resp.headers.get("X-Num-Faces"),
        "engine": "trellis2",
    }


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

    # 引擎选择：trellis2（自建多卡服务）或 hunyuan（HY-3D，默认）
    engine = (body.get("engine") or "hunyuan").lower()
    if engine in ("trellis2", "trellis", "trellis.2"):
        # trellis2 自带 RMBG-2.0 去背景，直接送原始框选图，跳过 GrabCut（避免双重去背景损质）
        preview_uri = image_data if image_data.startswith("data:") else ("data:image/png;base64," + b64_str)
        t2_params = {k: body.get(k) for k in ("pipeline_type", "steps", "texture_size", "decimation_target", "seed")}
        task_id = "t2-" + uuid.uuid4().hex
        _prune_trellis_tasks()
        _trellis_tasks[task_id] = {"status": "in_progress", "engine": "trellis2"}
        _trellis_bg[task_id] = asyncio.create_task(_run_trellis(task_id, raw_bytes, preview_uri, t2_params))
        return JSONResponse({"task_id": task_id, "status": "in_progress", "engine": "trellis2"})

    # 以下 hunyuan(HY-3D)：GrabCut 抠图去背景（手指/皮肤），只保留配饰主体
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


def _has_cjk(text: str) -> bool:
    """是否包含中文字符。TRELLIS generations 的 text 只接受英文，中文会触发网关 500。"""
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


async def _ensure_english_3d_prompt(prompt: str) -> str:
    """文生3D 前置：把中文描述转成英文 3D prompt，避免 TRELLIS 500。"""
    if not _has_cjk(prompt):
        return prompt
    optimized = await _call_chat_llm(
        OPTIMIZE_3D_SYSTEM_PROMPTS["desc-only"],
        f"用户的想法：{prompt}",
    )
    optimized = optimized.strip().strip('"').strip()
    if not optimized:
        raise HTTPException(502, "描述优化失败：AI 返回空结果")
    logger.info("generate-3d-text: optimized Chinese prompt to English: %s", optimized[:160])
    return optimized


@app.post("/api/generate-3d-text")
async def generate_3d_text(
    request: Request,
    decrypted_userinfo: Optional[str] = Header(None, alias="Decrypted-Userinfo"),
):
    """提交 TRELLIS.2 文生3D 任务（纯文字描述 → 3D 配饰模型）

    body: {prompt: str, pipeline_type?, steps?, texture_size?, decimation_target?, seed?}
    复用 t2- 任务表与 /api/check-3d 轮询。
    """
    _require_user(decrypted_userinfo)

    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(400, "请求必须是 JSON 对象")

    prompt = str(body.get("prompt") or "").strip()[:2000]
    if not prompt:
        raise HTTPException(400, "请输入 3D 配饰描述")

    # TRELLIS generations 的 text 必须为英文；中文输入先自动优化/翻译。
    prompt = await _ensure_english_3d_prompt(prompt)

    params = {k: body.get(k) for k in ("pipeline_type", "steps", "texture_size", "decimation_target", "seed")}
    task_id = "t2-" + uuid.uuid4().hex
    _prune_trellis_tasks()
    _trellis_tasks[task_id] = {"status": "in_progress", "engine": "trellis2", "mode": "text", "prompt": prompt}
    _trellis_bg[task_id] = asyncio.create_task(_run_trellis_text(task_id, prompt, params))
    return JSONResponse({
        "task_id": task_id,
        "status": "in_progress",
        "engine": "trellis2",
        "mode": "text",
        "optimized_prompt": prompt,
    })


@app.get("/api/check-3d")
async def check_3d_status(
    task_id: str = "",
    decrypted_userinfo: Optional[str] = Header(None, alias="Decrypted-Userinfo"),
):
    """轮询 HY-3D-3.1 任务状态"""
    if not task_id:
        return JSONResponse({"error": "missing task_id"}, status_code=400)

    # TRELLIS.2 任务：查内存任务表（后台任务写入）
    if task_id.startswith("t2-"):
        t = _trellis_tasks.get(task_id)
        if not t:
            return JSONResponse({"status": "failed", "task_id": task_id, "error": "任务不存在或已过期"})
        return JSONResponse({**t, "task_id": task_id})

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


@app.get("/api/trellis-glb/{task_id}")
async def trellis_glb(task_id: str):
    """同源提供 TRELLIS.2 生成的 GLB（避免浏览器直连 H20 网关的可达性问题）。"""
    if not task_id.startswith("t2-") or "/" in task_id or ".." in task_id:
        raise HTTPException(400, "bad task_id")
    p = _TRELLIS_DIR / f"{task_id}.glb"
    if not p.exists():
        raise HTTPException(404, "GLB 不存在或已过期")
    return FileResponse(str(p), media_type="model/gltf-binary", filename="accessory-3d.glb")


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


@app.post("/api/debug/trellis-post")
async def trellis_post(request: Request):
    """调试：向 TRELLIS.2 网关发 POST。body: {path, json?{...}, form?{...}, files?{name: dataURL}}"""
    body = await request.json()
    url = f"{TRELLIS2_BASE_URL}/{str(body.get('path', '')).lstrip('/')}"
    jbody = body.get("json")
    fdata = body.get("form") or None
    files = None
    if body.get("files"):
        files = {}
        for name, data_url in body["files"].items():
            header, _, b64 = str(data_url).partition(",")
            mime = "image/png"
            if "jpeg" in header or "jpg" in header:
                mime = "image/jpeg"
            files[name] = (f"{name}.png", base64.b64decode(b64), mime)
    try:
        async with httpx.AsyncClient(timeout=600.0, trust_env=False) as c:
            r = await c.post(url, json=jbody, data=fdata, files=files)
        ct = r.headers.get("content-type", "")
        out = {"url": url, "status": r.status_code, "content_type": ct, "len": len(r.content)}
        if "json" in ct:
            out["body"] = r.text[:1200]
        else:
            out["body_preview"] = r.content[:120]
        return out
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


@app.get("/api/debug/trellis-openapi")
async def trellis_openapi(detail: str = ""):
    """拉取 TRELLIS.2 网关的 openapi.json（调试用）。detail=<path> 时返回该路径完整定义。"""
    try:
        async with httpx.AsyncClient(timeout=15.0, trust_env=False) as c:
            r = await c.get(f"{TRELLIS2_BASE_URL}/openapi.json")
        if r.status_code != 200:
            return {"status": r.status_code, "body": r.text[:500]}
        spec = r.json()
        if detail:
            return {"detail": spec.get("paths", {}).get(detail)}
        paths = {}
        for p, ops in (spec.get("paths") or {}).items():
            paths[p] = [m.upper() for m in ops.keys() if m in ("get", "post", "put", "delete")]
        return {"status": 200, "title": spec.get("info", {}).get("title"), "paths": paths}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


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
