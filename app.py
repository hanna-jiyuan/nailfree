"""NailFree - AI 美甲试穿 Demo"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates


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


@app.post("/api/generate-3d")
async def generate_3d(
    request: Request,
    decrypted_userinfo: Optional[str] = Header(None, alias="Decrypted-Userinfo"),
):
    """提交 HY-3D-3.1 图生3D 任务（框选配饰图片 base64）"""
    user = _require_user(decrypted_userinfo)

    body = await request.json()
    image_data = body.get("image", "")
    if not image_data:
        raise HTTPException(400, "缺少配饰图片数据")

    # 解析 base64 data URI → 纯 base64
    if image_data.startswith("data:"):
        b64_str = image_data.split(",", 1)[1]
    else:
        b64_str = image_data

    api_key = _load_rai_api_key()
    api_url = "http://maas.devops.xiaohongshu.com/gateway-v2/v1/tasks"

    payload = {
        "model": "hy-3d-3.1",
        "input": {"image_base64": b64_str},
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


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/whoami")
def whoami(
    decrypted_userinfo: Optional[str] = Header(None, alias="Decrypted-Userinfo"),
) -> JSONResponse:
    user = _require_user(decrypted_userinfo)
    return JSONResponse({"email": user["email"], "name": user["name"], "userId": user["userId"]})


