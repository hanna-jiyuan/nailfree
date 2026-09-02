# Cowork fastapi-only scaffold

按 ai-demo-platform-guard-transform-skill / fastapi-only profile 的规范产物。

## 开发

```sh
pip install -r requirements.txt
python -m uvicorn app:app --reload --port 3000
```

## 部署

```sh
bash install.sh   # 创建 .venv 并装依赖
bash start.sh     # 启动 uvicorn (0.0.0.0:${APP_PORT:-3000})
bash health.sh    # 探活
```

或者用 `cowork.publish` 自动 pack + 上传 + 部署。
