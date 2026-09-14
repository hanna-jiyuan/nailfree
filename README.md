# Cowork fastapi-only scaffold

按 ai-demo-platform-guard-transform-skill / fastapi-only profile 的规范产物。

NailFree：上传手部照片生成美甲效果图，框选配饰生成 GLB，并导出 STL 或保留有限色分区的 3MF。

## 彩色 3MF

在 3D 结果区域设置颜色数、最长边尺寸和图案细节，点击“下载彩色 3MF”。转换由 Python 后端完成，无需安装 Bambu Studio。下载后在 Bambu Studio 中作为项目打开，确认实际打印机和耗材配置，再切片。

详见 [彩色 3MF 使用、限制及验证](docs/color-3mf.md)。

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
