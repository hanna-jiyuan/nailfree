# 彩色 3MF 导出

GLB → 基础颜色采样 → 有限色调色板 → Bambu 涂色 3MF。

转换完全在 Python 后端运行，不需要安装 Bambu Studio，不调用 AI 或 Bambu 云服务。它是独立实现，不包含 Bambu Studio 的 C++ 算法源码，也不是官方 TextureToColor 模块的 Python 绑定。

## 网页使用

生成 3D 配饰后，在“彩色 3MF 打印设置”里选择最多颜色数（2/4/8/16）、最长边尺寸（默认 10mm）和图案细节，点击“下载彩色 3MF”。完成后页面显示实际调色板和模型尺寸；模型本身颜色较少时，不会强行增加颜色。

在 Bambu Studio 中**作为项目打开**下载的文件，保留项目中的耗材颜色。选择实际打印机、喷嘴和每个色槽对应的耗材，确认尺寸与摆放方向，再切片和检查预览。此文件没有 G-code，不能直接发送给打印机执行。导出中的 PLA、0.4mm 喷嘴及 256mm 工作空间是兼容性占位参数，打印机和工艺预设 ID 为空，必须替换为实际配置。

STL 下载仍只用于无颜色几何。界面中的打印尺寸和颜色设置只应用于彩色 3MF；STL 维持原有模型数值单位。原始 GLB 可继续下载用于预览或重新配色。

## 转换方式

1. 检查 GLB 2.0 容器，仅加载内嵌资源。
2. 遍历场景节点并应用变换，保留重复实例和镜像部件。
3. 按细节等级细分三角面并插值 UV，最多输出 20 万个面。纹理使用双线性、多点采样；正确处理 glTF / trimesh 的 UV 方向以及 sRGB / 线性颜色乘法。
4. 使用按面片面积加权的 Lab 空间 K-means 将颜色归并到指定数量，保持结果确定性。
5. 将有限色同时写入标准 3MF `m:colorgroup` 和 Bambu 面片 `paint_color` 编码，并写入匹配的耗材调色板与模型、盘配置。

这是基础颜色的有限色近似，未实现官方算法的区域平滑、复杂网格修复或多色混合预测。它不会自动创建可打印的内部分色实体，不保证与官方 TextureToColor 输出逐面一致。具体打印分区由切片软件根据表面涂色生成。

## 输入和限制

- 支持静态三角网格、PBR 基础颜色纹理与颜色因子、顶点颜色、多个网格与材质，以及场景实例。
- 每个 GLB 最大 50MiB、输入/输出最多 20 万面、单纹理最多 1600 万像素。超出细分预算时会降低细节等级并提示。每个后端进程同时处理最多两项彩色转换。
- 颜色数 API 范围为 1–16。没有颜色的模型只能生成单色，不能恢复已丢失的颜色。
- 透明度、金属度、粗糙度、法线和自发光效果不会被还原为打印材质；不保证纹理渐变无损。小图案受到网格细度、色数和打印尺寸限制。
- 暂不支持外部纹理/缓冲区、Draco 等必需扩展、动画/骨骼/变形、UV1、纹理变换扩展、clamp/mirror 采样，以及纹理和顶点色同时叠加。此类输入会明确报错；请先烘焙为内嵌基础颜色贴图。
- GLB 默认以米为单位；API 未指定 `size_mm` 时转为毫米。指定后按最长边等比缩放，并将最小坐标移到原点。不会自动旋转模型或修复非流形网格。
- Bambu 导入使用 `Application=BambuStudio-02.00.00.00` 作为兼容性标记，否则其 CLI 会丢弃项目调色板。实际生成器在 `NailFree:Generator`、Description 及 `Metadata/nailfree.json` 中明确记录，文件不是由 Bambu 官方程序生成。

## API

沿用 `POST /api/convert-format`，要求原有 SSO 身份头。彩色转换接收 `glb_base64`，不由服务器下载任意网址。

```json
{
  "format": "3mf",
  "glb_base64": "<GLB 的 Base64，可带 data URI 前缀>",
  "color_count": 4,
  "refinement": 1,
  "size_mm": 10
}
```

`refinement` 为 0、1 或 2，分别表示不细分、最多细分一次或两次。成功响应为 `model/3mf` 文件，包含 `X-Color-Palette`、`X-Model-Size-Mm` 和 JSON 格式的 `X-Conversion-Warnings` 响应头。参数无效返回 400、文件过大返回 413、并发繁忙返回 429。

## 验证

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

自动化测试覆盖纹理方向、基础颜色因子、顶点色、场景实例与镜像、尺寸、色数、确定性、输入错误、认证和下载接口。可选 Bambu 读写测试用于验证软件兼容性，不是运行时依赖：

```sh
BAMBU_STUDIO_BIN=/Applications/BambuStudio.app/Contents/MacOS/BambuStudio \
  python -m pytest -q tests/test_color_3mf.py -k bambu_roundtrip
```

生成一个不调用 AI 的三色验证样例：

```sh
python -c 'import runpy; runpy.run_path("tests/test_color_3mf.py", run_name="__main__")'
```

输出位于 `artifacts/color-3mf/reference.glb` 和 `reference.3mf`。已在 macOS 上用 Bambu Studio 2.8.2.61 对 1、3、16 色样例完成命令行导入/再次导出，确认调色板、面片涂色和尺寸保持一致；完整测试共 24 项通过。这不能代替真实生成模型的切片预览和实物打印检查。

## 参考

- [Bambu TextureToColor 接口与算法说明](https://github.com/bambulab/BambuStudio/tree/master/src/libslic3r/TextureToColor)
- [Bambu 3MF 读写格式](https://github.com/bambulab/BambuStudio/blob/master/src/libslic3r/Format/bbs_3mf.cpp)
- [Bambu 面片色槽编码](https://github.com/bambulab/BambuStudio/blob/master/src/libslic3r/Model.cpp)
- [glTF 2.0 规范](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html)

上述链接用于算法概念、文件协议和兼容性核查；本实现没有复制官方算法源码。若以后改成直接链接或移植官方实现，需要重新评估 AGPL 及各依赖的许可证要求。
