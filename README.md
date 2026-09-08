# TRELLIS.2 图片生成体素 · Mac Studio

本分支只保留 **图片 → TRELLIS.2 三维生成 → 彩色体素 → VOX2**。
使用 Apple Silicon 的 PyTorch MPS，不需要 NVIDIA、CUDA、Conda 或 Xcode Metal 编译器。
原有 Web UI、训练、数据工具、GLB/纹理烘焙、网格处理和 CUDA 扩展已经移除。

## 安装和运行

在项目目录执行：

```bash
./setup.sh
```

安装脚本创建项目自己的 `.venv`。当前依赖版本针对原生 arm64 Python 3.14；本机已安装在 Homebrew 中。`requirements.lock.txt` 固定本机验证过的完整依赖版本。

批量处理：把 PNG/JPG/JPEG/WEBP 图片放进项目的 `vox/` 目录，然后：

```bash
./img_to_vox.sh
```

结果位于 `vox/vox/`，与旧批处理的 `输入文件夹/vox/` 规则一致。已有 `.vox` 自动跳过。
脚本可从任意工作目录启动；传入的相对路径相对于调用时的目录。

单张图片或指定文件夹：

```bash
./img_to_vox.sh -i /path/to/image.png -o /path/to/result.vox -h 128
./img_to_vox.sh --input_folder /path/to/images --skip --max_colors 128
./img_to_vox.sh --help
```

注意：`-h` 保留旧脚本语义，表示最大高度；帮助使用 `--help`。
同一输入文件夹不能存在同名但扩展名不同的图片，例如 `a.png` 和 `a.jpg`，以免覆盖输出。

每张图片生成：

- `名称.vox`：三维体素数据及调色板。
- `名称.palette.png`：调色板。
- `名称_preview_xy.png`：正面投影预览。
- `名称_pre.png`：模型实际使用的预处理图片。

这里的 `.vox` 是项目原有的 **VOX2 自定义格式**，不是 MagicaVoxel 的 `VOX ` 格式。
格式定义保留在 `vox_codec.ns`，Python 读写器是 `vox_io.py`。

## 参数与模型

默认用 512 推理流程，输出最大高度 256、最多 220 种颜色。
`-h` 只控制最终体素的高度，不改变神经网络推理分辨率。
为了适配 36 GB 统一内存，只加载 512 所需模型，按阶段移入 MPS；没有加载 1024/1536 模型。
颜色解码器仍然保留，因为它生成体素颜色和透明度；不进行贴图烘焙。

首次生成会从 Hugging Face 下载模型（约 10 GB，另有抠图模型），缓存于 `~/.cache/huggingface/`。
后续生成复用缓存。透明背景图片直接使用 alpha；不透明图片按需加载 BiRefNet 抠图。
同一批次复用模型，不需要每张图重新加载。

支持环境变量：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `SEED` | `0` | 随机种子 |
| `STEPS` | `12` | 三个生成阶段的采样步数，1–100 |
| `MAX_HEIGHT` | `256` | 最终 VOX Y 轴高度上限，1–1024 |
| `MAX_COLORS` | `220` | 色数上限，1–255 |
| `MATERIAL_MODE` | `color` | `color`、`auto`、`image`、`solid` |
| `TRELLIS_DEVICE` | `mps` | 本机默认 MPS；可设 `cpu` 排查问题 |
| `TRELLIS_MODEL` | `microsoft/TRELLIS.2-4B` | TRELLIS 权重目录或仓库 |
| `DINO_MODEL` | `camenduru/dinov3-vitl16-pretrain-lvd1689m` | 沿用原脚本的特征模型仓库 |
| `REMBG_MODEL` | `ZhengPeng7/BiRefNet` | 沿用原脚本的抠图模型仓库 |

例如：

```bash
SEED=42 ./img_to_vox.sh -i image.png -h 128
```

批量输入可在图片旁添加同名 `.conf`，覆盖该图片的输出参数：

```ini
# image.conf
max_height = 128
max_colors = 100
```

## 验证

```bash
.venv/bin/python -m unittest discover -s tests -v
```

回归测试比较 CPU/MPS 稀疏卷积与标准 dense Conv3d、不同长度序列的注意力、旋转位置编码，以及非立方体 VOX 的坐标和文件读回。每次实际转换还检查写出文件的尺寸及实心体素数量。

本机实测（2026-09-08）：Mac Studio M4 Max / 36 GB / macOS 27.0 / Python 3.14.3。
使用项目皇冠示例，`SEED=0`、12 步、最大高度 128：输出 **214 × 128 × 211**、**202,322** 个实心体素、220 色，文件 **324,491 bytes**。
最终批量运行退出码 0、`roundtrip=OK`；转换约 **96 秒**，包含缓存模型加载总共约 **148 秒**。
9 项回归测试全部通过，另已实际验证普通 RGB 图片的 MPS 抠图。
示例输入在 `vox/crown.webp`，结果在 `vox/vox/crown.vox`（`vox/` 已被 Git 忽略）。

Apple Silicon 适配思路参考 [trellis-mac](https://github.com/shivampkumar/trellis-mac)。本分支直接返回解码体素，没有引入其网格提取或纹理烘焙依赖。
TRELLIS.2 来源于 [Microsoft TRELLIS.2](https://github.com/microsoft/TRELLIS.2)，保留原 MIT LICENSE；模型按各模型仓库的许可使用。
