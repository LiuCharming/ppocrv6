# PP-OCRv6 服务

## Windows 本地 CPU 启动

安装 Python 3.11 后，在项目目录执行 PowerShell 命令：

```powershell
python -m venv .venv-cpu
.\.venv-cpu\Scripts\python.exe -m pip install -r requirements-cpu.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
powershell -ExecutionPolicy Bypass -File .\start-cpu.ps1
```

本地模式使用 CPU，默认监听 `127.0.0.1:8001`，模型从 BOS 下载。
接口：`http://127.0.0.1:8001/v1/ocr`；文档：`http://127.0.0.1:8001/docs`。
停止前台服务按 Ctrl+C。更换端口可使用 `start-cpu.ps1 -Port 8002`。
CPU 默认 2 路并发、16 个等待位；例如 4 路并发可使用 `start-cpu.ps1 -Concurrency 4 -QueueSize 32`。
本地启动脚本不读取 Docker 的 `.env`，默认无需密钥；需要时在启动前设置 `$env:API_KEY`。

```powershell
curl.exe http://127.0.0.1:8001/healthz
curl.exe -X POST http://127.0.0.1:8001/v1/ocr -F "file=@C:/images/test.jpg"
```

## Docker GPU 部署

单个 Docker 容器运行 PP-OCRv6_medium 和 HTTP API，直接上传图片，返回 JSON 全文、逐行文字、置信度和坐标。

## 部署

默认环境：Linux x86_64、NVIDIA GPU、支持 CUDA 11.8 的驱动、Docker Compose v2、NVIDIA Container Toolkit。
镜像使用 Python 3.11、paddlepaddle-gpu 3.2.0（cu118）、paddleocr/paddlex 3.7.0。
若 GPU 架构需要更新 CUDA，按 Paddle 官方兼容表调整 Dockerfile.model 的 Paddle 版本和 wheel 源。

默认通过 DaoCloud 国内代理拉取基础镜像，进入本目录执行：

```bash
docker run --rm --gpus all m.daocloud.io/docker.io/nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
cp .env.example .env
# 可直接使用默认配置；MODEL_API_KEY 可不填
docker compose config --quiet
docker compose up -d --build --remove-orphans
docker compose logs -f ocr-model
```

`--remove-orphans` 清理同一 Compose 项目此前的业务容器（如已部署旧版）。
沿用旧 `.env` 时可删除 `API_KEY`、`API_PORT`、`MODEL_TIMEOUT_SECONDS`，远程访问需将 `MODEL_BIND` 改为 `0.0.0.0`。

首次启动优先从百度 BOS 下载模型，权重保存在 `model-cache` 卷。模型初始化完成后 API 才可用。
GPU 不可用时启动失败。默认使用宿主 GPU 0，可在 `.env` 修改 `GPU_ID`。

| 用途 | 默认地址 |
|---|---|
| OCR 接口 | `http://服务器IP:8001/v1/ocr` |
| Swagger 文档 | `http://服务器IP:8001/docs` |
| 健康检查 | `http://服务器IP:8001/healthz` |

同一 Compose 网络中的服务可调用 `http://ocr-model:8000/v1/ocr`。
默认监听所有网卡，不要求密钥。设置非空 `MODEL_API_KEY` 后启用 Bearer 密钥鉴权。
沿用旧配置且希望关闭鉴权时，删除 `.env` 中的 `MODEL_API_KEY` 或将其留空，再重建容器。

## 国内镜像配置

构建和运行已默认配置国内源：

| 下载内容 | 默认源 | 配置项 |
|---|---|---|
| Debian 系统包及安全更新 | 清华镜像 | `APT_MIRROR`、`APT_SECURITY_MIRROR` |
| Python 依赖 | 清华 PyPI 镜像 | `PIP_INDEX_URL` |
| Paddle GPU 安装包 | 飞桨官方国内 CUDA 11.8 源 | `PADDLE_INDEX_URL` |
| OCR 模型权重 | 百度 BOS 优先 | `PADDLE_PDX_MODEL_SOURCE=BOS` |
| Docker 基础镜像 | DaoCloud 国内代理 | `BASE_IMAGE` |

变量已写入 `.env.example`，Compose 中也有默认值。Paddle GPU 安装显式使用飞桨源，覆盖通用 pip 源，以保留正确的 CUDA 版本。
配置依据：[清华 PyPI](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)、[清华 Debian](https://mirrors.tuna.tsinghua.edu.cn/help/debian/)、[PaddleX 模型源配置](https://github.com/PaddlePaddle/PaddleX/blob/release/3.7/paddlex/utils/flags.py)。

### Docker Hub 连接被重置时

默认 `BASE_IMAGE=m.daocloud.io/docker.io/library/python:3.11-slim-bookworm`，使用 [DaoCloud 官方文档](https://github.com/DaoCloud/public-image-mirror)提供的镜像前缀方式，无需重启 Docker。
如果旧 `.env` 仍为 `BASE_IMAGE=python:3.11-slim-bookworm`，它会覆盖新默认值。先修改，再重新构建：

```bash
sed -i 's|^BASE_IMAGE=.*|BASE_IMAGE=m.daocloud.io/docker.io/library/python:3.11-slim-bookworm|' .env
docker compose build --pull ocr-model
docker compose up -d
```

若 `.env` 没有 `BASE_IMAGE` 行，使用 Compose 的默认值即可。第三方代理是否可达仍取决于服务器网络和服务状态，也可通过 `BASE_IMAGE` 配置自己的国内仓库。

### 可选：Docker 引擎镜像加速

基础镜像的拉取发生在 Dockerfile 执行之前，因此 apt/pip 换源不能解决 Docker Hub 连接失败。
如果希望保留原始 `BASE_IMAGE=python:3.11-slim-bookworm`，可在服务器 Docker 引擎配置加速器。
`deploy/daemon.json.example` 提供了阿里云加速配置模板。将其中 `YOUR_ACCELERATOR_ID` 替换为你在阿里云容器镜像服务控制台取得的地址，按账号和服务器适用条件使用；详见[阿里云官方说明](https://help.aliyun.com/zh/acr/user-guide/accelerate-the-pulls-of-docker-official-images)。

将模板中的 `registry-mirrors` 字段合并到服务器 `/etc/docker/daemon.json`，保留已有配置，尤其是 NVIDIA 的 `runtimes`。模板中的占位地址不能直接使用。修改后执行：

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
docker info
docker pull python:3.11-slim-bookworm
```

重启 Docker 会影响该服务器已有容器。加速器的可用范围与缓存覆盖以服务商为准。
如果你有国内私有镜像仓库，也可以将官方 Python 镜像同步到仓库，然后把 `.env` 中 `BASE_IMAGE` 改成该镜像的完整地址；私有仓库需先 `docker login`。
基础镜像应保持 Python 3.11 / Debian bookworm slim，并保留标准的 `debian.sources` 文件。

配置好后重新构建：

```bash
docker compose build --pull ocr-model
docker compose up -d --remove-orphans
```

## 调用

```bash
curl -X POST http://localhost:8001/v1/ocr \
  -F 'file=@./example.png'
```

仅在设置了 `MODEL_API_KEY` 时，调用需添加 `Authorization: Bearer 你的密钥` 请求头。

返回示例（文字、置信度和坐标仅为示例）：

```json
{
  "text": "第一行文本",
  "lines": [
    {"text": "第一行文本", "confidence": 0.99, "polygon": [[10, 10], [200, 10], [200, 40], [10, 40]]}
  ],
  "model": "PP-OCRv6_medium",
  "width": 1024,
  "height": 768
}
```

`text` 将 OCR 返回的文字区域以换行连接；`lines` 保留各区域信息。
坐标以左上角为原点，单位为像素，相对于 EXIF 方向校正后的图片；宽高也对应校正后的图片。
无文字返回 HTTP 200、`text: ""`、`lines: []`。复杂多栏文档不额外重排语义阅读顺序。

## 配置与限制

- `MODEL_SIZE`：`medium`（默认）、`small`、`tiny`。
- `MODEL_PORT`：宿主端口，默认 8001；`MODEL_BIND`：默认 `0.0.0.0`，仅本机访问可改成 `127.0.0.1`。
- `MODEL_API_KEY`：可选调用密钥，默认不设置；为空时关闭鉴权，非空时通过容器内的 `API_KEY` 启用鉴权。
- `MAX_IMAGE_MB`：默认 10 MiB；`MAX_IMAGE_PIXELS`：默认 2500 万。文件大小校验在 multipart 解析后执行，反向代理也应限制请求体大小。
- 接收 multipart `file`，支持单帧 PNG/JPEG/WEBP/BMP/TIFF，不接收 PDF 或图片 URL。
- `OCR_CONCURRENCY`：并发推理工作数，默认 2。每个工作有独立模型副本和专用线程，避免同时使用同一 Paddle Predictor。GPU 显存不够时改为 1，再逐步压测调大。
- `OCR_QUEUE_SIZE`：等待队列长度，默认 16。请求超过“正在推理数 + 等待队列”时返回 503 和 `Retry-After: 1`；排队内的请求会在模型空闲后处理。
- `/healthz` 返回当前并发数、排队数和队列容量，可用于负载均衡健康检查。
- 默认关闭额外的文档方向分类、文档矫正和文本行方向分类模型。
- 错误 JSON 使用 `detail`：400 无效图片、401 密钥错误、413 图片过大、415 不支持的格式、422 参数缺失、500 推理失败、503 GPU 繁忙。

修改配置后执行 `docker compose up -d --build`。

## 验证与运维

接口测试使用替身模型，无需 GPU：

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

覆盖响应格式、BGR 转换、鉴权、无文字结果、图片校验、大小限制及并发队列满载处理。
实际部署后上传已知文本图片核对结果，并用 `nvidia-smi` 确认推理进程。

```bash
docker compose exec ocr-model python -c "import paddle; print(paddle.__version__); print(paddle.is_compiled_with_cuda()); print(paddle.device.cuda.device_count())"
docker compose ps
docker compose logs --tail=100 ocr-model
docker compose down
```

`docker compose down` 保留模型缓存卷；添加 `-v` 会删除缓存。

官方参考：[PP-OCRv6](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PP-OCRv6/PP-OCRv6.md)、[Paddle GPU 安装](https://www.paddlepaddle.org.cn/install/quick)。
