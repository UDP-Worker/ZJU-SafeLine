## 使用Docker进行部署

本指南用于将 `web_infer.py` 容器化并在其它设备部署。默认使用 CPU 推理，如需 GPU 请看文末说明。

### 0.下载和导入镜像
在本仓库的`release`中下载并导入镜像

```bash
gunzip -c mvopt-webinfer_latest.tar.gz | docker load
```

### 1. 准备模型与目录

建议创建一个目录用于持久化模型与上传文件，例如：

```
deploy/
  models/
    final_model.pt
  uploads/
```

### 2. docker-compose 部署

在 `deploy/` 下创建 `docker-compose.yml`，如需更改模型和上传文件的目录，请在`volumes`中做对应修改：

```yaml
services:
  webinfer:
    image: mvopt-webinfer:latest
    container_name: mvopt-webinfer
    ports:
      - "8000:8000"
    volumes:
      - ./models:/models:ro
      - ./uploads:/app/uploads
    command:
      - python
      - web_infer.py
      - --model
      - /models/final_model.pt
      - --device
      - auto
      - --interval
      - "0.25"
      - --height
      - "810"
      - --width
      - "892"
      - --threshold
      - "0.7"
      - --loop
      - --uploads-dir
      - /app/uploads
      - --cleanup-minutes
      - "60"
      - --cleanup-max-mb
      - "512"
      - --cleanup-interval
      - "300"
```

启动：

```bash
docker compose up -d
```

访问地址：
- 本机：`http://127.0.0.1:8000`
- 局域网：`http://<服务器IP>:8000`

### 3. 上传目录自动清理说明

容器内会按以下规则清理上传视频，避免磁盘占满：
- 超过 `--cleanup-minutes` 分钟的文件会被删除  
- 总大小超过 `--cleanup-max-mb` 时会从最旧开始删除  
- `--cleanup-interval` 控制清理频率（秒）

如果你希望关闭清理，可将 `--cleanup-minutes 0 --cleanup-max-mb 0`。

### 4. GPU 加速推理

当前 Dockerfile 安装的是 CPU 版 PyTorch。  
如需 GPU 推理，需满足：
- 主机安装 NVIDIA 驱动
- Docker 安装 nvidia-container-toolkit

并修改 Dockerfile或另建镜像：
1) 将 torch 安装替换为 CUDA 版本  
2) 运行容器时加 `--gpus all`  
3) 启动参数里使用 `--device cuda`

在 `docker-compose.yml` 中添加 GPU 相关配置。

