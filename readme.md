# ZJU-SafeLine

在PCB的生产线上，有些环节禁止工人进行取放板操作。本项目通过实时分析生产线上的监控录像，可以在工人进行违规取放操作时，进行报警提示，确保流水线安全。

在提供的视频样本中，我们手工进行了数据标注，并形成了数据集。在保留的测试集上，本项目实现了良好的违规取放检测效果：

| 准确率（Accuracy） | 精确率（Precision） | 召回率（Recall） | F1     |
| ------------------ | ------------------- | ---------------- | ------ |
| 0.9898             | 0.9545              | 0.9844           | 0.9692 |

#### 使用说明

我们建议在虚拟环境中开始使用，请从`env.yml`中创建虚拟环境。如果你准备使用我们标注的数据，请从`release`中下载`label.csv`并跳过第2步，如果你准备使用我们训练完成的模型，请跳至第6步。

##### 1.数据预处理

`python data_process.py --videos-dir ...` 

注意，预期视频目录层级如下，即在你用命令行参数提供的目录下，应该包含两个目录，每个目录下有若干视频：

```
videos/
├── abnormal
│   ├── 1.mp4
│   ├── 2.mp4
│	...
└── normal
    ├── 1.mp4
    ├── 2.mp4
    ...

```

##### 2.数据标注

`python label.py`

这将启动一个可视化的窗口，允许你进行数据标注工作。**你也可以从**`release`**中下载**`label.csv`**并跳过这一步，这意味着你将使用我们的标注结果**。

##### 3.数据集构建

`python build_diff_dataset.py`

这将利用先前数据标注的结果构建用于模型训练的数据集。

##### 4.模型训练

模型训练由 `train_diff_model.py` 完成。该程序读取第3步构建的数据集，训练一个用于二分类的深度学习模型，用以判断是否发生违规取放操作。训练过程中会自动划分训练集、验证集和测试集，并保存训练和测试指标。如果你希望直接使用默认配置进行训练，只需运行：

```
python train_diff_model.py
```

默认情况下，程序会优先使用可用的 GPU 进行训练，否则自动退化为 CPU。训练完成后，模型权重与训练日志会被保存到 `runs/` 目录，其中包含验证集效果最优的模型文件，可直接用于后续推理与部署。

如果你需要调整训练轮数、批大小、学习率或数据划分方式等参数，可以通过命令行参数进行控制，具体可运行 `python train_diff_model.py --help` 查看完整说明。

##### 5.训练/测试指标记录

模型训练过程中，会自动保存训练和测试指标。您可以在 `tensorboard` 中查看训练过程和测试结果：

`tensorboard --logdir ...`

默认配置下，`logdir` 被设置为 `runs`. `tensorboard` 启动后，可以在 ` http://localhost:6006/` 查看训练和测试指标。

##### 6.Web 推理与可视化

使用训练完成的模型启动推理服务：

```
python web_infer.py --model final_model.pt --port 8000
```

启动后访问 `http://127.0.0.1:8000` 即可上传视频或配置推理源。  
可选参数：
- `--uploads-dir` 设置上传目录
- `--max-upload-mb` 限制单个上传文件大小



#### 容器化部署

我们建议容器化部署到生产环境中。要将我们提供的容器镜像部署到生产环境中，请按照如下步骤操作。

##### 1. 准备模型与目录

建议创建一个目录用于持久化模型与上传文件，例如：

```
deploy/
  models/
  uploads/
```

##### 2.上传训练完成的模型

在本仓库的`release`中下载我们提供的模型`final_model.pt`或者将你的模型放入`models`目录中。

##### 3. docker-compose 部署

在 `deploy/` 下创建 `docker-compose.yml`，如需更改模型和上传文件的目录，请在`volumes`中做对应修改：

```yaml
services:
  webinfer:
    image: ghcr.io/udp-worker/mv-infer:latest
    container_name: safeline
    ports:
      - "8000:8000"
    volumes:
      - ./models:/models:ro
      - ./uploads:/app/uploads
    command:
      - python3
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
      - --max-upload-mb
      - "512"
```

启动：

```bash
docker compose up -d
```

访问地址：
- 本机：`http://127.0.0.1:8000`
- 局域网：`http://<服务器IP>:8000`

##### 4. 上传目录自动清理说明

容器内会按以下规则清理上传视频，避免磁盘占满：
- 超过 `--cleanup-minutes` 分钟的文件会被删除  
- 总大小超过 `--cleanup-max-mb` 时会从最旧开始删除  
- `--cleanup-interval` 控制清理频率（秒）

如果你希望关闭清理，可将 `--cleanup-minutes 0 --cleanup-max-mb 0`。

#### 项目结构

```
ZJU-SafeLine
    ├── build_diff_dataset.py
    ├── data_process.py
    ├── label.py
    ├── train_diff_model.py
    ├── web_infer.py
    ├── static
    ├── templates
    └── env.yml
```

`data_process.py` 从提供的视频中按帧提取所有的图片信息，并进行必要的预处理，`label.py` 提供一个可视化的数据标注程序，允许用户高效地对图片进行标注，`build_diff_dataset.py` 读取用户标注的标签，并将图片和标签对应，形成数据集，`train_diff_model.py` 训练深度学习模型，检测是否发生了违规操作，`web_infer.py` 提供基于 FastAPI 的推理服务与 Web UI，`static/` 和 `templates/` 保存前端静态资源与页面模板。
