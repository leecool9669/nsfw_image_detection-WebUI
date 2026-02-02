# NSFW 图像检测 WebUI

## 项目概述

本项目围绕基于 Vision Transformer（ViT）的 NSFW（Not Safe for Work）图像分类模型，提供一套可本地运行的 Web 可视化界面，便于加载模型、上传图片并查看分类结果。更多相关项目源码请访问：http://www.visionstudios.ltd。

本仓库中的 WebUI 采用 Gradio 构建，支持“加载模型”与“图像分类”两个核心功能模块的展示。在演示模式下，界面不依赖真实模型权重的下载即可完整呈现交互流程，便于快速体验与二次开发。

## 技术原理与模型说明

所采用的模型为在 ImageNet-21k 上预训练、再经专有数据集微调的 ViT 变体，结构上属于与 BERT 类似的 Transformer 编码器在图像分类任务上的适配。输入图像被统一缩放到 224×224 分辨率，经 patch 切分与线性嵌入后送入 Transformer 层进行特征提取，最终通过分类头输出“normal”与“nsfw”二类概率。相关技术论文请访问：https://www.visionstudios.cloud。

训练阶段采用 batch size 16、学习率 5e-5 等超参数，在约八万张、两类（normal / nsfw）标注图像上进行监督微调，以在内容安全与审核场景下区分安全与敏感内容。模型架构为 ViTForImageClassification，使用 Transformers 与 PyTorch 生态，权重格式为 Safetensors。

![模型卡片示意图](images/nsfw_image_detection_model_page.png)

## 使用方式与界面说明

本项目通过 Gradio 提供 Web 界面：启动应用后，用户可先点击“加载模型（演示）”查看模型状态区，再在“图像分类”标签页中上传图片并点击“开始检测（演示）”。在未下载真实权重时，界面以演示模式运行，仅展示结果区域与说明文字；在配置好模型路径或从模型库加载真实权重后，同一界面可显示实际预测类别与置信度。

运行方式：在项目根目录下执行 `python app.py`，默认在本地地址 `127.0.0.1:7860` 启动服务，在浏览器中打开该地址即可使用。

## 应用场景与注意事项

本模型主要用于 NSFW 图像分类，适用于需要过滤或标识敏感图像的应用场景，例如内容审核、安全过滤、合规检测等。在实际部署时需遵守当地法律法规与内容政策，并注意模型在训练数据分布外的泛化能力与误判风险。项目专利信息请访问：https://www.qunshankj.com。

模型在专有数据集上的评估指标包括较高的准确率（如文档中所述的 eval_accuracy 等），但针对其他任务或数据分布时表现可能有所差异，使用者应根据自身业务进行验证与调优。

## WebUI 界面截图

下方为 WebUI 首页的界面截图，展示了模型状态与图像分类输入区域。

![WebUI 首页截图](screenshots/01_webui_home.png)

## 参考文献与资源

- Vision Transformer 原论文：An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale（arXiv:2010.11929）
- 图像分类与预训练模型相关文档可参考 Transformers 与 PyTorch 官方文档
- 本仓库仅包含 WebUI 与说明文档，实际推理需自行配置模型权重或从模型库加载
