"""
NSFW 图像检测模型 WebUI
使用 Gradio 创建交互式界面
"""
import gradio as gr
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from PIL import Image
import torch
from transformers import ViTImageProcessor, ViTForImageClassification

# 全局变量
model = None
processor = None
model_loaded = False

def load_model():
    """加载模型"""
    global model, processor, model_loaded
    try:
        model_id = "Falconsai/nsfw_image_detection"
        model_path = Path(__file__).parent / "models"
        
        print(f"正在加载模型: {model_id}")
        
        # 尝试从本地加载
        if model_path.exists() and any(model_path.iterdir()):
            try:
                print(f"尝试从本地加载模型: {model_path}")
                processor = ViTImageProcessor.from_pretrained(str(model_path))
                model = ViTForImageClassification.from_pretrained(str(model_path))
                print("从本地加载模型成功")
            except Exception as e:
                print(f"本地加载失败: {e}，使用在线加载")
                processor = ViTImageProcessor.from_pretrained(model_id)
                model = ViTForImageClassification.from_pretrained(model_id)
                print("使用在线模型加载成功")
        else:
            # 使用在线模型
            print("使用在线模型")
            processor = ViTImageProcessor.from_pretrained(model_id)
            model = ViTForImageClassification.from_pretrained(model_id)
            print("在线模型加载成功")
        
        model.eval()  # 设置为评估模式
        model_loaded = True
        return "模型加载成功！"
    except Exception as e:
        error_msg = f"模型加载失败: {str(e)}"
        print(error_msg)
        return error_msg

def detect_image(image):
    """检测图像是否为 NSFW"""
    if not model_loaded or model is None or processor is None:
        return "请先加载模型", None, None
    
    if image is None:
        return "请上传一张图片", None, None
    
    try:
        # 确保图像是 PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            image = Image.open(image)
        
        # 预处理图像
        inputs = processor(images=image, return_tensors="pt")
        
        # 推理
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # 获取预测结果
        predicted_class_idx = logits.argmax(-1).item()
        confidence = probabilities[0][predicted_class_idx].item()
        
        # 类别标签
        class_names = ["normal", "nsfw"]
        predicted_class = class_names[predicted_class_idx]
        
        # 获取所有类别的概率
        all_probs = {
            class_names[i]: probabilities[0][i].item() 
            for i in range(len(class_names))
        }
        
        # 格式化结果
        result_text = f"预测结果: {predicted_class}\n"
        result_text += f"置信度: {confidence:.4f} ({confidence*100:.2f}%)\n\n"
        result_text += "所有类别概率:\n"
        for class_name, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
            result_text += f"  {class_name}: {prob:.4f} ({prob*100:.2f}%)\n"
        
        # 创建可视化结果
        result_dict = {
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "probabilities": all_probs,
            "timestamp": datetime.now().isoformat()
        }
        
        return result_text, result_dict, image
    except Exception as e:
        error_msg = f"检测失败: {str(e)}"
        print(error_msg)
        return error_msg, None, None

def test_with_sample_image():
    """使用示例图像进行测试"""
    if not model_loaded or model is None:
        return "请先加载模型", None, None
    
    try:
        # 创建一个简单的测试图像（纯色图像）
        test_image = Image.new('RGB', (224, 224), color='lightblue')
        
        # 进行检测
        result_text, result_dict, processed_image = detect_image(test_image)
        
        if result_dict:
            result_summary = f"测试图像检测完成\n\n{result_text}"
            return result_summary, processed_image, json.dumps(result_dict, indent=2, ensure_ascii=False)
        else:
            return result_text, processed_image, ""
    except Exception as e:
        error_msg = f"测试失败: {str(e)}"
        print(error_msg)
        return error_msg, None, ""

# 创建 Gradio 界面
with gr.Blocks() as app:
    gr.Markdown("# NSFW 图像检测模型 WebUI")
    gr.Markdown("基于 Vision Transformer (ViT) 的 NSFW 图像检测模型，可以自动识别图像是否为不当内容。")
    
    with gr.Tab("模型加载"):
        gr.Markdown("## 加载模型")
        gr.Markdown("点击下面的按钮加载 NSFW 图像检测模型。首次加载可能需要一些时间。")
        load_btn = gr.Button("加载模型", variant="primary", size="lg")
        load_status = gr.Textbox(label="加载状态", interactive=False, lines=3)
        load_btn.click(load_model, outputs=load_status)
    
    with gr.Tab("图像检测"):
        gr.Markdown("## 上传图像进行检测")
        gr.Markdown("上传一张图片，模型会自动检测是否为 NSFW 内容。")
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="上传图像", type="pil")
                detect_btn = gr.Button("开始检测", variant="primary", size="lg")
            with gr.Column():
                detect_result = gr.Textbox(label="检测结果", interactive=False, lines=10)
                result_json = gr.JSON(label="详细结果（JSON）", visible=False)
        detect_btn.click(
            detect_image, 
            inputs=image_input, 
            outputs=[detect_result, result_json, image_input]
        )
    
    with gr.Tab("快速测试"):
        gr.Markdown("## 使用示例图像进行测试")
        gr.Markdown("点击下面的按钮，使用预设的测试图像进行模型测试，无需上传图片。")
        test_btn = gr.Button("运行测试", variant="primary", size="lg")
        test_result = gr.Textbox(label="测试结果", interactive=False, lines=10)
        test_image_display = gr.Image(label="测试图像", interactive=False)
        test_json = gr.Textbox(label="测试结果（JSON）", interactive=False, lines=10)
        test_btn.click(
            test_with_sample_image, 
            outputs=[test_result, test_image_display, test_json]
        )

if __name__ == "__main__":
    # 自动加载模型
    print("正在启动 WebUI...")
    print("模型将在首次使用时自动加载")
    app.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        share=False
    )
