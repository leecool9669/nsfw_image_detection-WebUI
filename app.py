# -*- coding: utf-8 -*-
"""NSFW 图像检测 WebUI 演示（不加载真实模型，仅界面展示）。"""
from __future__ import annotations
import gradio as gr

def fake_load_model():
    return "模型状态：nsfw_image_detection（ViT 图像分类）已就绪（演示模式，未加载真实权重）"

def fake_detect(image) -> tuple[str, str]:
    if image is None:
        return "", "请先上传或选择一张图片再进行检测。"
    out_text = (
        "【演示模式】当前未加载真实模型，以下为界面展示示例。\n\n"
        "加载真实模型后，将在此显示：\n"
        "· 预测类别：normal / nsfw\n"
        "· 各类别置信度或概率\n"
        "· 可视化结果（如柱状图或标签）"
    )
    return "normal（演示）", out_text

def build_ui():
    with gr.Blocks(title="NSFW Image Detection · WebUI") as demo:
        gr.Markdown("## NSFW 图像检测 · WebUI 演示")
        gr.Markdown(
            "本界面用于展示基于 Vision Transformer (ViT) 的 NSFW 图像分类模型的典型使用流程，"
            "包括模型加载状态与图像分类结果的可视化展示。"
        )
        with gr.Row():
            load_btn = gr.Button("加载模型（演示）", variant="primary")
            status_box = gr.Textbox(label="模型状态", value="尚未加载", interactive=False)
        load_btn.click(fn=fake_load_model, outputs=status_box)
        with gr.Tabs():
            with gr.Tab("图像分类"):
                gr.Markdown("上传一张图片，模型将对其进行 NSFW 分类（normal / nsfw）。")
                img_inp = gr.Image(label="输入图片", type="pil")
                label_out = gr.Textbox(label="预测类别", interactive=False)
                detail_out = gr.Textbox(label="结果说明", lines=8, interactive=False)
                run_btn = gr.Button("开始检测（演示）")
                run_btn.click(fn=fake_detect, inputs=img_inp, outputs=[label_out, detail_out])
        gr.Markdown("---\n*说明：当前为轻量级演示界面，未实际下载与加载模型参数。*")
    return demo

def main():
    build_ui().launch(server_name="127.0.0.1", server_port=7860, share=False)

if __name__ == "__main__":
    main()
