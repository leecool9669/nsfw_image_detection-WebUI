"""
NSFW 图像检测模型下载脚本
从 HuggingFace 下载 Falconsai/nsfw_image_detection 模型
"""
import os
from huggingface_hub import snapshot_download, hf_hub_download
from pathlib import Path

def download_model():
    """下载 nsfw_image_detection 模型"""
    model_id = "Falconsai/nsfw_image_detection"
    local_dir = Path(__file__).parent / "models"
    local_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"开始下载模型: {model_id}")
    print(f"保存路径: {local_dir}")
    
    try:
        # 先下载配置文件（不下载大文件）
        print("正在下载模型配置文件...")
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            ignore_patterns=["*.bin", "*.safetensors", "*.pt", "*.pth"],
        )
        print("模型配置文件下载完成")
        
        # 查找最小的模型文件
        # 先尝试下载 quantized 版本（如果存在）
        try:
            print("尝试下载量化模型（更小）...")
            # 检查是否有量化版本
            quantized_file = hf_hub_download(
                repo_id=model_id,
                filename="pytorch_model.bin",
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
            )
            print(f"模型权重文件下载完成: {quantized_file}")
        except Exception as e:
            print(f"尝试下载 pytorch_model.bin 失败: {e}")
            # 尝试 safetensors 格式
            try:
                print("尝试下载 safetensors 格式...")
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename="model.safetensors",
                    local_dir=str(local_dir),
                    local_dir_use_symlinks=False,
                )
                print(f"模型权重文件下载完成: {model_file}")
            except Exception as e2:
                print(f"下载模型权重文件失败: {e2}")
                print("将使用在线加载模式（模型会在首次使用时自动下载）")
        
        print("模型下载完成！")
        return True
    except Exception as e:
        print(f"下载模型时出错: {e}")
        print("将使用在线加载模式（模型会在首次使用时自动下载）")
        return False

if __name__ == "__main__":
    download_model()
