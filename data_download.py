# download_data.py
"""
自动下载RAVDESS数据集
"""

import os
import zipfile
import requests
from tqdm import tqdm
from pathlib import Path


def download_file(url, save_path):
    """
    下载文件并显示进度条
    """
    print(f"📥 开始下载: {url}")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(save_path, 'wb') as f, tqdm(
        desc="下载中",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    
    print(f"✓ 下载完成: {save_path}")


def extract_zip(zip_path, extract_to):
    """
    解压zip文件
    """
    print(f"📂 解压中: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    print(f"✓ 解压完成: {extract_to}")


def download_ravdess():
    """
    下载并准备RAVDESS数据集
    """
    # 创建data文件夹
    data_dir = Path('./data')
    data_dir.mkdir(exist_ok=True)
    
    # RAVDESS下载链接（Zenodo官方镜像）
    url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
    zip_path = data_dir / "RAVDESS.zip"
    extract_path = data_dir / "RAVDESS"
    
    # 1. 检查是否已下载
    if extract_path.exists() and len(list(extract_path.glob('Actor_*'))) == 24:
        print("✓ RAVDESS数据集已存在，跳过下载")
        return
    
    # 2. 下载
    if not zip_path.exists():
        try:
            download_file(url, zip_path)
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            print("请手动下载：https://zenodo.org/record/1188976")
            print(f"并保存到: {zip_path}")
            return
    else:
        print(f"✓ 找到已下载的文件: {zip_path}")
    
    # 3. 解压
    try:
        extract_zip(zip_path, data_dir)
    except Exception as e:
        print(f"❌ 解压失败: {e}")
        return
    
    # 4. 检查文件结构
    actor_folders = list(extract_path.glob('Actor_*'))
    print(f"\n✓ 数据集准备完成！")
    print(f"  - 文件夹数量: {len(actor_folders)}")
    
    # 统计文件数
    total_files = sum(len(list(folder.glob('*.wav'))) for folder in actor_folders)
    print(f"  - 音频文件总数: {total_files}")
    
    # 5. 可选：删除zip文件节省空间
    if zip_path.exists():
        delete = input("\n是否删除zip文件以节省空间? (y/n): ").lower()
        if delete == 'y':
            zip_path.unlink()
            print(f"✓ 已删除: {zip_path}")


if __name__ == '__main__':
    print("=" * 50)
    print("RAVDESS 数据集下载工具")
    print("=" * 50)
    
    download_ravdess()
    
    print("\n" + "=" * 50)
    print("✓ 全部完成！")
    print("=" * 50)