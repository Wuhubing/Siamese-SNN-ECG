import os
import requests
from tqdm import tqdm

def download_file(url, local_file):
    """下载单个文件"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 检查是否成功
        
        with open(local_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"\n下载 {os.path.basename(local_file)} 时出错: {str(e)}")
        if os.path.exists(local_file):
            os.remove(local_file)
        return False

def download_mitbih_data():
    """下载MIT-BIH心律失常数据库的子集（记录100-124）"""
    
    # 创建数据目录
    os.makedirs('data/raw', exist_ok=True)
    
    # 基础URL
    base_url = "https://physionet.org/files/mitdb/1.0.0"
    
    # 要下载的记录编号
    record_numbers = range(100, 125)  # 100-124
    
    # 文件扩展名
    extensions = ['.dat', '.hea', '.atr']
    
    print("开始下载MIT-BIH数据集...")
    
    # 下载进度条
    total_files = len(record_numbers) * len(extensions)
    success_count = 0
    
    with tqdm(total=total_files, desc="下载进度") as pbar:
        for record in record_numbers:
            for ext in extensions:
                filename = f"{record}{ext}"
                url = f"{base_url}/{filename}"
                local_file = os.path.join('data/raw', filename)
                
                # 如果文件已存在，跳过下载
                if os.path.exists(local_file):
                    success_count += 1
                    pbar.update(1)
                    continue
                
                # 下载文件
                if download_file(url, local_file):
                    success_count += 1
                pbar.update(1)
    
    # 验证下载
    print("\n验证下载的文件...")
    missing_files = []
    for record in record_numbers:
        for ext in extensions:
            filename = f"{record}{ext}"
            local_file = os.path.join('data/raw', filename)
            if not os.path.exists(local_file):
                missing_files.append(filename)
    
    if missing_files:
        print("\n以下文件下载失败:")
        for filename in missing_files:
            print(f"- {filename}")
    else:
        print("\n所有文件下载成功!")
    
    # 打印数据集信息
    print("\n数据集信息:")
    print(f"- 记录数量: {len(record_numbers)}")
    print(f"- 每个记录的文件: {', '.join(extensions)}")
    print(f"- 成功下载文件数: {success_count}/{total_files}")
    print(f"- 存储位置: {os.path.abspath('data/raw')}")

if __name__ == '__main__':
    download_mitbih_data()