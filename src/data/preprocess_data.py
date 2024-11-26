import os
import wfdb
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

def load_and_process_record(record_path):
    """加载并处理单个ECG记录"""
    try:
        # 读取记录和标注
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, 'atr')
        
        # 获取信号数据和标注
        signals = record.p_signal
        labels = annotation.symbol
        sample_points = annotation.sample
        
        return signals, labels, sample_points
    except Exception as e:
        print(f"Error processing record {record_path}: {str(e)}")
        return None, None, None

def extract_heartbeats(signals, sample_points, window_size=250):
    """从信号中提取心跳片段"""
    heartbeats = []
    for point in sample_points:
        # 确保窗口在信号范围内
        start = max(0, point - window_size//2)
        end = min(len(signals), point + window_size//2)
        
        # 提取心跳片段
        beat = signals[start:end]
        
        # 如果片段长度不足，进行填充
        if len(beat) < window_size:
            pad_width = window_size - len(beat)
            beat = np.pad(beat, ((0, pad_width), (0, 0)), mode='constant')
            
        heartbeats.append(beat)
    
    return np.array(heartbeats)

def preprocess_data():
    """预处理MIT-BIH数据集"""
    print("Starting data preprocessing...")
    
    # 创建processed目录
    os.makedirs('data/processed', exist_ok=True)
    
    # 获取所有记录
    record_paths = []
    raw_dir = 'data/raw'
    for file in os.listdir(raw_dir):
        if file.endswith('.dat'):
            record_name = file[:-4]
            record_paths.append(os.path.join(raw_dir, record_name))
    
    # 收集所有数据
    all_heartbeats = []
    all_labels = []
    
    for record_path in tqdm(record_paths, desc="Processing records"):
        signals, labels, sample_points = load_and_process_record(record_path)
        if signals is None:
            continue
            
        # 提取心跳
        heartbeats = extract_heartbeats(signals, sample_points)
        
        # 只保留主要类别: N(正常), V(室性早搏), A(房性早搏)
        valid_beats = []
        valid_labels = []
        for beat, label in zip(heartbeats, labels):
            if label in ['N', 'V', 'A']:
                valid_beats.append(beat)
                valid_labels.append(label)
        
        all_heartbeats.extend(valid_beats)
        all_labels.extend(valid_labels)
    
    # 转换为numpy数组
    X = np.array(all_heartbeats)
    y = np.array(all_labels)
    
    # 标签编码
    label_map = {'N': 0, 'V': 1, 'A': 2}
    y = np.array([label_map[label] for label in y])
    
    # 数据集划分
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    # 准备数据字典
    processed_data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'classes': ['Normal', 'VEB', 'SVEB'],
        'scaler': scaler
    }
    
    # 保存处理后的数据
    with open('data/processed/processed_data.pkl', 'wb') as f:
        pickle.dump(processed_data, f)
    
    print("\nData preprocessing completed!")
    print(f"Total samples: {len(y)}")
    print(f"Training samples: {len(y_train)}")
    print(f"Validation samples: {len(y_val)}")
    print(f"Test samples: {len(y_test)}")
    print("\nClass distribution:")
    for i, class_name in enumerate(['Normal', 'VEB', 'SVEB']):
        print(f"{class_name}: {sum(y == i)} samples")

if __name__ == '__main__':
    preprocess_data()