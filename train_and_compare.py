import torch
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
from src.models.snn import SNN, CombinedLoss
from src.data.dataset import create_dataloaders
from src.utils.training import train_model

def train_models():
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # 加载数据
    with open('data/processed/processed_data.pkl', 'rb') as f:
        processed_data = pickle.load(f)
    
    # 定义要训练的模型配置
    models_to_train = [
        {
            'name': 'baseline',
            'config': {
                'num_epochs': 100,
                'batch_size': 128,
                'lr': 0.001
            }
        },
        {
            'name': 'larger_batch',
            'config': {
                'num_epochs': 100,
                'batch_size': 256,
                'lr': 0.001
            }
        },
        {
            'name': 'higher_lr',
            'config': {
                'num_epochs': 100,
                'batch_size': 128,
                'lr': 0.002
            }
        }
    ]
    
    # 创建结果目录
    os.makedirs('results', exist_ok=True)
    
    # 训练所有模型
    results = {}
    for model_config in models_to_train:
        print(f"\nTraining {model_config['name']}...")
        model, model_results = train_model(
            processed_data,
            model_name=model_config['name'],
            **model_config['config']
        )
        results[model_config['name']] = model_results
    
    return results

def plot_comparison(results):
    """绘制不同模型的训练过程对比图"""
    plt.figure(figsize=(15, 5))
    
    # 损失曲线
    plt.subplot(121)
    for name, result in results.items():
        plt.plot(result['val_losses'], label=f'{name}_val')
        plt.plot(result['train_losses'], '--', label=f'{name}_train')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 准确率曲线
    plt.subplot(122)
    for name, result in results.items():
        plt.plot(result['val_accs'], label=f'{name}_val')
        plt.plot(result['train_accs'], '--', label=f'{name}_train')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/training_curves_comparison.png')
    plt.close()

def create_comparison_table(results):
    """创建性能对比表格"""
    data = []
    for name, result in results.items():
        data.append({
            'Model': name,
            'Best Val Acc': f"{max(result['val_accs']):.4f}",
            'Test Acc': f"{result['test_acc']:.4f}",
            'Best Epoch': result['best_epoch'] + 1,
            'Training Time (min)': f"{result['training_time']:.1f}"
        })
    
    df = pd.DataFrame(data)
    return df

def main():
    print("Starting model training and comparison...")
    
    # 训练模型
    results = train_models()
    
    # 绘制对比图
    plot_comparison(results)
    
    # 创建对比表格
    comparison_table = create_comparison_table(results)
    print("\nModel Performance Comparison:")
    print(comparison_table.to_string(index=False))
    
    # 保存表格
    comparison_table.to_csv('results/model_comparison.csv', index=False)
    
    print("\nResults have been saved to the 'results' directory:")
    print("- Training curves: results/training_curves_comparison.png")
    print("- Performance comparison: results/model_comparison.csv")

if __name__ == '__main__':
    main()