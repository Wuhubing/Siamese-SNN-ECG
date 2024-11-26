# Siamese-SNN-ECG

An efficient ECG classification framework based on Siamese Neural Network (SNN) and Spiking Neural Network.

## Project Overview

This project presents a novel ECG classification framework that combines the advantages of Siamese Neural Networks and Spiking Neural Networks for automatic arrhythmia detection and classification. Key features:

- Utilizes Siamese network architecture to enhance few-shot learning capability
- Integrates Spiking Neural Network to reduce computational complexity
- Achieves accurate classification of three ECG signal types: Normal, Ventricular Ectopic Beat (VEB), and Supraventricular Ectopic Beat (SVEB)

## Requirements

- Python 3.8+
- PyTorch 1.8+
- NumPy
- scikit-learn
- wfdb
- tqdm

## Installation

```bash
git clone https://github.com/Wuhubing/Siamese-SNN-ECG.git
cd Siamese-SNN-ECG
pip install -r requirements.txt
```

## Data Preparation

1. Download MIT-BIH Arrhythmia Database:

```bash
python download_data.py
```

2. Preprocess data:
```bash
python src/data/preprocess_data.py
```

## Model Training
```bash
python train_and_compare.py
```
## Project Structure
```bash
Siamese-SNN-ECG/
├── data/ # Data directory
├── src/ # Source code
│ ├── data/ # Data processing modules
│ ├── models/ # Model definitions
│ └── utils/ # Utility functions
├── results/ # Experimental results
└── best_model.pth # Pre-trained model
```

## Author

- Weibing Wang
- University of Wisconsin-Madison
- Email: wwang652@wisc.edu

## Citation

If you use this code or method in your research, please cite:

```bibtex
@article{wang2024towards,
title={Towards Efficient Healthcare Monitoring: A Novel Siamese-SNN Framework for Robust ECG Classification},
author={Wang, Weibing},
year={2024}
}
```

## License

MIT License

