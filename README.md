# ECG Classification with Siamese Neural Network

A deep learning project for ECG heartbeat classification using Siamese Neural Networks.

## Project Structure
```
ecg_snn_project/
├── data/                    # Data directory
│   ├── processed/          # Processed dataset
│   └── raw/               # Raw MIT-BIH data
├── notebooks/             # Jupyter notebooks
│   ├── ECG.ipynb         # Main training notebook
│   └── data_analysis.ipynb # Data analysis notebook
├── scripts/              # Utility scripts
│   ├── download_data.sh  # Dataset download script
│   ├── prepare_data.py   # Data preprocessing script
│   └── train.py         # Training script
├── src/                 # Source code
│   ├── data/           # Data processing modules
│   ├── models/         # Model architectures
│   └── utils/          # Utility functions
├── environment.yml     # Conda environment file
└── requirements.txt    # Pip requirements file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/ecg_snn_project.git
cd ecg_snn_project
```

2. Create environment (choose one):

Using conda:
```bash
conda env create -f environment.yml
conda activate ecg_snn
```

Or using pip:
```bash
python -m venv env
source env/bin/activate  # Linux/Mac
env\Scripts\activate     # Windows
pip install -r requirements.txt
```

## Data Preparation

1. Download MIT-BIH dataset:
```bash
chmod +x scripts/download_data.sh  # Linux/Mac only
./scripts/download_data.sh
```

2. Process the data:
```bash
python scripts/prepare_data.py
```

## Training

Using script:
```bash
python scripts/train.py
```

Or using notebook:
```bash
jupyter notebook notebooks/ECG.ipynb
```

## Results

The model achieves:
- Test# ECG-Classification-with-Siamese-Neural-Network
# ECG-Classification-with-Siamese-Neural-Network
# Siamese-SNN-ECG
