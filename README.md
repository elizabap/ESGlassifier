# 🌍 ESGlassifier
## From Text to Transparency: Extracting Environmental, Social, and Governance Aspects from Sentence-Level Text

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange)](https://huggingface.co/)

This repository contains the code, experiments, and documentation for our master’s thesis on cost-effective sentence-level ESG classification using transformer-based NLP models. It implements:

- Supervised learning with BERT, RoBERTa, and DistilRoBERTa
- Semi-supervised learning with pseudo-labeling
- Few-shot learning using SetFit

## 📁 Project Structure

```
ESGlassifier/
│
├── data/                   
│   ├── base_data/              # Sampled 10k data three times 
│   ├── domain_2k/              # Manually labeled ESG dataset from ESGBERT 
│   ├── domain_data/            # Domain-specific data sampled 7k from each domain
│   ├── llm_labeled/            # LLM-annotated data (pseudo-labeled)
│   ├── pseudo_labeled/         # Pseudo-labeled data via weak supervision
│   └── sentiment/              # Preprocessed data from ESG sentiment
│
├── eda/
│   ├── EDA.ipynb               
│   ├── load_dataset.ipynb      # Notebook to load and explore datasets
│   └── top_ngrams.csv         
│
├── labeling/
│   ├── pseudo_labeling/        # Folder for pseudo-label generation
│   └── llm_labeling.py         # Script for LLM-based label generation
│
├── pipeline/
│   ├── models/
│   │
│   ├── utils/
│   │   ├── data_loader.py      # Data loading and preprocessing
│   │   └── seed.py             # Random seed setup for reproducibility
│   │
│   ├── config.yaml             # Experiment configuration file
│   ├── train.py                # Standard training script
│   ├── train_cv.py             # Cross-validation training script
│   ├── train_classified_cv.py  # Cross-validation training specifically for semi-supervised, mostly the same as train_cv 
│   └── tune.py                 # Hyperparameter tuning script
│
├── results/
│   ├── few_shot/
│   ├── latex_tables/
│   ├── results_table_latex.py
│   └── visualize_all_sweeps.py           
├── README.md
└── requirements.txt
```

## 🚀 Setup and Training

### 1. Environment Setup
```bash
conda create --name venv python=3.9 -y
conda activate venv
pip install -r requirements.txt
```

### 2. Dataset Preparation
The datasets are available in the data folder, downloaded from Huggingface. The dataset structure includes:
- Base dataset with 10k samples
- Domain-specific ESG dataset (2k samples)
- Pseudo-labeled and LLM-annotated variants
- Sentiment-annotated data

### 3. Training Models

#### Standard Training
```bash
cd pipeline
python train.py --model model_of_your_choice
```

If you are running the SetFit model, you will only need to run`train.py`. This will automatically evaluate the model across multiple shots (e.g., 4, 8, 16, 32 samples per class) and random seeds, using the logic defined in `train.py` and `setfit.py`.



#### Cross-Validation Training
```bash
python train_cv.py --config path/to/config.yaml --model model_of_your_choice

```

#### Label-Specific Cross-Validation
```bash
python train_classified_cv.py \
    --pseudo_dataset path/to/pseudo.csv \
    --human_dataset path/to/human.csv \
    --config path/to/config.yaml \
    --model model_of_your_choice
```

#### Hyperparameter Tuning
```bash
python tune.py
```

## 💻 Running on NTNU HPC (Idun)

### 1. Sync Files
```bash
rsync -avz ~/path/to/local/project/ username@idun-login1.hpc.ntnu.no:/cluster/home/username/master/
```

### 2. Log in and Verify
```bash
ssh username@idun-login1.hpc.ntnu.no
ls -lh ~/master
```

### 3. Allocate GPU
```bash
srun --partition=GPUQ --gres=gpu:1 --time=04:00:00 --mem=32G --cpus-per-task=4 --pty bash
nvidia-smi  # Verify GPU allocation
```

### 4. Set Up Environment
```bash
module load Anaconda3/2024.02-1
conda create --name venv python=3.9 -y
conda activate venv
pip install -r requirements.txt
```

### 5. Run Training
```bash
cd ~/master/pipeline
python train.py
```

## 🔧 Configuration

The training configuration in `pipeline/config.yaml` includes:
- Model architecture selection
- Training hyperparameters
- Dataset paths and preprocessing settings
- Logging and evaluation metrics
- Cross-validation settings
