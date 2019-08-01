#!/bin/bash

set -e  # Exit immediately on error

export PYTHONHASHSEED=0

# Save conda environment
conda env export > ../conda_environment.yaml && git add ../conda_environment.yaml

# Process GtR data
dvc run -w ..\
  -d model_config.yaml\
  -d data/raw/gtr_projects.csv\
  -d data/aux/gtr_projects_abstractText_drop.txt\
  -d baking_cookies/data/gtr.py\
  -d baking_cookies/features/text_preprocessing.py\
  -o data/processed/gtr_tokenised.csv\
  python baking_cookies/data/gtr.py

git add ../data/processed/.gitignore gtr_tokenised.csv.dvc


# Test-train split
dvc run -w ..\
  -d model_config.yaml\
  -d data/processed/gtr_tokenised.csv\
  -d baking_cookies/models/train_test_split.py\
  -o data/processed/gtr_leadFunder_id_train.csv\
  -o data/processed/gtr_leadFunder_id_test.csv\
  python baking_cookies/models/train_test_split.py

git add ../data/processed/.gitignore gtr_leadFunder_id_train.csv.dvc


# Train model
dvc run -w ..\
  -d model_config.yaml\
  -d data/processed/gtr_tokenised.csv\
  -d data/processed/gtr_leadFunder_id_train.csv\
  -d baking_cookies/models/train_model.py\
  -o models/gtr_leadFunder.pkl\
  python baking_cookies/models/train_model.py

git add ../models/.gitignore gtr_leadFunder.pkl.dvc


# Evaluate model
dvc run -w ..\
  -d model_config.yaml\
  -d data/processed/gtr_tokenised.csv\
  -d data/processed/gtr_leadFunder_id_test.csv\
  -d models/gtr_leadFunder.pkl\
  -d baking_cookies/models/evaluate.py\
  -o reports/figures/gtr_leadFunder_confusion_matrix.png\
  -M models/metrics.json\
  -f Dvcfile\
  python baking_cookies/models/evaluate.py

git add Dvcfile ../models/metrics.json ../reports/figures/.gitignore
