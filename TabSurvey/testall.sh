#!/bin/bash

N_TRIALS=1
EPOCHS=1

MODELS=(  "LinearModel" "KNN" "DecisionTree" "RandomForest" "XGBoost" "CatBoost" "LightGBM" "MLP" 
          "DeepGBM"  "DeepFM" "SAINT"
          )

CONFIGS=( "config/CICMalMem2022.yml"
          "config/CICIDS2018.yml"
          )

for config in "${CONFIGS[@]}"; do
  for model in "${MODELS[@]}"; do
    printf "\n\n----------------------------------------------------------------------------\n"
    printf 'Training %s with %s\n' "$model" "$config"
    python train.py --config "$config" --model_name "$model" --n_trials $N_TRIALS --epochs $EPOCHS
  done
done
