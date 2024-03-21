#!/bin/bash

N_TRIALS=2
EPOCHS=3

MODELS=(  "LinearModel" "KNN" "SVM" "DecisionTree" "RandomForest" "XGBoost" "CatBoost" "LightGBM" "MLP" "TabNet" "VIME" 
          "TabTransformer" "ModelTree" "NODE" "DeepGBM" "RLN" "DNFNet" "STG" "NAM" "DeepFM" "SAINT" "DANet"
          )

CONFIGS=( "config/CICDDoS2019.yml"
          )

for config in "${CONFIGS[@]}"; do
  for model in "${MODELS[@]}"; do
    printf "\n\n----------------------------------------------------------------------------\n"
    printf 'Training %s with %s\n' "$model" "$config"
    python train.py --config "$config" --model_name "$model" --n_trials $N_TRIALS --epochs $EPOCHS
  done
done
