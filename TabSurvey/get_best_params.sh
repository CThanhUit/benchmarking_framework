#!/bin/bash

N_TRIALS=2
EPOCHS=3

MODELS=(  "XGBoost" "LightGBM" "KNN" "SAINT" "TabTransformer" "TabNet" 
          )

CONFIGS=( "config/CICIoT2023.yml"
          )

for config in "${CONFIGS[@]}"; do
  for model in "${MODELS[@]}"; do
    printf "\n\n----------------------------------------------------------------------------\n"
    printf 'Training %s with %s\n' "$model" "$config"
    python train.py --config "$config" --model_name "$model" --n_trials $N_TRIALS --epochs $EPOCHS --optimize_hyperparameters --use_gpu
  done
done
