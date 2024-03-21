@echo off
setlocal

:: Set the number of trials and epochs
set N_TRIALS=2
set EPOCHS=3

:: Define the list of models
set MODELS=LinearModel KNN SVM DecisionTree RandomForest XGBoost CatBoost LightGBM MLP TabNet VIME TabTransformer ModelTree NODE DeepGBM RLN DNFNet STG NAM DeepFM SAINT DANet

:: Define the list of configurations
set CONFIGS=config\adult.yml config\california_housing.yml

:: Loop through configurations and models
for %%c in (%CONFIGS%) do (
  for %%m in (%MODELS%) do (
    echo.
    echo -------------------------------------------------------------------------------
    echo Training %%m with %%c
    python train.py --config "%%c" --model_name "%%m" --n_trials %N_TRIALS% --epochs %EPOCHS%
  )
)

endlocal
