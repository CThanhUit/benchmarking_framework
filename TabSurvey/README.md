# TOWARD BUILD A BENCHMARKING FRAMEWORK TO EVALUATE DATASETS FOR INTRUSION DETECTION SYSTEMS.
## I. Overview
This framework supports preprocessing datasets, training models, and analysis, comparing results between machine learning models based on a benchmarking framework.
## II. Roadmap

## III. Feautures:
### Add new dataset:
1. **Config File**:
-  Add the config file config_{datasetname}.yml to DataLoader/config:
```bat
#Info of the dataset
ds_name: "" #Required
ds_size: None
ds_paper_link: ""
ds_link: "" #Required
csv_link: None
fts_names: [ ] #Required
target_variable: "" #Required

# Actual number of samples in the csv file of each class
real_cnt: { } #Required

# Map the actual label in the csv file with the label in the paper
label_map: { }

# Group the actual label in the csv file with the coresponding category in the paper
category_map: { } #Required
binary_map: { } #Required

# Map the actual label in the csv file with the label in the paper
label_true_name: [ ] #Required

 # List the label be dropped
label_drop: [ ]
````
        
- Add the config file {datasetname}.yml to config directory.
```bat
dataset: 
model_name: # LinearModel, KNN, SVM, DecisionTree, RandomForest, MLP
                           # XGBoost, CatBoost, LightGBM,
                           # TabNet, VIME, TabTransformer, ModelTree, NODE, DeepGBM, RLN, DNFNet,
                           # STG, NAM, DeepFM, SAINT
objective: 
# optimize_hyperparameters: True

# GPU parameters
use_gpu: 
gpu_ids: 
data_parallel: 

# Optuna parameters - https://optuna.org/
n_trials: 
direction: 

# Cross validation parameters
num_splits: 
shuffle: 
seed: 221 

# Preprocessing parameters
scale: 
target_encode: 
one_hot_encode: 

# Training parameters
batch_size: 
val_batch_size: 
early_stopping_rounds: 
epochs: 
logging_period: 

# About the data
num_classes:   # for classification
num_features: 
cat_idx: []
# cat_dims: will be automatically set.
cat_dims: []
```
2. **Modify load_data.py**:
- In the `load_data()` function of the `load_data.py` file, you need to add an `elif` statement to handle the case when `args.dataset` matches the name of the dataset you are adding:
- Within this `elif` block, perform the following:
    - Load the dataset from a CSV file (your dataset) or a ZIP file either from the cloud using DataLoader. Below is an example of handling data loading from a ZIP file (downloaded from the cloud and specified as a local path).
```python
elif args.dataset == "CICMalMem2022":
    folder_path = "/content/datasets"
    if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    file_name = "CICMalMem2022_dataloader.csv"
    file_path = os.path.join(folder_path, file_name)
    label = "Category_dtloader"
    if os.path.isfile(file_path):
        df = pd.read_csv(file_path, low_memory=False)
    else:
        dt = BaseLoadDataset("CICMalMem2022")
        dt.DownLoad_Data(load_type="raw")
        dt.Load_Data(load_type="raw", limit_cnt=50000)
        dt.Preprocess_Data()
        df = dt.To_dataframe()
        df = df.dropna()
        for feature in df.drop(columns=label).columns.tolist():
            if df[feature].dtypes == 'object':
                le = LabelEncoder()
                df[feature]=df[feature].astype('str')
                df[feature] = le.fit_transform(df[feature])
        df=df.drop(columns = ['Category', 'Binary_dtloader', 'Class'])
        df.to_csv(file_path, index=False)
    features = df.drop(columns=label).columns.tolist()
    X=df[features].to_numpy()
    y=df[label].to_numpy()
```
3. **Setup Params**:
   - As mentioned earlier, training requires specific parameters, which are obtained from the `best_params.yml` file if `--optimize_hyperparameters` is not used. To run with a new dataset, you should run this option and use the best_params in the output file to overwrite `best_params.yml`. Make sure to use a GPU, and note the GPU index in the dataset's config file.
   - Modify the `python train` line in the `testall.sh` file to include the `--optimize_hyperparameters` and `--use_gpu` options. The line should look like this:
```bash
     python train.py --config "$config" --model_name "$model" --n_trials $N_TRIALS --epochs $EPOCHS --optimize_hyperparameters --use_gpu
```
After running these, the results will be stored in `"output/<model_name>/<data_name>/results.txt"`. You can then extract the results for reporting or create a new `best_params` file using the `get_best_params.py` script.
## Run a single model on a single dataset:
To run a single model on a single dataset call:
```bash
    python train.py --config "$config" --model_name "$model" --n_trials $N_TRIALS --epochs $EPOCHS
```
All parameters set in the config file, can be overwritten by command line arguments:
`--optimize_hyperparameters`: Uses Optuna to run a hyperparameter optimization. If not set, the parameters listed in the best_params.yml file are used.

`--n_trails` <number trials>: Number of trials to run for the hyperparameter search

`--epochs` <number epochs>: Max number of epochs

`--use_gpu`: If set, available GPUs are used (specified by gpu_ids)
The trained model and  result is saved in output directory.
## Run multiple models on multiple datasets:
To run multiple models on multiple datasets, you can call:
```bash
./testall.sh
```
## Evaluate results:
The results are saved in `report_tabsurvey.csv`:
|Name	|Model	|MCC - mean	|ACC - mean	|TPR - mean	|FPR - mean	|F1 - mean	|TPR weight - mean	|PPV weight - mean	|F1 weight - mean	|AUC_f - mean|
|-------|-------|-----------|-----------|-----------|-----------|-----------|-------------------|-------------------|-------------------|------|
|CICMalMem2022	|LinearModel	|0.998105092	|99.90524964	|99.89589238	|0.085526811	|99.90455832	|99.90524964	|99.90526935	|99.90524956	|0.999051828
|CICMalMem2022	|KNN	|0.992007023	|99.60032316	|99.53497828	|0.335258523	|99.59720198	|99.60032316	|99.60042348	|99.60032128	|0.995998599
|CICMalMem2022	|DecisionTree	|0.999655434	|99.98277228	|99.98611834	|0.020526271	|99.98264883	|99.98277228	|99.98277289	|99.98277229	|0.99982796
|CICMalMem2022	|RandomForest	|0.999758817	|99.98794069	|99.99652959	|0.020526271	|99.9878549	|99.98794069	|99.98794219	|99.9879407	|0.999880017

