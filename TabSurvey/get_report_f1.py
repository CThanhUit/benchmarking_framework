import re
import yaml
import os
import glob
import argparse

model_list = [
    'MLP', 'TabNet', 'VIME', 'TabTransformer', 'NODE', 'DeepGBM', 'STG', 'NAM', 'DeepFM', 'SAINT', 'DANet',
    'LinearModel', 'KNN', 'SVM', 'DecisionTree', 'RandomForest',
    'XGBoost', 'CatBoost', 'LightGBM', 'ModelTree',
    'RLN', 'DNFNet'
    ]
import pandas as pd
## ONLY F1-SCORE
data_names = [ 'Drebin-215', 'Malgenome-215','AndroVul', 'Baltaci']
result = {}
for data_name in data_names:
    sub_result = {}   
    for model_name in model_list:
        sub_result[model_name] = {}
        file_path = f"output/{model_name}/{data_name}/results.txt"
        try:
            # Read the content of the results.txt file
            with open(file_path, 'r') as file:
                content = file.read()

            # Sử dụng biểu thức chính quy để tìm thông tin F1 score và Accuracy
            f1_mean = re.findall(r"F1 score - mean: ([\d.]+)", content)[-1]
            # f1_std = re.findall(r"F1 score - std: ([\d.]+)", content)[-1]
            # sub_result[model_name] = {data_name:f'{f1_mean}±{f1_std}'}
            sub_result[model_name] = {data_name:'{:.4f}'.format(float(f1_mean))}
        except:
            sub_result[model_name] = {data_name:"-"}
    result[data_name] = sub_result


df_result= pd.concat([pd.DataFrame.from_dict(result[data_name], orient ='index') for data_name in data_names], axis=1)

ourmodel_result= pd.DataFrame.from_dict({'OurModel': {'Drebin-215':0.9887 ,'Malgenome-215':0.9921,'AndroVul':0.8966, 'Baltaci':0.7676}}, orient ='index')


df = pd.concat([df_result, ourmodel_result], axis=0)

df.to_csv("result_f1.csv")