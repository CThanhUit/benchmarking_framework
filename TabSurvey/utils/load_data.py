import sklearn.datasets
from sklearn.preprocessing import QuantileTransformer, LabelEncoder, OneHotEncoder

import numpy as np
import pandas as pd


###########################
import sys
sys.path.append('../')
import DataLoader
from DataLoader import *
import os
from os import path
###########################

def discretize_colum(data_clm, num_values=10):
    """ Discretize a column by quantiles """
    r = np.argsort(data_clm)
    bin_sz = (len(r) / num_values) + 1  # make sure all quantiles are in range 0-(num_quarts-1)
    q = r // bin_sz
    return q


def load_data(args):
    print("Loading dataset " + args.dataset + "...")
    limit_samples=25000
    folder_path = "/home/jupyter-iec_cyberlearning/datasets"
    label = 'Default_label'
    if not os.path.exists(folder_path):
          os.makedirs(folder_path)
    file_name =  "data_" + args.dataset + ".csv"
    file_path = os.path.join(folder_path, file_name)
# =====================================================================================================
    if args.dataset == "CICIoT2023":
        dt = CICIoT2023(print_able = False)
    elif args.dataset == "CICDDoS2019":
        dt = CICDDoS2019(print_able = False)
    elif args.dataset == "CICIDS2018":
        dt = CICIDS2018(print_able = False)
    elif args.dataset == "CICIDS2017":
        dt = CICIDS2017(print_able = False)
    elif args.dataset == "ToNIoT":
        dt = ToNIoT(print_able = False)
    else:
        raise AttributeError("Dataset \"" + args.dataset + "\" not available")
        
    if os.path.isfile(file_path):
        dt.Load_Data(path = file_path, load_type="preload", limit_cnt=limit_samples)
    else:
        dt.DownLoad_Data(path = folder_path, load_type="raw")
        dt.Load_Data(path = folder_path, load_type="raw", limit_cnt=limit_samples)
        dt.To_csv(path = file_path)
    dt.Preprocess_Data(drop_cols=None, type_encoder='LabelEncoder', type_scaler='QuantileTransformer' , type_select='SelectKBest', num_fts='all')
    X, y = dt.Split_data(target_variable=label)

    print("Dataset loaded!")
    print(X.shape)
    print(y.shape)
    args.cat_dims = []
    # Preprocess target
    if args.target_encode:
        le = LabelEncoder()
        y = le.fit_transform(y)

        # Setting this if classification task
        if args.objective == "classification":
            args.label_classes = le.classes_
            args.num_classes = len(args.label_classes)
            print("Having", args.num_classes, "classes as target.")
    args.num_features = X.shape[1]
    return X, y
