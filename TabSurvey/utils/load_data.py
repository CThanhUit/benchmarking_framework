import sklearn.datasets
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

import numpy as np
import pandas as pd


###########################
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

    if args.dataset == "CaliforniaHousing":  # Regression dataset
        X, y = sklearn.datasets.fetch_california_housing(return_X_y=True)

    elif args.dataset == "Adult" or args.dataset == "AdultCat":  # Binary classification dataset with categorical data, if you pass AdultCat, the numerical columns will be discretized.
        url_data = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

        features = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
        label = "income"
        columns = features + [label]
        df = pd.read_csv(url_data, names=columns)

        # Fill NaN with something better?
        df.fillna(0, inplace=True)
        if args.dataset == "AdultCat":
            columns_to_discr = [('age', 10), ('fnlwgt', 25), ('capital-gain', 10), ('capital-loss', 10),
                                ('hours-per-week', 10)]
            for clm, nvals in columns_to_discr:
                df[clm] = discretize_colum(df[clm], num_values=nvals)
                df[clm] = df[clm].astype(int).astype(str)
            df['education_num'] = df['education_num'].astype(int).astype(str)
            args.cat_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        X = df[features].to_numpy()
        y = df[label].to_numpy()
# ================================================== CICDDoS2019 ==========================================
    elif args.dataset == "CICDDoS2019":
        folder_path = "/content/datasets"
        if not os.path.exists(folder_path):
              os.makedirs(folder_path)
        file_name = "CICDDoS2019_dataloader.csv"
        file_path = os.path.join(folder_path, file_name)
        label = "Category_dtloader"
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path, low_memory=False)
        else:
            dt = CICDDoS2019()
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
            df=df.drop(columns = ['Unnamed: 0', 'Flow ID',' Source IP', ' Destination IP', ' Timestamp',	'Binary_dtloader', ' Label'])
            df.to_csv(file_path, index=False)
        features = df.drop(columns=label).columns.tolist()
        X=df[features].to_numpy()
        y=df[label].to_numpy()
# ================================================== CICMalMem2022 ==========================================
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
# ================================================== CICIDS2018 ==========================================
    elif args.dataset == "CICIDS2018":
        folder_path = "/content/datasets"
        if not os.path.exists(folder_path):
              os.makedirs(folder_path)
        file_name = "CICIDS2018_dataloader.csv"
        file_path = os.path.join(folder_path, file_name)
        label = "Binary_dtloader"
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path, low_memory=False)
        else:
            dt = BaseLoadDataset("CICIDS2018")
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
            df=df.drop(columns = ['Timestamp', 'Category_dtloader', 'Label'])
            df.to_csv(file_path, index=False)
        features = df.drop(columns=label).columns.tolist()
        X=df[features].to_numpy()
        y=df[label].to_numpy()

    else:
        raise AttributeError("Dataset \"" + args.dataset + "\" not available")

    print("Dataset loaded!")
    print(X.shape)

    # Preprocess target
    if args.target_encode:
        le = LabelEncoder()
        y = le.fit_transform(y)

        # Setting this if classification task
        if args.objective == "classification":
            args.num_classes = len(le.classes_)
            print("Having", args.num_classes, "classes as target.")

    num_idx = []
    args.cat_dims = []

    # Preprocess data
    for i in range(args.num_features):
        if args.cat_idx and i in args.cat_idx:
            le = LabelEncoder()
            X[:, i] = le.fit_transform(X[:, i])

            # Setting this?
            args.cat_dims.append(len(le.classes_))

        else:
            num_idx.append(i)

    if args.scale:
        print("Scaling the data...")
        scaler = StandardScaler()
        X[:, num_idx] = scaler.fit_transform(X[:, num_idx])

    if args.one_hot_encode:
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        new_x1 = ohe.fit_transform(X[:, args.cat_idx])
        new_x2 = X[:, num_idx]
        X = np.concatenate([new_x1, new_x2], axis=1)
        print("New Shape:", X.shape)

    return X, y
