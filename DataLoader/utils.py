import numpy as np
np.random.seed(1337)  # for reproducibility
import pandas as pd
import os
from typing import List, Tuple, Generator, Iterator
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import sys
from tabulate import tabulate
import csv
import time
from natsort import natsorted
import pickle
import multiprocessing as mp
from os import path

from sklearn import preprocessing
from sklearn import base
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import (StandardScaler, OrdinalEncoder, LabelEncoder, MinMaxScaler, OneHotEncoder, QuantileTransformer)
from sklearn.model_selection import train_test_split

import zipfile
import tarfile
import rarfile
from tqdm.auto import tqdm
TupleOrList = tuple([Tuple, List])

class CustomMerger(base.BaseEstimator, base.TransformerMixin):
  """Merge List of DataFrames"""

  def __init__(self):
    pass

  def fit(self, X: pd.DataFrame, y=None):
    return self

  def transform(self, X: pd.DataFrame, y='deprecated', copy=True):
    if isinstance(X, TupleOrList):
        return pd.concat(X, ignore_index=True).reset_index(drop=True)

    return X
  
class CustomEncoder(base.BaseEstimator, base.TransformerMixin):
  """Custom encoder data"""
  def __init__(self):
      self.mapping = {}
      self.inverse_mapping = {}
      self.next_label = 0
  
  def fit(self, data):
      self._check_for_null(data)
      for value in data:
          if value not in self.mapping:
              self.mapping[value] = self.next_label
              self.inverse_mapping[self.next_label] = value
              self.next_label += 1
  
  def transform(self, data, copy=True):
      self._check_for_null(data)
      encoded_data = []
      for value in data:
          encoded_value = self.mapping.get(value, -1)  # Return -1 for unseen values
          encoded_data.append(encoded_value)

      if isinstance(encoded_data, TupleOrList):
        return pd.concat(encoded_data, ignore_index=True).reset_index(drop=True)
      return encoded_data
  
  def _check_for_null(self, data):
      if any(value is None or (isinstance(value, float) and math.isnan(value)) for value in data):
          raise ValueError("Input data contains null or NaN values.")
  
class CustomScaler(base.BaseEstimator, base.TransformerMixin):
  """Standardize custom features"""

  def __init__(self):
    self.mean_ = None
    self.std_ = None

  def fit(self, X, y=None):
    self.mean_ = sum(X) / len(X)
    self.std_ = (sum((x - self.mean_) ** 2 for x in X) / len(X)) ** 0.5
    return self

  def transform(self, X, y='deprecated', copy=True):
    if self.mean_ is None or self.std_ is None:
        raise ValueError("Scaler has not been fitted yet.")
    scaled_X = []
    for x in X:
        scaled_x = (x - self.mean_) / self.std_
        scaled_X.append(scaled_x)

    if isinstance(scaled_X, TupleOrList):
        return pd.concat(scaled_X, ignore_index=True).reset_index(drop=True)
    return scaled_X
  
def ExtractFile(file_path, extract_to='.'):
    """
    Extracts a compressed file to the specified directory.
    
    Args:
        file_path (str): The path to the compressed file.
        extract_to (str): The directory to extract the files to. Defaults to the current directory.
    """
    total_size = 0
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    try:
        if file_path.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                total_size = sum([info.file_size for info in zip_ref.infolist()]) / (1024 * 1024)
                print(f"Attention !!! Your chosen dataset will take {total_size:.2f} MB in local storage. Use Ctrl+C to abort before the process start.")
                for file in tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist()), desc="Extracting ZIP"):
                    zip_ref.extract(member=file, path=extract_to)

        elif file_path.endswith('.tar.gz') or file_path.endswith('.tgz'):
            with tarfile.open(file_path, 'r:gz') as tar_ref:
                total_size = sum([member.size for member in tar_ref.getmembers()]) / (1024 * 1024)
                print(f"Attention !!! Your chosen dataset will take {total_size:.2f} MB in local storage. Use Ctrl+C to abort before the process start.")
                for member in tqdm(iterable=tar_ref.getmembers(), total=len(tar_ref.getmembers()), desc="Extracting TAR.GZ"):
                    tar_ref.extract(member=member, path=extract_to)
        
        elif file_path.endswith('.rar'):
            with rarfile.RarFile(file_path, 'r') as rar_ref:
                total_size = sum([info.file_size for info in rar_ref.infolist()]) / (1024 * 1024)
                print(f"Attention !!! Your chosen dataset will take {total_size:.2f} MB in local storage. Use Ctrl+C to abort before the process start.")
                for file in tqdm(iterable=rar_ref.infolist(), total=len(rar_ref.infolist()), desc="Extracting RAR"):
                    rar_ref.extract(member=file, path=extract_to)
        else:
            print(f"Unsupported file format: {file_path}")
    except Exception as e:
        print(f"Failed to extract file: {file_path}")
        print(e)
    return total_size
# def Interpolate_with_gaussian_noise(data: pd.Series) -> pd.Series:
#   """Couldn't find a proper name. Very slow ..."""
#   DTYPE = np.float32

#   series = data.astype(DTYPE)
#   values = series.tolist()
#   processed = []

#   series_size = len(values)

#   prev_rssi = 0
#   prev_seq = -1
#   for seq, rssi in enumerate(values):
#     if not np.isnan(rssi):
#         avg_rssi = np.mean([prev_rssi, rssi])
#         std_rssi = np.std([prev_rssi, rssi])
#         std_rssi = std_rssi if std_rssi > 0 else np.nextafter(DTYPE(0), DTYPE(1))
#         diff = seq - prev_seq - 1

#         processed.extend(np.random.normal(avg_rssi, std_rssi, size=diff))
#         processed.append(rssi)
#         prev_seq, prev_rssi = seq, rssi

#   avg_rssi = np.mean([prev_rssi, 0.])
#   std_rssi = np.std([prev_rssi, 0.])
#   diff = series_size - prev_seq - 1
#   processed.extend(np.random.normal(avg_rssi, std_rssi, size=diff))

#   series = pd.Series(data=processed, index=data.index, dtype=DTYPE)
#   return series


# def Interpolate_with_constant(data: pd.Series, constant: int = 0) -> pd.Series:
#   """Interpolate missing values with constant value."""
#   return data.fillna(value=constant)


# class CustomInterpolation(base.BaseEstimator, base.TransformerMixin):
#   """Custom interpolation function to be used in"""

#   STRATEGIES_ALL = ['none', 'gaussian', 'constant']

#   def __init__(self, source: str, strategy: str = 'constant', constant: float = 0, target=None):
#     if strategy not in self.STRATEGIES_ALL:
#       raise ValueError(f'"{strategy}" is not available strategy')

#     self.strategy = strategy
#     self.constant = constant

#     self.source = source
#     self.target = source if target is None else target

#   def with_constant(self, data: pd.DataFrame) -> pd.DataFrame:
#     df = data.copy()
#     df[self.target] = df[self.source].fillna(value=self.constant)
#     return df

#   def with_gaussian(self, data: pd.DataFrame) -> pd.DataFrame:
#     df = data.copy()
#     df[self.target] = Interpolate_with_gaussian_noise(df[self.source])
#     return df

#   def with_none(self, data: pd.DataFrame) -> pd.DataFrame:
#     df = data.copy()
#     src = [self.source] if isinstance(self.source, [str]) else self.source
#     df = df.dropna(subset=src)
#     return df

#   def do_interpolation(self, X: pd.DataFrame) -> pd.DataFrame:
#     if self.strategy == 'constant':
#       return self.with_constant(X)

#     if self.strategy == 'gaussian':
#       return self.with_gaussian(X)

#     if self.strategy == 'none':
#       return self.with_none(X)

#     raise ValueError(f'"{self.strategy}" is not available strategy')

#   def fit(self, X: pd.DataFrame, y=None):
#     return self

#   def transform(self, X, y='deprecated', copy=True):
#     if isinstance(X, (List, Tuple,)):
#       with mp.Pool(processes=2) as p:
#           return p.map(self.do_interpolation, X)

#     return self.do_interpolation(X)


# class CustomSplitter(base.BaseEstimator, base.TransformerMixin):
#   def __init__(self, X: TupleOrList = None, y: str = 'class', drop: TupleOrList = None):
#     self.X = X
#     self.y = y
#     self.drop = drop

#   def fit(self, X: pd.DataFrame, y=None):
#     return self

#   def transform(self, df: pd.DataFrame, y='deprecated', copy=True):
#     df = df.copy() if copy else df
#     if self.drop:
#       df.drop(labels=self.drop, axis=1, inplace=True)

#     if self.X:
#       return df[self.X], df[self.y].ravel()

#     return df.drop(self.y), df[self.y].ravel()


# class SyntheticFeatures(base.BaseEstimator, base.TransformerMixin):
#   """Rolling window for mean & std features."""

#   def __init__(self, source: str, window_size: int = 10, target=None):
#     self.source = source
#     self.target = source if target is None else target

#     if not isinstance(window_size, int) or not window_size > 0:
#       raise ValueError(f'Window should be positive integer. Got `{window_size}` instead.')

#     self.window = window_size

#   def fit(self, X, y=None):
#     return self

#   def do_synthetics(self, data: pd.DataFrame) -> pd.DataFrame:
#     df = data.copy()
#     df[f'{self.target}_mean'] = df[self.source].rolling(self.window).mean()
#     df[f'{self.target}_std'] = df[self.source].rolling(self.window).std()
#     df[f'{self.target}_median'] = df[self.source].rolling(self.window).median()
#     return df

#   def transform(self, X: pd.DataFrame, y='deprecated', copy=True):
#     if isinstance(X, (List, Tuple,)):
#       with mp.Pool(processes=2) as p:
#           return p.map(self.do_synthetics, X)

#     return self.do_synthetics(X)


# def Poly_features(df: pd.DataFrame, include: List[str], degree: int, include_bias=False, *args,
#                 **kwargs) -> pd.DataFrame:
#   """The `PolynomialFeatures` from sklern drops/loses information about column names from pandas, which is not very convinient.
#   This is a workaround for this behaviour to preserve names.
#   """
#   X, excluded = df.loc[:, include], df.drop(include, axis=1)
#   poly = preprocessing.PolynomialFeatures(degree=degree, include_bias=include_bias, *args, **kwargs).fit(X)

#   # Next line converts back to pandas, while preserving column names
#   X = pd.DataFrame(poly.transform(X), columns=poly.get_feature_names(X.columns), index=X.index)

#   data = pd.concat([X, excluded], axis=1, )
#   data = data.reset_index(drop=True)

#   # Transform column names. Ex. 'rssi rssi_avg' -> 'rssi*rssi_avg'
#   data = data.rename(lambda cname: cname.replace(' ', '*'), axis='columns')

#   return data