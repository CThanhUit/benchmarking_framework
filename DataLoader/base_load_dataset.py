from DataLoader.utils import *
import os
import sys
from tabulate import tabulate
from zipfile import ZipFile, is_zipfile
from tqdm.auto import tqdm
import time 
import yaml
import numpy as np
import pandas as pd
SEED = 42

class BaseLoadDataset():

  def __init__(base_self, dataset_name = None, seed = SEED, print_able = True) -> None:
    np.random.seed(seed)
    np.set_printoptions(suppress=True)
    base_self.__PRINT_ABLE = print_able
    base_self.__data_df = pd.DataFrame()

    base_self.__ds_name = None
    base_self.__ds_size = None
    base_self.__ds_paper_link = None
    base_self.__ds_link = None
    base_self.__csv_link = None
    base_self.__target_variable = None
    base_self.__fts_names = []
    base_self.__real_cnt = {}  # Actual number of samples in the csv file of each class
    base_self.__label_map = {} # Map the actual label in the csv file with the label in the paper
    base_self.__category_map = {} # Group the actual label in the csv file with the coresponding category in the paper
    base_self.__binary_map = {}
    base_self.__label_true_name = [] # Map the actual label in the csv file with the label in the paper
    base_self.__label_drop = [] # List the label be dropped
    base_self.__label_cnt = { } # Actual number of samples loaded by function
    base_self.__error_cnt = 0
    base_self.__set_config(dataset_name)
    base_self.Show_basic_metadata()
    base_self.__fixLabel()


    # Set config from *.yml
  def __set_config(base_self, dataset_name = None):
    if dataset_name is None:
      print("Dataset not found!")
    else:
      config_setting_path = 'DataLoader/config/config_' + dataset_name + '.yml'
      config_settings = {}
      if os.path.exists(config_setting_path)==False:
        raise FileNotFoundError(f'Not found `{config_setting_path}`!')
      with open(config_setting_path, 'r') as config_file:
        config_settings = yaml.safe_load(config_file)
      try:
        base_self.__target_variable = (config_settings['target_variable'])
        base_self.__fts_names = (config_settings['fts_names'])
        base_self.__real_cnt = (config_settings['real_cnt'])
        base_self.__label_map = (config_settings['label_map'])
        base_self.__category_map = (config_settings['category_map'])
        base_self.__binary_map = (config_settings['binary_map'])
        base_self.__label_true_name = (config_settings['label_true_name'])
        base_self.__label_drop = (config_settings['label_drop'])
        base_self.__ds_name = (config_settings['ds_name'])
        base_self.__ds_size = (config_settings['ds_size'])
        base_self.__ds_paper_link = (config_settings['ds_paper_link'])
        base_self.__ds_link = (config_settings['ds_link'])
        base_self.__csv_link = (config_settings['csv_link'])
      except KeyError as e:
        print(f"KeyError: {e} in {config_setting_path}")

  def __print(base_self, str) -> None:
    if base_self.__PRINT_ABLE:
        print(str)

  def __add_mode_features(base_self, dataset, FLAG_GENERATING: bool = False) -> pd.DataFrame:
    pass

  #=========================================================================================================================================

  def __fixLabel(base_self):
    for x in base_self.__label_map:
      y = base_self.__label_map[x]
      if y not in base_self.__real_cnt:
        base_self.__print('label map ' + x)
        base_self.__print('real_cnt ' + y)
        base_self.__real_cnt[y] = 0
        base_self.__real_cnt[y] += base_self.__real_cnt[x]
        base_self.__real_cnt.pop(x)
    base_self.__print("True count:")
    base_self.__print(base_self.__real_cnt)

  #=========================================================================================================================================

  def __reDefineLabel(base_self):
    # base_self.__data_df.rename(columns = {base_self.__target_variable: 'Label'}, inplace = True, errors='ignore')
    # base_self.__target_variable='Label'
    base_self.__data_df['Category_dtloader'] = base_self.__data_df[base_self.__target_variable].apply(lambda x: base_self.__category_map[x] if x in base_self.__category_map else x)
    base_self.__data_df['Binary_dtloader'] = base_self.__data_df[base_self.__target_variable].apply(lambda x: base_self.__binary_map[x] if x in base_self.__binary_map else x)
    return base_self.__data_df

  #=========================================================================================================================================

  def __load_raw_default(base_self, dir_path, limit_cnt:sys.maxsize, frac = None):
    base_self.__label_cnt = {}
    base_self.__Null_cnt = 0

    tt_time = time.time()
    df_ans = pd.DataFrame()
    for root, _, files in os.walk(dir_path):
        for file in files:
            base_self.__print("Begin file " + file)
            if not file.endswith(".csv"):
                continue
            list_ss = []
            time_file = time.time()
            for chunk in pd.read_csv(os.path.join(root,file), index_col=None, names=base_self.__fts_names, header=0, chunksize=10000, low_memory=False):
                # # This command is only for CICmMalMem2022
                if base_self.__ds_name == 'CICMalMem2022':
                  chunk[base_self.__target_variable] = ['-'.join(value.split('-')[:2]) for value in chunk[base_self.__target_variable]]
                
                # # This command is only for UNSWNB15
                # chunk.fillna('Normal', inplace=True)

                dfse = chunk[base_self.__target_variable].value_counts()
                
                for x in dfse.index:
                    if x in base_self.__label_drop:
                      continue

                    sub_set = chunk[chunk[base_self.__target_variable] == x]
                    x_cnt = float(dfse[x])

                    if x in base_self.__label_map:
                      x = base_self.__label_map[x]
                    if x not in base_self.__label_cnt:
                        base_self.__label_cnt[x] = 0
                    if limit_cnt == base_self.__label_cnt[x] :
                        continue

                    max_cnt_chunk = min(int(limit_cnt * (x_cnt / base_self.__real_cnt[x]) + 1), sub_set.shape[0])
                    if frac != None:
                        max_cnt_chunk = min(int(frac * x_cnt + 1), sub_set.shape[0])
                    max_cnt_chunk = min(max_cnt_chunk, limit_cnt - base_self.__label_cnt[x])
                    sub_set = sub_set.sample(n=max_cnt_chunk,replace = False, random_state = SEED)
                    sub_set[base_self.__target_variable] = sub_set[base_self.__target_variable].apply(lambda y: base_self.__label_map[y] if y in base_self.__label_map else y)
                    list_ss.append(sub_set)
                    base_self.__label_cnt[x] += sub_set.shape[0]
            df_ans = CustomMerger().fit_transform([df_ans] + list_ss)
        
            base_self.__print("Update label:")
            base_self.__print(base_self.__label_cnt)
            base_self.__print("Time load:" + time.time() + time_file)
            base_self.__print(f"========================== Finish {file} =================================")

    
    print("Total time load:", time.time() - tt_time)
    base_self.__data_df = df_ans
          
  #=========================================================================================================================================

  def __download(base_self, url, filename):
    import functools
    import pathlib
    import shutil
    import requests
    
    r = requests.get(url, stream=True, allow_redirects=True, verify = False)
    if r.status_code != 200:
      r.raise_for_status()  # Will only raise for 4xx codes, so...
      raise RuntimeError(f"Request to {url} returned status code {r.status_code}")
    file_size = int(r.headers.get('Content-Length', 0))
    print("Start download file - Total file_size:", file_size)
    path = pathlib.Path(filename).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    desc = "(Unknown total file size)" if file_size == 0 else ""
    r.raw.read = functools.partial(r.raw.read, decode_content=True)  # Decompress if needed
    with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc) as r_raw:
      with path.open("wb") as f:
        shutil.copyfileobj(r_raw, f)

    return path  
  
  #=========================================================================================================================================

  def Add_more_fts(base_self):
    #Add more features
    print("===================================== Add more features ====================================")
    base_self.__data_df = base_self._add_mode_features(base_self._data_df)
    base_self.__data_df.dropna(inplace =True)
    base_self.__data_df.drop(columns=['1'],inplace =True)
    print("================================== Done Add more features ==================================")
    return


  #=========================================================================================================================================
  
  def DownLoad_Data(base_self, datadir = os.getcwd(),  load_type="raw"):
    print(f"Attention !!! Your chosen dataset will take {base_self.__ds_size} MB in local storage. Use Ctrl+C to abort before the process start.")
    time.sleep(7)     
    if load_type=="preload":
      datapath = os.path.join(datadir, (base_self.__ds_name + "_dataloader.csv"))
      data_url = base_self.__csv_link
      if data_url is None:
        print("Not supported!!!")
        return
      if os.path.exists(datapath) == True:
        print("Data already!!! No need to download.")
        return
      else:
        print("======================== File Data not found!!! Start downloading ==========================")
        print("File saved at:", base_self.__download(data_url, datapath))
        print("================================== End download data =======================================")
        return 
    
    if load_type=="raw":
      data_dir = os.path.join(datadir,  base_self.__ds_name)
      data_file = os.path.join(datadir,  (base_self.__ds_name + ".zip"))
      data_url = base_self.__ds_link
      if os.path.exists(data_dir) == True:
        print("Data already!!! No need to download.")
        return
      else:
        print("======================== Folder Data not found!!! Start downloading ========================")
        if os.path.exists(data_file) == False:
          print("======================== File Data Zip not found!!! Start downloading ======================")
          print("File saved at:", base_self.__download(data_url, data_file))
          print("===================================== End download data ====================================")
        if is_zipfile(data_file):
          print("====================================== Unzipping Data!!! ===================================")
          os.makedirs(data_dir, exist_ok=True)
          with ZipFile(data_file,"r") as zip_ref:
            for file in tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist())):
              zip_ref.extract(member=file, path=data_dir)
          print("File saved at:", datadir)
          print("===================================== End download data ====================================")
        else:
          print("=================================== Zip file not valid!!! ==================================")
      return 

  #=========================================================================================================================================

  def Load_Data(base_self, datadir = os.getcwd(),  load_type="raw",limit_cnt=sys.maxsize, frac = None):
    if load_type=="preload":
      datapath = os.path.join(datadir, (base_self.__ds_name + "_dataloader.csv"))
      if os.path.exists(datapath) == True:
        print("====================================== Start load data =======================================")
        base_self.__data_df =  pd.read_csv(datapath, index_col=None, header=0)
        base_self.__reDefineLabel()
        print("======================================== Data loaded =========================================")
        return
      else:
        print("=================================== File Data not found!!! ===================================")
        return 
    
    if load_type=="raw":
      datapath = os.path.join(datadir, base_self.__ds_name)
      if os.path.exists(datapath) == True:
        print("====================================== Start load data =======================================")
        base_self.__load_raw_default(datapath, limit_cnt, frac)
        base_self.__reDefineLabel()
        print("======================================== Data loaded =========================================")
        return
      else:
        print("===================== Folder Data not found!!! Please download first =========================")
      return

  #=========================================================================================================================================
  
  # def Clean_Data(base_self, df, target_variable=None):
  #   # Remove duplicated samples (rows)
  #   df = df.drop_duplicates()
  #   # Impute missing data or infinite values with mean value of each feature
  #   if target_variable is None:
  #       X = df
  #   else:
  #       X = df.drop(target_variable, axis=1)
  #   # Remove zero features (columns)
  #   X = X.loc[:, (X != 0).any()]    
  #   # Remove duplicated features (columns)
  #   X = X.loc[:, ~X.columns.duplicated()]

  #   X.replace([np.inf, -np.inf], np.nan, inplace=True)

  #   # Concatenate the target variable (if any) and the reduced feature DataFrame
  #   if target_variable in df.columns:
  #       y = df[target_variable]
  #       y.reset_index(drop=True, inplace=True)
  #       X.reset_index(drop=True, inplace=True)
  #       X = pd.concat([X, y], axis=1)

  #   return X        

  #=========================================================================================================================================

  def Show_basic_metadata(base_self):
    print("=================================== Show dataset metadata ===================================")
    print("Dataset name:", base_self.__ds_name)
    print("Original public at paper:", base_self.__ds_paper_link)
    print("Dataset link:", base_self.__ds_link)
    print("Total size on disk (MB):", base_self.__ds_size)
    print("Total ",len(base_self.__fts_names), "features: ")
    print(base_self.__fts_names)
    print("Total ",len(base_self.__label_true_name), "classes: ")
    print(base_self.__label_true_name)

  def Show_basic_analysis(base_self):
    print("============================= Show basic analysis of data frame ==============================")
    print("====================================== Dataframe be like =====================================")
    print('\n' + tabulate(base_self.__data_df.head(5), headers='keys', tablefmt='psql'))
    print("========================================== Data info =========================================")
    print(base_self.__data_df.info())
    print("====================================== Label distribution ====================================")
    print(base_self.__data_df[base_self.__target_variable].value_counts())

  #=========================================================================================================================================
  
  def To_dataframe(base_self):
    return base_self.__data_df
  
  #=========================================================================================================================================

  def To_csv(base_self, datadir = os.getcwd()):
    file_name = base_self.__ds_name + "_dataloader.csv"
    data_file = os.path.join(datadir, file_name)
    if os.path.exists(data_file) == True:
      print("File is already exists at path:", data_file)
    else:
      base_self.__data_df.to_csv(data_file, index=True)
      print("File saved at:", data_file)
    return