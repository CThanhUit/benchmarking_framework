from DataLoader.utils import *
from zipfile import ZipFile, is_zipfile
from tqdm.auto import tqdm
import time 

SEED = 42

# Rules for naming variables and functions

# Public:
# Method: 
    # - Viết in hoa chữ cái đầu
    # - Mỗi từ cách nhau bằng dấu "_"
# Attribute:
    # - viết hoa chữ cái đầu
    # - Mỗi từ cách nhau dấu "_"
    # - Có "_" ở trước

# Private:
# Method:
    # - Ko viết in hoa
    # - Mỗi từ cách nhau dấu "_"
    # - Có "_" ở trước
# Attribute:
    # - viết thường toàn bộ
    # - Mỗi từ cách nhau dấu "_"
    # - Có "_" ở trước



class BaseModel():
  
  def __init__(base_self, seed = SEED, print_able = True, save_csv = True) -> None:
    np.random.seed(seed)
    np.set_printoptions(suppress=True)
    base_self.__PRINT_ABLE = print_able
    base_self.__SAVE_CSV = save_csv
    base_self.__data_df = pd.DataFrame()
    base_self.__target_variable = None
    base_self._fts_names = []

    base_self._real_cnt = {}  # Actual number of samples in the csv file of each class
    base_self._label_map = {} # Map the actual label in the csv file with the label in the paper

    base_self._category_map = {} # Group the actual label in the csv file with the coresponding category in the paper
    base_self._binary_map = {}
    base_self._label_true_name = [] # Map the actual label in the csv file with the label in the paper

    base_self._label_drop = [] # List the label be dropped
    base_self._label_cnt = {} # Actual number of samples loaded by function
    base_self._error_cnt = 0
    base_self._fixLabel()
    base_self._set_metadata()

  def _set_metadata(base_self) -> None:
    base_self._ds_name = ""
    base_self._ds_size = ""
    base_self._ds_fts = []
    base_self._ds_label = base_self._label_true_name
    base_self._ds_original_link = ""
    base_self._ds_paper_link = ""
    # base_self._ds_name = ""


  def _print(base_self, str) -> None:
    if base_self._Print_able:
      print(str)

  def _add_mode_features(base_self, dataset, FLAG_GENERATING: bool = False) -> pd.DataFrame:
    pass
  
  def _fixLabel(base_self):
    for x in base_self._label_map:
      y = base_self._label_map[x]
      if y not in base_self._real_cnt:
        base_self._real_cnt[y] = 0
      base_self._real_cnt[y] += base_self._real_cnt[x]
      base_self._real_cnt.pop(x)
    # base_self._print("True count:")
    # base_self._print(base_self._real_cnt)

  def _reDefineLabel_by_Category(base_self):
    base_self._data_df.rename(columns = {" Label": "Label"}, inplace = True, errors='ignore')
    base_self._data_df['Category'] = base_self._data_df['Label'].apply(lambda x: base_self._category_map[x] if x in base_self._category_map else x)
    # data_df.drop(data_df[data_df['Label'] not in _category_map].index, inplace = True)
    return base_self._data_df



  def _load_raw_default(base_self, dir_path, limit_cnt:sys.maxsize, frac = None):
    # base_self._data_df = df_ans
    pass
          

  # def _prepare_datasets(base_self, PATH_TO_DATA) -> pd.DataFrame:
    
  #   return df

  def _download(base_self, url, filename):
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
  

  # Public API
  #=========================================================================================================================================
  #=========================================================================================================================================
  
  
  def Add_more_fts(base_self):
    #Add more features
    base_self._print("=================================== Add more features ===================================")
    base_self._data_df = base_self._add_mode_features(base_self._data_df)
    base_self._data_df.dropna(inplace =True)
    base_self._data_df.drop(columns=['1'],inplace =True)
    base_self._print("=================================== Done Add more features ===================================")
    return


  #=========================================================================================================================================
  
  
  def DownLoad_Data(base_self, datadir = os.getcwd(),  load_type= "full"):
    print(f"Attention !!! Your chosen dataset will take {base_self.datasize} MB in local storage. Use Ctrl+C to abort before the process start.")
    time.sleep(7)
    pass

  #=========================================================================================================================================

  def Load_Data(base_self, datadir = os.getcwd(),  load_type="full", limit_cnt=sys.maxsize, frac = None):
    pass

#=========================================================================================================================================
  
  
  # def Train_test_split(base_self, testsize=0.2):
  #   base_self._print("=================================== Begin Split File ===================================")
  #   df = base_self._data_df.drop(columns=['prr'])

  #   print("=================================== Dataframe be like:")
  #   print('\n' + tabulate(base_self._data_df.head(5), headers='keys', tablefmt='psql'))

  #   # np_data = df.to_numpy(copy=True)
  #   X_train, X_test, y_train, y_test = train_test_split(df.drop(columns = ['target']).to_numpy(copy=True), 
  #                                                       df['target'].to_numpy(copy=True),    
  #                                                       test_size=testsize, random_state=42)


  #   # y_train = LabelEncoder().fit_transform(y_train)
  #   # y_test = LabelEncoder().fit_transform(y_test)


  #   # X_train = StandardScaler().fit_transform(X_train)

  #   # X_test = StandardScaler().fit_transform(X_test)

  #   print("Training data shape:",X_train.shape, y_train.shape)
  #   print("Testing data shape:",X_test.shape, y_test.shape)

  #   print("Label Train count:")
  #   unique= np.bincount(y_train)
  #   print(np.asarray((unique)))
  #   print("Label Test count:")
  #   unique= np.bincount(y_test)
  #   print(np.asarray((unique)))
  #   base_self._print("=================================== Split File End===================================")
  #   return X_train, X_test, y_train, y_test


  #=========================================================================================================================================
  
  
  def Show_basic_metadata(base_self):
    print("=================================== Show dataset metadata ===================================")
    print("Dataset name:", base_self._ds_name)
    print("Original public at paper:", base_self._ds_paper_link)
    print("Original public at link:", base_self._ds_original_link)
    print("Total size on disk (MB):", base_self._ds_size)
    print("Total ",len(base_self._ds_fts), "features: ")
    print(base_self._ds_fts)
    print("Total ",len(base_self._ds_label), "classes: ")
    print(base_self._ds_label)

  def Show_basic_analysis(base_self):
    print("=================================== Show basic analysis of data frame ===================================")
    print("=================================== Dataframe be like:")
    print('\n' + tabulate(base_self._data_df.head(5), headers='keys', tablefmt='psql'))
    print("=================================== Data info:")
    print(base_self._data_df.info())
    print("=================================== Label distribution")
    print(base_self._data_df["Label"].value_counts())
    # plt.show()


  #=========================================================================================================================================
  
  
  def To_dataframe(base_self):
    return base_self._data_df

  def To_csv(base_self, datadir = os.getcwd()):
    file_name = base_self.__ds_name + "_dataloader.csv"
    data_file = os.path.join(datadir, file_name)
    if os.path.exists(data_file) == True:
      print("File is already exists at path:", data_file)
    else:
      base_self.__data_df.to_csv(data_file, index=True)
      print("File saved at:", data_file)
    return
