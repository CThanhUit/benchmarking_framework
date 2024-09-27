# Network Intrusion Detection System DataLoader

This project aggregates wellknown datasets using for building NIDS. We also provide dataloader for each datasets and support function for preprocessing data.
* Version 0.1


## Dataset support 
|           | Name dataset  |   Status    |
|-----------|---------------|-------------|
|     1     |  CICIoT2023   |    Done     |
|     2     | CICMalMem2022 |    Done     |
|     3     |  CICIDS2018   |    Done     |
|     4     |  CICIDS2017   |    Done     |
|     5     |    ToNIoT     |    Done     |
|     6     |    BoTIoT     |    Done     |
|     7     |    N_BaIoT    |    Done     |
|     8     |  CICDDoS2019  |    Done     |
|     9     |   UNSWNB15    |    None     |
|     10    |   CIDDS001    |    None     |
|     11    | EnrichingIoT  |    None     |
|     12    |     NetML     |    None     |
### CICDIoT2023

Link download: https://www.unb.ca/cic/datasets/iotdataset-2023.html

Link paper: https://www.mdpi.com/1424-8220/23/13/5941

Function suppport: 

 * Show basic metadata
 * Download data
 * Load data
 * Preprocess Data
 * Show basic data analysis
 * To dataframe
 * To csv

### CICMalMem2022

Link download: https://www.unb.ca/cic/datasets/iotdataset-2023.html

Link paper: https://pdfs.semanticscholar.org/b2e2/0dc7a34753311472a5f2314fbf866d7eddd0.pdf

Function suppport: 

 * Show basic metadata
 * Download data
 * Load data
 * Preprocess Data
 * Show basic data analysis
 * To dataframe
 * To csv

### CICDDoS2019

Link download: https://www.unb.ca/cic/datasets/ddos-2019.html

Link paper: https://ieeexplore.ieee.org/abstract/document/8888419/

Function suppport: 

 * Show basic metadata
 * Download data
 * Load data
 * Preprocess Data
 * Show basic data analysis
 * To dataframe
 * To csv

### CICIDS2018

Link download: https://www.unb.ca/cic/datasets/ids-2018.html

Link paper: https://www.semanticscholar.org/paper/Toward-Generating-a-New-Intrusion-Detection-Dataset-Sharafaldin-Lashkari/a27089efabc5f4abd5ddf2be2a409bff41f31199

Function suppport: 

 * Show basic metadata
 * Download data
 * Load data
 * Preprocess Data
 * Show basic data analysis
 * To dataframe
 * To csv

### CICIDS2017

Link download: https://www.unb.ca/cic/datasets/ids-2017.html

Link paper: https://www.semanticscholar.org/paper/Toward-Generating-a-New-Intrusion-Detection-Dataset-Sharafaldin-Lashkari/a27089efabc5f4abd5ddf2be2a409bff41f31199

Function suppport: 

 * Show basic metadata
 * Download data
 * Load data
 * Preprocess Data
 * Show basic data analysis
 * To dataframe
 * To csv

### ToNIoT
Link download: https://research.unsw.edu.au/projects/toniot-datasets

Link paper: https://www.researchgate.net/publication/352055999_A_new_distributed_architecture_for_evaluating_AI-based_security_systems_at_the_edge_Network_TON_IoT_datasets

Function suppport: 

 * Show basic metadata
 * Download data
 * Load data
 * Preprocess Data
 * Train test split
 * Show basic data analysis
 * To dataframe
 * To csv

<!-- ### NetML
Link download: https://github.com/ACANETS/NetML-Competition2020

Link paper: https://arxiv.org/abs/2004.13006

Function suppport: 

 * Show basic metadata
 * Download data
 * Load data
 * Preprocess Data
 * Show basic data analysis
 * To dataframe
 * To csv -->

## How to use

```python
from DataLoader import CICDIoT2023

ciciot2023 = CICDIoT2023()
ciciot2023.Show_basic_metadata()
ciciot2023.DownLoad_Data()
ciciot2023.Load_Data()
ciciot2023.Preprocess_Data()
ciciot2023.Train_test_split()
ciciot2023.Show_basic_analysis()
df = ciciot2023.To_dataframe()
ciciot2023.To_csv()
```

### Function description ###
```
ciciot2023(seed = 42, print_able = True)
 
CICIoT2023 dataset class.
- Parameter:
    + seed = int:
        Seed for random function. Default is 42.
    + print_able = bool:
        Allow to print description. Default is True.
```
```
Show_basic_metadata()

    Show basic metadata of the dataset.
- Parameters:
    None
- Returns:
    None
```
```
DownLoad_Data(datadir = os.getcwd(),  load_type="preload")

Download the dataset.
- Parameters:
    + path = str:
        Path to save the dataset. Default is current working directory.
    + load_type = "preload" or "raw":
        Type of dataset to download:There are 2 types of download: "preload" will download a .csv file; "raw" will download the full dataset with a zip file. Default is "raw".
- Returns:
    None
```

```
Load_Data(datadir = os.getcwd(),  load_type="preload", frac = None, limit_cnt=sys.maxsize)

Load the dataset.
- Parameters:
    + path = str:
        Path to the folder containing the dataset. Default is current working directory.
    + load_type = "preload" or "raw":
        Load data to dataframe. If using "raw" option, you should specific limit sample of each class by set number to "limit_cnt" because the dataset is very large. Default is "raw".
    + limit_cnt = int:
        Maximun sample in each class. Default is sys.maxsize. Only used when load_type="raw"
    + frac = float between [0.,1.]:
        Get data by ratio. Default is None. Only used when load_type="raw"
- Returns:
    None
```
```
Preprocess_Data(target_variable=None)

Preprocess data by cleaning, encoding, scaling, and selecting features.
- Parameters:
    + drop_cols = list:
        List of columns to drop. Default is None.
    + type_encoder = "LabelEncoder", "OneHotEncoder" or "CustomEncoder": 
        Type of encoder to encode features. Default is "LabelEncoder".
    + type_scaler = "StandardScaler", "MinMaxScaler", "QuantileTransformer" or "CustomScaler":
        Type of scaler to scale features. Default is "QuantileTransformer".
    + type_select = SelectKBest:
        Type of feature selection method. Default is "SelectKBest".
    + num_fts = int:
        Number of features to keep. Default is 'all'.
- Returns:
    None
```
```
Train_test_split(testsize=0.2, target_variable=None , type_classification='multiclass')

Split data into training and testing sets.
- Parameters:
    + testsize = float between [0.,1.]:
        Size of the testing set. Default is 0.2.
    + target_variable = string:
        Name of column contains the data label. Default is None.
    + type_classification = "multiclass" or "binary":
        Type of classification. Default is 'multiclass'.
- Returns:
    + X_train = np.array:
        Training features.
    + X_test = np.array:
        Testing features.
    + y_train = np.array:
        Training labels.
    + y_test = np.array:
        Testing labels.
```
```
Show_basic_analysis()

Show basic metadata of the dataset.
- Parameters:
    None
- Returns:
    None
```
```
To_dataframe():

Return the dataset as a pandas DataFrame.
- Parameters:
    None
- Returns:
    A pandas DataFrame
```
```
To_csv(datadir = os.getcwd())

Save the dataset as a csv file.
- Parameters:
    + path = str:
        Path to save the csv file. Default is current working directory.
- Returns:
    None
```
