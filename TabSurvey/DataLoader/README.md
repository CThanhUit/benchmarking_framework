# IDS DataLoader

* Version 0.1

## Dataset support 
|           | Name dataset  |   Status    |
|-----------|---------------|-------------|
|     1     |  CICIoT2023   |    Done     |
|     2     | CICMalMem2022 |    Done     |
|     3     |  CICIDS2018   |    Done     |
|     4     |  CICIDS2017   |    Done     |
|     5     |  CICDDoS2019  |    Done     |
### CICDIoT2023

Link download: https://www.unb.ca/cic/datasets/iotdataset-2023.html

Link paper: https://www.mdpi.com/1424-8220/23/13/5941

Function suppport: 

 * Show basic metadata
 * Download data
 * Load data
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
 * Show basic data analysis
 * To dataframe
 * To csv

### NetML
Link download: https://github.com/ACANETS/NetML-Competition2020

Link paper: https://arxiv.org/abs/2004.13006

Function suppport: 

 * Show basic metadata
 * Download data
 * Load data
 * Show basic data analysis
 * To dataframe
 * To csv

## How to use

```python
import DataLoader
from DataLoader import BaseLoadDataset

a = BaseLoadDataset(dataset_name='BaseDataset')
BaseDataset.Show_basic_metadata()
BaseDataset.DownLoad_Data()
BaseDataset.Load_Data()
BaseDataset.Show_basic_analysis()
df = BaseDataset.To_dataframe()
BaseDataset.To_csv()
```

### Function description ###
```
BaseLoadDataset(dataset_name='BaseDataset')

- Parameter:
    + seed = int
        Seed for random function
    + print_able = bool 
        Allow to print description

- Attribution:
    + data_df - dataframe for clean data
```
```
Show_basic_metadata()
```
```
DownLoad_Data(datadir = os.getcwd(),  load_type="preload")

- Parameter:
    + datadir = str
        Path to Data dir
    + load_type = "preload" or "raw"
        Download the dataset from the repository. There are 2 types of download: "preload" will download a small subset; "raw" will download the full dataset with a 3GB zip file, and unzip to over 28GB data.
        If the dataset is already at "datadir", it will not continue downloading data
```

```
Load_Data(datadir = os.getcwd(),  load_type="preload", frac = None, limit_cnt=sys.maxsize)

- Parameter:
    + datadir = str
        Path to Data dir
    + load_type = "preload" or "raw"
        Load data to dataframe. If using "raw" option, you should specific limit sample of each class by set number to "limit_cnt" because the dataset is very large. 
    + limit_cnt = int
        Maximun sample in each class.
    + frac = float
        Get data by ratio between [0.,1.]
```
```
Preprocess_Data(target_variable=None)

- Prameter:
    + target_variable = str
        A target variable is retained when cleaning data. If using None option, it means the target variable is default.
```
```
Show_basic_analysis()
```
```
To_dataframe():

- Return:
    + Current object dataframe
```
```
To_csv(datadir = os.getcwd())

- Parameter:
    + datadir = str
        Path to Data dir
```