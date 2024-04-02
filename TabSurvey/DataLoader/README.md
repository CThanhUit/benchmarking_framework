# IDS DataLoader

* Version 0.1

## Dataset support 
|           | Name dataset  |   Status    |
|-----------|---------------|-------------|
|     1     |  CICIoT2023   |    Done     |
|     2     | CICMalMem2022 |    Done     |
|     3     |  CICDDoS2019  |    Done     |
|     4     |  CICIDS2018   |    Done     |
|     5     |  CICIDS2017   |    Done     |
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

## How to use

```python
import DataLoader
from DataLoader import BaseLoadDataset

a = BaseLoadDataset(dataset_name='BaseDataset', seed = 42, print_able = True)
BaseDataset.Show_basic_metadata()
BaseDataset.DownLoad_Data(datadir = os.getcwd(),  load_type="raw")
BaseDataset.Load_Data(datadir = os.getcwd(),  load_type="raw", frac = None, limit_cnt=sys.maxsize)
BaseDataset.Show_basic_analysis()
df = BaseDataset.To_dataframe()
BaseDataset.To_csv(datadir = os.getcwd())
```

### Function description ###
```
BaseLoadDataset(dataset_name = 'BaseDataset', seed = 42, print_able = True)

- Parameter:
    + dataset_name = 'BaseDataset'
        Name of dataset.
    + seed = int
        Seed for random function
    + print_able = bool 
        Allow to print description

- Attribution:
    + data_df - dataframe
```
```
DownLoad_Data(datadir = os.getcwd(),  load_type="raw")
    Download dataset
- Parameter:
    + datadir = str
        Path to Data dir
    + load_type = "preload" or "raw". Default is "raw"
        Download the dataset from the repository. There are 2 types of download: "preload" will download a small subset; "raw" will download the full dataset with a 3GB zip file, and unzip to over 28GB data.
        If the dataset is already at "datadir", it will not continue downloading data
```

```
Load_Data(datadir = os.getcwd(),  load_type="raw", frac = None, limit_cnt=sys.maxsize)
    Load dataset.
- Parameter:
    + datadir = str
        Path to Data dir
    + load_type = "preload" or "raw".  Default is "raw"
        Load data to dataframe. If using "raw" option, you should specific limit sample of each class by set number to "limit_cnt" because the dataset is very large. 
    + limit_cnt = int
        Maximun sample in each class.
    + frac = float
        Get data by ratio between [0.,1.]
```
```
Show_basic_metadata()
    Displays some information about the dataset.
```
```
Show_basic_analysis()
    Displays some basic analysis of the dataframe.
```
```
To_dataframe():
    Export to dataframe.
- Return:
    + Current object dataframe
```
```
To_csv(datadir = os.getcwd())

    Export to file *.csv.
- Parameter:
    + datadir = str
        Path to save file.
```
## How to add new dataset
-   DataLoader currently only supports loading data sets in *.csv format
### **Config File**:
-   Add the config file config_{datasetname}.yml to DataLoader/config:
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