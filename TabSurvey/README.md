In this guide, we will install and use TabSurvey to create a baseline for testing the performance of various models on your dataset.

**Note:** This tutorial assumes that you are working on a machine with a graphics card and have installed the appropriate CUDA driver to enable GPU usage.
Overview of the steps we will cover:
- Environment Setup
- Verification of Operation
- Adding a Dataset

In this guide, we will not install Docker as described in the original author's Git repository due to changes in library packages, which would require modifications to the Dockerfile. Additionally, for our specific use of TabSurvey, creating multiple environments and a Docker container to install Miniconda3 is unnecessary. Another reason is that we can use an existing IEC server with Conda, so we only need to set up the appropriate Python environment, making it easy to clone and edit code between the IEC server and your local machine.

## I. Environment Setup
### Install Conda
Install Miniconda3 on your LOCAL machine. You can refer to the official installation guide at: https://docs.conda.io/projects/miniconda/en/latest/#quick-command-line-install

- On WINDOWS, open the Command Prompt (CMD) and execute the following commands:
```bat
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o miniconda.exe
start /wait "" miniconda.exe /S
del miniconda.exe
```
After the installation, you can check and find the "Anaconda Prompt (Miniconda3)" program in the Start menu.

- On LINUX or WSL2, open a terminal and execute the following commands to install Miniconda3:
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```
After installation, launch Miniconda with the following commands for bash and zsh shells:
```bash
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
```

### Install enviroment
To install the Python 3.7 environment, since the original author of TabSurvey used the "tensorflow-gpu=1.15.0" package for GPU support, on WSL, we will install the "tensorflow-directml" package as an alternative (it supports Python 3.7 the best). 

To create the Python 3.7 environment, execute the following command:
```bash
conda create --name tabsurvey python=3.7
conda activate tabsurvey
```

On the IEC server, you can directly create the Python 3.7 environment without the previous steps:
```bash
conda create --name tabsurvey python=3.7
eval "$(conda shell.bash hook)"
conda activate tabsurvey
```

After activating the environment with `conda activate`, you will clone the TabSurvey repository. Note that this repository is a fork that you fixed to suit your specific task:
```bash
git clone https://github.com/ntruongn/TabSurvey.git
cd TabSurvey
```

Finally, you will install some necessary libraries for TabSurvey.
- WSL:
```sh
pip install -r requirements_wsl.txt
```
- Windows, Linux, IEC Server:
```bat 
pip install -r requirements.txt
```

## II. Verification of Operation

To check if everything is running smoothly on Linux, use the following command in the terminal to run the test:
```bash
./testall.sh
```

For Windows, you can execute the following batch file:
```bash
testall.bat
```

## III. Adding a New Dataset for Baseline Testing in TabSurvey

After completing the installation, we will add our own dataset to TabSurvey. To add a new dataset, pay attention to two files:

1. **Config File** with the `.yml` extension in the `config` directory.
   - Important information to specify includes:
     - `dataset`: The name of the dataset.
     - `objective`: Binary, classification, or regression task.
     - `direction`: Optimization direction (refer to the original article).
     - `num_features`: Total number of features in the dataset.
     - `num_classes`: Number of classes in a multi-class classification task. Set to 1 for binary classification or regression.
     - `cat_idx`: List of indices of categorical features in the dataset (if any).

2. **Modify load_data.py**:
   - In the `load_data()` function of the `load_data.py` file, you need to add an `elif` statement to handle the case when `args.dataset` matches the name of the dataset you are adding.
   - Within this `elif` block, perform the following:
     - Load the dataset from a CSV file (your dataset) either from the cloud using `pd.read_csv("path/on/cloud.csv")` or by adding it from a ZIP file. Below is an example of handling data loading from a ZIP file (downloaded from the cloud and specified as a local path).

3. **Setup Params**:
   - As mentioned earlier, training requires specific parameters, which are obtained from the `best_params.yml` file if `--optimize_hyperparameters` is not used. To run with a new dataset, you should run this option and use the best_params in the output file to overwrite `best_params.yml`. Make sure to use a GPU, and note the GPU index in the dataset's config file.
   - Modify the `python train` line in the `testall.sh` file to include the `--optimize_hyperparameters` and `--use_gpu` options. The line should look like this:
     ```bash
     python train.py --config "$config" --model_name "$model" --n_trials $N_TRIALS --epochs $EPOCHS --optimize_hyperparameters --use_gpu
     ```
   - You can review and execute the code in the following files:
     - `testall_optimize_gpu.bat` (on Windows)
     - `testall_optimize_gpu.sh` (on Linux)

After running these, the results will be stored in `"output/<model_name>/<data_name>/results.txt"`. You can then extract the results for reporting or create a new `best_params` file using the `get_best_params.py` script. Additionally, you can refer to the `get_report_f1.py` file to extract the necessary results for your specific task. The sample result for an F1 report extracted from this file is provided in your document.

In summary, the above is a basic guide for installing TabSurvey and using it with your own dataset. You can write additional scripts to process the results returned in the `output.txt` file for each model on each dataset to extract the necessary reporting tables for your task.
