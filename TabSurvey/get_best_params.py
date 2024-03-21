import re
import yaml
import os
import glob
import argparse
from torch import device


def get_best_params(data_name):
    """
    Extract information about "Best Parameters" from "results.txt" files for a specific dataset.

    Parameters:
    - data_name (str): The name of the dataset for which "Best Parameters" information is to be extracted.

    Returns:
    - data_best_params (dict): A dictionary containing "Best Parameters" information for each corresponding model.
    """
    path_pattern = f"output/*/{data_name}/results.txt"
    # Use glob to find all matching files
    matching_files = glob.glob(path_pattern)

    # Create a dictionary to store the best parameters for each model
    data_best_params = {}

    # Iterate through the matching files
    for file_path in matching_files:
        # Extract the wildcard portion from the path pattern to determine the model name
        model_name = file_path.split(os.sep)[-3]

        # Read the content of the results.txt file
        with open(file_path, 'r') as file:
            content = file.read()

        matches = re.findall(r"Best Parameters: (.+)", content)
        if matches:
            # Lấy giá trị cuối cùng trong danh sách kết quả
            best_params_str = matches[-1]
        else:
            print(f"Best parameters not found in {file_path}.")
            continue
        
        # print(best_params_str)

        # Convert the extracted string to a Python dictionary
        best_params = eval(best_params_str)
        # Add the best parameters to the dictionary for the current model
        data_best_params[model_name] = best_params
        # print(best_params_dict)
    return data_best_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract best parameters from files.")
    parser.add_argument("--output_file", default="config/best_params.yml", help="Output YAML file name")
    parser.add_argument("--data_names", nargs='+', required=True, help="List of data names")
    
    args = parser.parse_args()
    
    output_file = args.output_file
    data_names = args.data_names

    best_params_dict = {'parameters': {}}   
    for data_name in data_names:
        data_best_params = get_best_params(data_name)
        best_params_dict['parameters'][data_name] = data_best_params
        # Write the dictionary to the output YAML file
    with open(output_file, 'w') as yaml_file:
        yaml.dump(best_params_dict, yaml_file, default_flow_style=False)

    print("Best parameters extracted and saved to", output_file)