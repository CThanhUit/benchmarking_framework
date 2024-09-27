import numpy as np
import pandas as pd
import os
import pickle
import datetime
import json
import matplotlib.pyplot as plt
import itertools
output_dir = "output/"


def save_loss_to_file(args, arr, name, extension=""):
    filename = get_output_path(args, directory="logging", filename=name, extension=extension, file_type="txt")
    np.savetxt(filename, arr)


def save_predictions_to_file(arr, args, extension=""):
    filename = get_output_path(args, directory="predictions", filename="p", extension=extension, file_type="npy")
    np.save(filename, arr)


def save_model_to_file(model, args, extension=""):
    filename = get_output_path(args, directory="models", filename="m", extension=extension, file_type="pkl")
    pickle.dump(model, open(filename, 'wb'))


def load_model_from_file(model, args, extension=""):
    filename = get_output_path(args, directory="models", filename="m", extension=extension, file_type="pkl")
    return pickle.load(open(filename, 'rb'))


def save_results_to_json_file(args, jsondict, resultsname, append=True):
    """ Write the results to a json file. 
        jsondict: A dictionary with results that will be serialized.
        If append=True, the results will be appended to the original file.
        If not, they will be overwritten if the file already exists. 
    """
    filename = get_output_path(args, filename=resultsname, file_type="json")
    if append:
        if os.path.exists(filename):
            old_res = json.load(open(filename))
            for k, v in jsondict.items():
                old_res[k].append(v)
        else:
            old_res = {}
            for k, v in jsondict.items():
                old_res[k] = [v]
        jsondict = old_res
    json.dump(jsondict, open(filename, "w"))


def save_results_to_file(args, results, train_time=None, test_time=None, best_params=None):
    filename = get_output_path(args, filename="results", file_type="txt")

    with open(filename, "a") as text_file:
        text_file.write(str(datetime.datetime.now()) + "\n")
        text_file.write(args.model_name + " - " + args.dataset + "\n\n")

        for key, value in results.items():
            text_file.write("%s: %.5f\n" % (key, value))

        if train_time:
            text_file.write("\nTrain time: %f\n" % train_time)

        if test_time:
            text_file.write("Test time: %f\n" % test_time)

        if best_params:
            text_file.write("\nBest Parameters: %s\n\n\n" % best_params)
            
def save_plot_confusion_matrix(args ,cm, target_names, extension=None, figsize=None, cmap=None, normalize=False):
  
    filename = get_output_path(args, directory="Confusion_matrix", filename="cfm", extension=extension, file_type="png")
    
    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy
    if figsize is None:
        figsize = (15, 12)
    else:
        x, y = figsize
        if x < 15:
            figsize = (15, 12)
    if cmap is None:
        cmap = plt.get_cmap('Blues')
    norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]*100
    plt.figure(figsize=figsize)
    plt.imshow(norm_cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, fontsize='large')
        plt.yticks(tick_marks, target_names, fontsize='large')
    
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    thresh= 50
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:,}\n{:0.2f}%".format(cm[i, j], norm_cm[i, j]),
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if norm_cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label', fontsize='x-large')
    plt.xlabel('Predicted label', fontsize='x-large')
    plt.savefig(filename, bbox_inches='tight')
    plt.show()

def save_reports_to_csv_file(args, results, train_time=None, test_time=None):
    new_data = {}
    new_data["Name"]=args.dataset
    new_data["Model"]=args.model_name
    for key, value in results.items():
      new_data[key]=round(value, 5)
    new_data["train_time"]=train_time
    new_data["test_time"]=test_time
    new_df = pd.DataFrame(new_data, index=[0])
    
    dir_path = output_dir
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_path = os.path.join(dir_path, "report_tabsurvey.csv")
    
    if os.path.exists(file_path):
        # Read data from exist file
        df = pd.read_csv(file_path)
        # Add new data
        df = pd.concat([df, new_df], ignore_index=True)
        # Save report
        df.to_csv(file_path, index=False)
    else:
        print("---Create new report!----")
        # Create new report
        df = new_df
        df.to_csv(file_path, index=False)

def save_hyperparameters_to_file(args, params, results, time=None):
    filename = get_output_path(args, filename="hp_log", file_type="txt")

    with open(filename, "a") as text_file:
        text_file.write(str(datetime.datetime.now()) + "\n")
        text_file.write("Parameters: %s\n\n" % params)

        for key, value in results.items():
            text_file.write("%s: %.5f\n" % (key, value))

        if time:
            text_file.write("\nTrain time: %f\n" % time["train_time"])
            text_file.write("Test time: %f\n" % time["test_time"])

        text_file.write("\n---------------------------------------\n")


def get_output_path(args, filename, file_type, directory=None, extension=None):
    # For example: output/LinearModel/Covertype
    dir_path = output_dir + args.model_name + "/" + args.dataset

    if directory:
        # For example: .../models
        dir_path = dir_path + "/" + directory

    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    file_path = dir_path + "/" + filename

    if extension is not None:
        file_path += "_" + str(extension)

    file_path += "." + file_type

    # For example: .../m_3.pkl

    return file_path


def get_predictions_from_file(args):
    dir_path = output_dir + args.model_name + "/" + args.dataset + "/predictions"

    files = os.listdir(dir_path)
    content = []

    for file in files:
        content.append(np.load(dir_path + "/" + file))

    return content
