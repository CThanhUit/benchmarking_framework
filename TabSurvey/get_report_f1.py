import logging
import sys

import optuna

from models import str2model
from utils.load_data import load_data
from utils.scorer import get_scorer
from utils.timer import Timer
from utils.io_utils import save_results_to_file, save_hyperparameters_to_file, save_loss_to_file, save_reports_to_csv_file, save_plot_confusion_matrix
from utils.parser import get_parser, get_given_parameters_parser

from sklearn.model_selection import KFold, StratifiedKFold  # , train_test_split

# save report
import pandas as pd
import numpy as np
import os

def cross_validation(model, X, y, args, save_model=False):
    # Record some statistics and metrics
    sc = get_scorer(args)
    train_timer = Timer()
    test_timer = Timer()

    if args.objective == "regression":
        kf = KFold(n_splits=args.num_splits, shuffle=args.shuffle, random_state=args.seed)
    elif args.objective == "classification" or args.objective == "binary":
        kf = StratifiedKFold(n_splits=args.num_splits, shuffle=args.shuffle, random_state=args.seed)
    else:
        raise NotImplementedError("Objective" + args.objective + "is not yet implemented.")

    for i, (train_index, test_index) in enumerate(kf.split(X, y)):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=args.seed)

        # Create a new unfitted version of the model
        curr_model = model.clone()

        # Train model
        train_timer.start()
        loss_history, val_loss_history = curr_model.fit(X_train, y_train, X_test, y_test)  # X_val, y_val)
        train_timer.end()

        # Test model
        test_timer.start()
        curr_model.predict(X_test)
        test_timer.end()
        
        # Save model weights and the truth/prediction pairs for traceability
        curr_model.save_model_and_predictions(y_test, i)
        
        # Compute scores on the output
        print(sc.eval(y_test, curr_model.predictions, curr_model.prediction_probabilities))
        
        if save_model:
            save_loss_to_file(args, loss_history, "loss", extension=i)
            save_loss_to_file(args, val_loss_history, "val_loss", extension=i)
            save_plot_confusion_matrix(args ,sc.get_confusion_matrix(), args.label_classes, extension=i, figsize=(len(np.unique(y_test)), len(np.unique(y_test))*0.8))


    # Best run is saved to file
    if save_model:
        print("Results:", sc.get_results())
        print("Train time:", train_timer.get_average_time())
        print("Inference time:", test_timer.get_average_time())

        # Save the all statistics to a file
        save_results_to_file(args, sc.get_results(),
                             train_timer.get_average_time(), test_timer.get_average_time(),
                             model.params)
        save_reports_to_csv_file(args, sc.get_results(),
                             train_timer.get_average_time(), test_timer.get_average_time())
    # print("Finished cross validation")
    return sc, {'train_time': train_timer.get_average_time(), 'test_time': test_timer.get_average_time()}


class Objective(object):
    def __init__(self, args, model_name, X, y):
        # Save the model that will be trained
        self.model_name = model_name

        # Save the trainings data
        self.X = X
        self.y = y

        self.args = args

    def __call__(self, trial):
        # Define hyperparameters to optimize
        trial_params = self.model_name.define_trial_parameters(trial, self.args)

        # Create model
        model = self.model_name(trial_params, self.args)

        # Cross validate the chosen hyperparameters
        sc, time = cross_validation(model, self.X, self.y, self.args)

        save_hyperparameters_to_file(self.args, trial_params, sc.get_results(), time)

        return sc.get_objective_result()


def main(args):
    print("Start hyperparameter optimization")
    X, y = load_data(args)

    model_name = str2model(args.model_name)

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = args.model_name + "_" + args.dataset
    storage_name = "sqlite:///{}.db".format(study_name)

    study = optuna.create_study(direction=args.direction,
                                study_name=study_name,
                                storage=storage_name,
                                load_if_exists=True)
    study.optimize(Objective(args, model_name, X, y), n_trials=args.n_trials)
    print("Best parameters:", study.best_trial.params)

    # Run best trial again and save it!
    model = model_name(study.best_trial.params, args)
    sc, timer=cross_validation(model, X, y, args, save_model=True)
    # /////////////////////////////////////////////////////////
    result = sc.get_results()
    print("Results:", result)
    print("Avegare Train time:", timer['train_time'])
    print("Avegare Test time:", timer['test_time'])
    # ////////////////////////////////////////////////////////

def main_once(args):
    print("Train model with given hyperparameters")
    X, y = load_data(args)

    model_name = str2model(args.model_name)

    parameters = args.parameters[args.dataset][args.model_name]
    model = model_name(parameters, args)

    sc, timer = cross_validation(model, X, y, args)
    # print(sc.get_results())
    # /////////////////////////////////////////////////////////
    result = sc.get_results()
    print("Results:", result)
    print("Avegare Train time:", timer['train_time'])
    print("Avegare Test time:", timer['test_time'])
    # ////////////////////////////////////////////////////////


if __name__ == "__main__":
    parser = get_parser()
    arguments = parser.parse_args()
    # print(arguments)

    if arguments.optimize_hyperparameters:
        main(arguments)
    else:
        # Also load the best parameters
        parser = get_given_parameters_parser()
        arguments = parser.parse_args()
        main_once(arguments)
