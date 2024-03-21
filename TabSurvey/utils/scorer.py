from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.metrics import roc_curve, auc, recall_score, precision_score, matthews_corrcoef
import numpy as np


def get_scorer(args):
    if args.objective == "regression":
        return RegScorer()
    elif args.objective == "classification":
        return ClassScorer()
    elif args.objective == "binary":
        return BinScorer()
    else:
        raise NotImplementedError("No scorer for \"" + args.objective + "\" implemented")
###########################################################################################
def cfs_matrix(y_test, y_pred, len_labels):

  cfs_mt = np.full((len_labels, len_labels), 0)

  for x,y in zip(y_test, y_pred):
    cfs_mt[x,y] += 1
  return cfs_mt

def calc_index(y_test, y_pred, len_labels):

  #     P
  #     0   1
  # T 0 TN FP
  #   1 FN TP

  cnf_matrix = cfs_matrix(y_test, y_pred, len_labels)
  FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
  FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
  TP = np.diag(cnf_matrix)
  TN = cnf_matrix.sum() - (FP + FN + TP)


  if len_labels == 2:
    TN = cnf_matrix[0,0]
    FP = cnf_matrix[0,1]
    FN = cnf_matrix[1,0]
    TP = cnf_matrix[1,1]

  # Sensitivity, hit rate, recall, or true positive rate
  TPR = TP/(TP+FN) *100.
  # Specificity or true negative rate
  TNR = TN/(TN+FP) *100.
  # Precision or positive predictive value
  PPV = TP/(TP+FP) *100.
  # Negative predictive value
  NPV = TN/(TN+FN) *100.
  # Fall out or false positive rate
  FPR = FP/(FP+TN) *100.
  # False negative rate
  FNR = FN/(TP+FN) *100.
  # False discovery rate
  FDR = FP/(TP+FP) *100.
  # F1 Score
  F1 = 2 * (PPV * TPR) / (PPV + TPR)

  # Matthew’s correlation coefficient
  MCC = ((TP * TN) - (FP * FN)) / (( (TP+FP) * (TP+FN) * (TN+FP) * (TN+FN) ) ** 0.5)

  # Overall accuracy
  ACC = (TP+TN)/(TP+FP+FN+TN) *100.

  false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,  y_pred)
  auc_func = auc(false_positive_rate, true_positive_rate)


  tpr_func = recall_score(y_test,y_pred,average='weighted', zero_division = 0) *100.
  ppv_func = precision_score(y_test,y_pred,average='weighted', zero_division = 0) *100.
  f1_func = f1_score(y_test,y_pred,average='weighted', zero_division = 0) *100.
  mcc_func = matthews_corrcoef(y_test,y_pred)
  if len_labels == 2:
    print("------------------------")
    # print("ACC: {:.4f}".format(ACC))
    # print("MCC: {:.4f}".format(MCC))
    # print("TPR: {:.4f}".format(TPR))
    # print("PPV: {:.4f}".format(PPV))
    # print("FPR: {:.4f}".format(FPR))
    # print("F1 : {:.4f}".format(F1))
  else:
    ACC = sum(TP+TN)/(sum(TP+FP+FN+TN)) *100.
    MCC = (TP * TN) - (FP * FN) / (( (TP+FP) * (TP+FN) * (TN+FP) * (TN+FN) ) ** 0.5) * 100
    TPR = sum(TP)/sum((TP+FN)) *100.
    PPV = sum(TP)/sum((TP+FP)) *100.
    FPR = sum(FP)/sum((FP+TN)) *100.
    F1  = 2 * (PPV * TPR) / (PPV + TPR)
    print("------------------------")
    # print("ACC: {:.4f}".format(ACC))
    # print("MCC: {:.4f}".format(MCC))
    # print("TPR: {:.4f}".format(TPR))
    # print("PPV: {:.4f}".format(PPV))
    # print("FPR: {:.4f}".format(FPR))
    # print("F1 : {:.4f}".format(F1))


  print("TPR-weight: {:.4f}".format(tpr_func))
  print("PPV-weight: {:.4f}".format(ppv_func))
  print("F1-weight : {:.4f}".format(f1_func))
  print("MCC-func  : {:.4f}".format(mcc_func))
  print("AUC-func  : {:.4f}".format(auc_func))
  # print("CFS MATRIX:\n",cnf_matrix)
  return mcc_func, ACC, TPR, FPR, F1, tpr_func, ppv_func, f1_func, auc_func
###########################################################################################

class Scorer:

    """
        y_true: (n_samples,)
        y_prediction: (n_samples,) - predicted classes
        y_probabilities: (n_samples, n_classes) - probabilities of the classes (summing to 1)
    """
    def eval(self, y_true, y_prediction, y_probabilities):
        raise NotImplementedError("Has be implemented in the sub class")

    def get_results(self):
        raise NotImplementedError("Has be implemented in the sub class")

    def get_objective_result(self):
        raise NotImplementedError("Has be implemented in the sub class")


class RegScorer(Scorer):

    def __init__(self):
        self.mses = []
        self.r2s = []

    # y_probabilities is None for Regression
    def eval(self, y_true, y_prediction, y_probabilities):
        mse = mean_squared_error(y_true, y_prediction)
        r2 = r2_score(y_true, y_prediction)

        self.mses.append(mse)
        self.r2s.append(r2)

        return {"MSE": mse, "R2": r2}

    def get_results(self):
        mse_mean = np.mean(self.mses)
        mse_std = np.std(self.mses)

        r2_mean = np.mean(self.r2s)
        r2_std = np.std(self.r2s)

        return {"MSE - mean": mse_mean,
                "MSE - std": mse_std,
                "R2 - mean": r2_mean,
                "R2 - std": r2_std}

    def get_objective_result(self):
        return np.mean(self.mses)


class ClassScorer(Scorer):

    def __init__(self):
        self.loglosses = []
        self.aucs = []
        self.accs = []
        self.f1s = []

    def eval(self, y_true, y_prediction, y_probabilities):
        logloss = log_loss(y_true, y_probabilities)
        # auc = roc_auc_score(y_true, y_probabilities, multi_class='ovr')
        auc = roc_auc_score(y_true, y_probabilities, multi_class='ovo', average="macro")

        acc = accuracy_score(y_true, y_prediction)
        f1 = f1_score(y_true, y_prediction, average="weighted")  # use here macro or weighted?

        self.loglosses.append(logloss)
        self.aucs.append(auc)
        self.accs.append(acc)
        self.f1s.append(f1)

        return {"Log Loss": logloss, "AUC": auc, "Accuracy": acc, "F1 score": f1}

    def get_results(self):
        logloss_mean = np.mean(self.loglosses)
        logloss_std = np.std(self.loglosses)

        auc_mean = np.mean(self.aucs)
        auc_std = np.std(self.aucs)

        acc_mean = np.mean(self.accs)
        acc_std = np.std(self.accs)

        f1_mean = np.mean(self.f1s)
        f1_std = np.std(self.f1s)

        return {"Log Loss - mean": logloss_mean,
                "Log Loss - std": logloss_std,
                "AUC - mean": auc_mean,
                "AUC - std": auc_std,
                "Accuracy - mean": acc_mean,
                "Accuracy - std": acc_std,
                "F1 score - mean": f1_mean,
                "F1 score - std": f1_std}

    def get_objective_result(self):
        return np.mean(self.loglosses)


class BinScorer(Scorer):

    def __init__(self):
        # self.loglosses = []
        # self.aucs = []
        # self.accs = []
        # self.f1s = []
        self.mcc_funcs = []
        self.ACCs = []
        self.TPRs = []
        self.FPRs = []
        self.F1s = []
        self.tpr_funcs = []
        self.ppv_funcs = []
        self.f1_funcs = []
        self.auc_funcs = []

    def eval(self, y_true, y_prediction, y_probabilities):
        # logloss = log_loss(y_true, y_probabilities)
        # auc = roc_auc_score(y_true, y_probabilities[:, 1])

        # acc = accuracy_score(y_true, y_prediction)
        # f1 = f1_score(y_true, y_prediction, average="micro")  # use here macro or weighted?

        # self.loglosses.append(logloss)
        # self.aucs.append(auc)
        # self.accs.append(acc)
        # self.f1s.append(f1)

        # return {"Log Loss": logloss, "AUC": auc, "Accuracy": acc, "F1 score": f1}
        mcc_func, ACC, TPR, FPR, F1, tpr_func, ppv_func, f1_func, auc_func = calc_index(y_true, y_prediction, len_labels=2)


        self.mcc_funcs.append(mcc_func)
        self.ACCs.append(ACC)
        self.TPRs.append(TPR)
        self.FPRs.append(FPR)
        self.F1s.append(F1)
        self.tpr_funcs.append(tpr_func)
        self.ppv_funcs.append(ppv_func)
        self.f1_funcs.append(f1_func)
        self.auc_funcs.append(auc_func)


        return {'MCC': mcc_func,
                'ACC': ACC,
                'TPR': TPR,
                'FPR': FPR,
                'F1': F1,
                'TPR weight': tpr_func,
                'PPV weight': ppv_func,
                'F1 weight': f1_func,
                'AUC_f': auc_func}

    def get_results(self):
        # logloss_mean = np.mean(self.loglosses)
        # logloss_std = np.std(self.loglosses)

        # auc_mean = np.mean(self.aucs)
        # auc_std = np.std(self.aucs)

        # acc_mean = np.mean(self.accs)
        # acc_std = np.std(self.accs)

        # f1_mean = np.mean(self.f1s)
        # f1_std = np.std(self.f1s)

        # return {"Log Loss - mean": logloss_mean,
        #         "Log Loss - std": logloss_std,
        #         "AUC - mean": auc_mean,
        #         "AUC - std": auc_std,
        #         "Accuracy - mean": acc_mean,
        #         "Accuracy - std": acc_std,
        #         "F1 score - mean": f1_mean,
        #         "F1 score - std": f1_std}
        return {'MCC - mean': np.mean(self.mcc_funcs),
                'ACC - mean': np.mean(self.ACCs),
                'TPR - mean': np.mean(self.TPRs),
                'FPR - mean': np.mean(self.FPRs),
                'F1 - mean': np.mean(self.F1s),
                'TPR weight - mean': np.mean(self.tpr_funcs),
                'PPV weight - mean': np.mean(self.ppv_funcs),
                'F1 weight - mean': np.mean(self.f1_funcs),
                'AUC_f - mean': np.mean(self.auc_funcs)}

    def get_objective_result(self):
        return np.mean(self.auc_funcs)
