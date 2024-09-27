from sklearn.metrics import  accuracy_score, f1_score, auc, roc_curve, recall_score, precision_score, matthews_corrcoef
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
def cfs_matrix(y_true, y_pred, labels=None):

  cfs_mt = np.full((len(labels), len(labels)), 0)
  for x,y in zip(y_true, y_pred):
    cfs_mt[x,y] += 1
  return cfs_mt

def calc_index(y_true, y_pred, y_probabilities):
    labels = np.unique(np.concatenate((y_true, y_pred)))

  #     P
  #     0   1
  # T 0 TN FP
  #   1 FN TP

    cnf_matrix = cfs_matrix(y_true, y_pred, labels)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    
    if len(labels) == 2:
        TN = cnf_matrix[0,0]
        FP = cnf_matrix[0,1]
        FN = cnf_matrix[1,0]
        TP = cnf_matrix[1,1]
        
        MCC = (((TP * TN) - (FP * FN)) / np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))) *100.
        ACC = (TP+TN)/((TP+FP+FN+TN)) *100.
        PPV = (TP)/((TP+FP)) *100.
        TPR = (TP)/((TP+FN)) *100.
        F1  = 2 * (PPV * TPR) / (PPV + TPR)
        FPR = (FP)/((FP+TN)) *100.
          
        mcc_func = matthews_corrcoef(y_true, y_pred)*100.
        acc_func = accuracy_score(y_true, y_pred)*100.
        tpr_func = recall_score(y_true,y_pred,average='weighted', zero_division = 0) *100.
        ppv_func = precision_score(y_true,y_pred,average='weighted', zero_division = 0) *100.
        f1_func  = f1_score(y_true,y_pred,average='weighted', zero_division = 0) *100.
        # auc_func = roc_auc_score(y_true, y_probabilities, multi_class='ovo', average="macro")*100.
        
        return MCC, ACC, PPV, TPR, F1, FPR, mcc_func, acc_func, tpr_func, ppv_func, f1_func

    MCC_list = []
    ACC_list = []
    PPV_list = []
    TPR_list = []
    F1_list = []
    FPR_list = []
    for i, label in enumerate(labels):
        tp = TP[i]
        tn = TN[i]
        fp = FP[i]
        fn = FN[i]
        # Matthew’s correlation coefficient
        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = numerator / denominator * 100. if denominator !=0 else 0
        # Overall accuracy
        acc = (tp + tn) / (tp + fp + fn + tn) * 100.
        # Precision or positive predictive value
        ppv = (tp) / (tp + fp) * 100. if (tp + fp) != 0 else 0
        # Sensitivity, hit rate, recall, or true positive rate
        tpr = (tp) / (tp + fn) * 100. if (tp + fn) != 0 else 0
        # F1 Score
        f1  = 2 * (ppv * tpr) / (ppv + tpr) if (ppv + tpr) != 0 else 0
        # Fall out or false positive rate
        fpr = (fp) / (fp + tn) * 100. if (fp + tn) != 0 else 0
        
        MCC_list.append(mcc)
        ACC_list.append(acc)
        PPV_list.append(ppv)
        TPR_list.append(tpr)
        F1_list.append(f1)
        FPR_list.append(fpr)
        
    MCC = np.mean(MCC_list)
    ACC = np.mean(ACC_list)
    PPV = np.mean(PPV_list)
    TPR = np.mean(TPR_list)
    F1 = np.mean(F1_list)
    FPR = np.mean(FPR_list)
    mcc_func = matthews_corrcoef(y_true, y_pred)*100.
    acc_func = accuracy_score(y_true, y_pred)*100.
    tpr_func = recall_score(y_true,y_pred,average='weighted', zero_division = 0) *100.
    ppv_func = precision_score(y_true,y_pred,average='weighted', zero_division = 0) *100.
    f1_func  = f1_score(y_true,y_pred,average='weighted', zero_division = 0) *100.
    return cnf_matrix, MCC, ACC, PPV, TPR, F1, FPR, mcc_func, acc_func, tpr_func, ppv_func, f1_func
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
        
    def get_confusion_matrix(self):
        raise NotImplementedError("Has be implemented in the sub class")
# class RegScorer(Scorer):

#     def __init__(self):
#         self.mses = []
#         self.r2s = []

#     # y_probabilities is None for Regression
#     def eval(self, y_true, y_prediction, y_probabilities):
#         mse = mean_squared_error(y_true, y_prediction)
#         r2 = r2_score(y_true, y_prediction)

#         self.mses.append(mse)
#         self.r2s.append(r2)

#         return {"MSE": mse, "R2": r2}

#     def get_results(self):
#         mse_mean = np.mean(self.mses)
#         mse_std = np.std(self.mses)

#         r2_mean = np.mean(self.r2s)
#         r2_std = np.std(self.r2s)

#         return {"MSE - mean": mse_mean,
#                 "MSE - std": mse_std,
#                 "R2 - mean": r2_mean,
#                 "R2 - std": r2_std}

#     def get_objective_result(self):
#         return np.mean(self.mses)


class ClassScorer(Scorer):

    def __init__(self):
        self.cnf_matrixs=[]
        self.MCCs = []
        self.ACCs = []
        self.PPVs = []
        self.TPRs = []
        self.F1s = []
        self.FPRs = []
        self.mcc_funcs = []
        self.acc_funcs = []
        self.tpr_funcs = []
        self.ppv_funcs = []
        self.f1_funcs = []
    def eval(self, y_true, y_prediction, y_probabilities):
        cnf_matrix, MCC, ACC, PPV, TPR, F1, FPR, mcc_func, acc_func, tpr_func, ppv_func, f1_func = calc_index(y_true, y_prediction, y_probabilities)

        self.cnf_matrixs = cnf_matrix
        self.MCCs.append(MCC)
        self.ACCs.append(ACC)
        self.PPVs.append(PPV)
        self.TPRs.append(TPR)
        self.F1s.append(F1)
        self.FPRs.append(FPR)
        self.mcc_funcs.append(mcc_func)
        self.acc_funcs.append(acc_func)
        self.tpr_funcs.append(tpr_func)
        self.ppv_funcs.append(ppv_func)
        self.f1_funcs.append(f1_func)


        return {'MCC': MCC,
                'ACC': ACC,
                'PPV': PPV,
                'TPR': TPR,
                'F1': F1,
                'FPR': FPR,
                'mcc_func': mcc_func,
                'acc_func': acc_func,
                'tpr_func': tpr_func,
                'ppv_func': ppv_func,
                'f1_func': f1_func}

    def get_results(self):
        return {'MCC - mean': np.mean(self.MCCs),
                'ACC - mean': np.mean(self.ACCs),
                'PPV - mean': np.mean(self.PPVs),
                'TPR - mean': np.mean(self.TPRs),
                'F1 - mean': np.mean(self.F1s),
                'FPR - mean': np.mean(self.FPRs),
                'mcc_func - mean': np.mean(self.mcc_funcs),
                'acc_func - mean': np.mean(self.acc_funcs),
                'tpr_func - mean': np.mean(self.tpr_funcs),
                'ppv_func - mean': np.mean(self.ppv_funcs),
                'f1_funcs - mean': np.mean(self.f1_funcs)}

    def get_objective_result(self):
        return np.mean(self.ACCs)

    def get_confusion_matrix(self):
        return self.cnf_matrixs
        
class BinScorer(Scorer):

    def __init__(self):
        self.cnf_matrixs=[]
        self.MCCs = []
        self.ACCs = []
        self.PPVs = []
        self.TPRs = []
        self.F1s = []
        self.FPRs = []
        self.mcc_funcs = []
        self.acc_funcs = []
        self.tpr_funcs = []
        self.ppv_funcs = []
        self.f1_funcs = []

    def eval(self, y_true, y_prediction, y_probabilities):
        cnf_matrix, MCC, ACC, PPV, TPR, F1, FPR, mcc_func, acc_func, tpr_func, ppv_func, f1_func = calc_index(y_true, y_prediction, y_probabilities)

        self.cnf_matrixs = cnf_matrix
        self.MCCs.append(MCC)
        self.ACCs.append(ACC)
        self.PPVs.append(PPV)
        self.TPRs.append(TPR)
        self.F1s.append(F1)
        self.FPRs.append(FPR)
        self.mcc_funcs.append(mcc_func)
        self.acc_funcs.append(acc_func)
        self.tpr_funcs.append(tpr_func)
        self.ppv_funcs.append(ppv_func)
        self.f1_funcs.append(f1_func)

        return {'MCC': MCC,
                'ACC': ACC,
                'PPV': PPV,
                'TPR': TPR,
                'F1': F1,
                'FPR': FPR,
                'mcc_func': mcc_func,
                'acc_func': acc_func,
                'tpr_func': tpr_func,
                'ppv_func': ppv_func,
                'f1_func': f1_func}

    def get_results(self):
        return {'MCC - mean': np.mean(self.MCCs),
                'ACC - mean': np.mean(self.ACCs),
                'PPV - mean': np.mean(self.PPVs),
                'TPR - mean': np.mean(self.TPRs),
                'F1 - mean': np.mean(self.F1s),
                'FPR - mean': np.mean(self.FPRs),
                'mcc_func - mean': np.mean(self.mcc_funcs),
                'acc_func - mean': np.mean(self.acc_funcs),
                'tpr_func - mean': np.mean(self.tpr_funcs),
                'ppv_func - mean': np.mean(self.ppv_funcs),
                'f1_funcs - mean': np.mean(self.f1_funcs)}

    def get_objective_result(self):
        return np.mean(self.ACCs)
        
    def get_confusion_matrix(self):
        return self.cnf_matrixs