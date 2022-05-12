from datetime import datetime
import numpy as np
from ml.model import BaseModel
import utils
import pandas as pd
import file_utils
import os
from sklearn.metrics import roc_auc_score, f1_score


def train_model_prod(df, model: BaseModel, filename: str):
    model.initialize()
    print(model.model.summary())
    ts, x, y = ts_train, x_train, y_train = model.ig.get_io(df, True)
    print("Train start: {}".format(datetime.utcfromtimestamp(ts[0]).strftime('%Y-%m-%d %H:%M:%S')))
    print("Train end: {}".format(datetime.utcfromtimestamp(ts[-1]).strftime('%Y-%m-%d %H:%M:%S')))
    model.train(x, y)
    model.save(filename)


def train_test_model_multiclass(df, models, interval_s, intervals, n, tshs, results_path=None):
    for model in models:
        interval = int(interval_s / model.ig.granularity_seconds())
        for i in range(intervals + 1, 1, -1):

            df_train = df.iloc[:interval * -i]
            df_val = df.iloc[:interval * - (i - 1)]

            ts_train, x_train, y_train = model.ig.get_io(df_train, True)

            ts_val, x_val, y_val = model.ig.get_io(df_val, True)
            ts_val, x_val, y_val = ts_val[-interval:], x_val[-interval:], y_val[-interval:]

            ts_test, x_test, y_test = model.ig.get_io(df.iloc[int(-interval * 2):], True)
            ts_test, x_test, y_test = ts_test[-interval:], x_test[-interval:], y_test[-interval:]

            print("Train start: {}".format(datetime.utcfromtimestamp(ts_train[0]).strftime('%Y-%m-%d %H:%M:%S')))
            print("Train end: {}".format(datetime.utcfromtimestamp(ts_train[-1]).strftime('%Y-%m-%d %H:%M:%S')))
            print("Validation start: {}".format(datetime.utcfromtimestamp(ts_val[0]).strftime('%Y-%m-%d %H:%M:%S')))
            print("Validation end: {}".format(datetime.utcfromtimestamp(ts_val[-1]).strftime('%Y-%m-%d %H:%M:%S')))
            print("Test start: {}".format(datetime.utcfromtimestamp(ts_test[0]).strftime('%Y-%m-%d %H:%M:%S')))
            print("Test end: {}".format(datetime.utcfromtimestamp(ts_test[-1]).strftime('%Y-%m-%d %H:%M:%S')))
            for x in range(n, 0, -1):
                model.initialize()
                print(model.model.summary())

                model.train(x_train, y_train)
                val_true = model.ig.scalerY.inverse_transform(y_val)
                val_pred = model._predict(x_val)
                test_true = model.ig.scalerY.inverse_transform(y_test)
                test_pred = model._predict(x_test)

                for tsh in tshs:
                    tp_h_v, tp_l_v, fp_h_v, fp_l_v, p_v = evaluate_multiclass(val_true, val_pred, tsh)
                    tp_h_t, tp_l_t, fp_h_t, fp_l_t, p_t = evaluate_multiclass(test_true, test_pred, tsh)

                    if results_path is not None:
                        model_name = utils.model_to_str(model)
                        new_results = pd.DataFrame([[model_name, i, x, tsh,
                                                     tp_h_v, tp_l_v, fp_h_v, fp_l_v, p_v,
                                                     tp_h_t, tp_l_t, fp_h_t, fp_l_t, p_t]], columns=[
                            'model', 'month', 'n', 'tsh',
                            'tp_h_v', 'tp_l_v', 'fp_h_v', 'fp_l_v', 'p_v',
                            'tp_h_t', 'tp_l_t', 'fp_h_t', 'fp_l_t', 'p_t'])
                        utils.pretty_print(new_results)
                        if os.path.isfile(results_path):
                            results = file_utils.load_from_file(results_path)
                            new_results = results.append(new_results)
                        file_utils.dump_to_file(new_results, results_path)


def train_test_model_classification(df, models, interval_s, intervals, n, target_fpr, results_path=None):
    for model in models:
        interval = int(interval_s / model.ig.granularity_seconds())
        for i in range(intervals + 1, 1, -1):

            df_train = df.iloc[:interval * -i]
            df_val = df.iloc[interval * - (i + 2):interval * - (i - 1)]

            ts_train, x_train, y_train = model.ig.get_io(df_train, True)

            ts_val, x_val, y_val = model.ig.get_io(df_val, True)
            ts_val, x_val, y_val = ts_val[-interval:], x_val[-interval:], y_val[-interval:]

            ts_test, x_test, y_test = model.ig.get_io(df.iloc[int(-interval * 2):], True)
            ts_test, x_test, y_test = ts_test[-interval:], x_test[-interval:], y_test[-interval:]

            print("Train start: {}".format(datetime.utcfromtimestamp(ts_train[0]).strftime('%Y-%m-%d %H:%M:%S')))
            print("Train end: {}".format(datetime.utcfromtimestamp(ts_train[-1]).strftime('%Y-%m-%d %H:%M:%S')))
            print("Validation start: {}".format(datetime.utcfromtimestamp(ts_val[0]).strftime('%Y-%m-%d %H:%M:%S')))
            print("Validation end: {}".format(datetime.utcfromtimestamp(ts_val[-1]).strftime('%Y-%m-%d %H:%M:%S')))
            print("Test start: {}".format(datetime.utcfromtimestamp(ts_test[0]).strftime('%Y-%m-%d %H:%M:%S')))
            print("Test end: {}".format(datetime.utcfromtimestamp(ts_test[-1]).strftime('%Y-%m-%d %H:%M:%S')))
            for x in range(n, 0, -1):
                model.initialize()
                print(model.model.summary())

                model.train(x_train, y_train)
                val_true = model.ig.scalerY.inverse_transform(y_val)
                val_pred = model._predict(x_val)
                test_true = model.ig.scalerY.inverse_transform(y_test)
                test_pred = model._predict(x_test)

                acc_v, TPR_v, FPR_v, AUC_v, F1_v, TP_v, FP_v, TN_v, FN_v = evaluate_classification(val_true, val_pred, model.roc, target_fpr)
                AUC_T_v = roc_auc_score(val_true, val_pred)

                acc_t, TPR_t, FPR_t, AUC_t, F1_t, TP_t, FP_t, TN_t, FN_t = evaluate_classification(test_true, test_pred, model.roc, target_fpr)
                AUC_T_t = roc_auc_score(test_true, test_pred)

                if results_path is not None:
                    model_name = utils.model_to_str(model)
                    new_results = pd.DataFrame([[model_name, i, x,
                                                 acc_v, TPR_v, FPR_v, AUC_v, AUC_T_v, F1_v, TP_v, FP_v, TP_v - FP_v, TN_v, FN_v,
                                                 acc_t, TPR_t, FPR_t, AUC_t, AUC_T_t, F1_t, TP_t, FP_t, TP_t - FP_t, TN_t, FN_t]], columns=[
                        'model', 'month', 'n',
                        'acc_v', 'TPR_v', 'FPR_v', 'AUC_v', 'AUC_T_v', 'F1_v', 'TP_v', 'FP_v', 'P_v', 'TN_v', 'FN_v',
                        'acc_t', 'TPR_t', 'FPR_t', 'AUC_t', 'AUC_T_t', 'F1_t', 'TP_t', 'FP_t', 'P_t', 'TN_T', 'FN_t'])
                    utils.pretty_print(new_results)
                    if os.path.isfile(results_path):
                        results = file_utils.load_from_file(results_path)
                        new_results = results.append(new_results)
                    file_utils.dump_to_file(new_results, results_path)
        del model


def train_test_model(df, models, interval_s, intervals, n, results_path=None):
    for model in models:
        interval = int(interval_s / model.ig.granularity_seconds())
        mapes_val = []
        mapes_test = []
        for i in range(intervals + 1, 1, -1):
            mapes_interval_val = []
            mapes_interval_test = []
            df_train = df.iloc[:interval * -i]
            df_val = df.iloc[interval * - (i + 2):interval * - (i - 1)]

            ts_train, x_train, y_train = model.ig.get_io(df_train, True)

            ts_val, x_val, y_val = model.ig.get_io(df_val, True)
            ts_val, x_val, y_val = ts_val[-interval:], x_val[-interval:], y_val[-interval:]

            ts_test, x_test, y_test = model.ig.get_io(df.iloc[int(-interval * 2):], True)
            ts_test, x_test, y_test = ts_test[-interval:], x_test[-interval:], y_test[-interval:]

            print("Train start: {}".format(datetime.utcfromtimestamp(ts_train[0]).strftime('%Y-%m-%d %H:%M:%S')))
            print("Train end: {}".format(datetime.utcfromtimestamp(ts_train[-1]).strftime('%Y-%m-%d %H:%M:%S')))
            print("Validation start: {}".format(datetime.utcfromtimestamp(ts_val[0]).strftime('%Y-%m-%d %H:%M:%S')))
            print("Validation end: {}".format(datetime.utcfromtimestamp(ts_val[-1]).strftime('%Y-%m-%d %H:%M:%S')))
            print("Test start: {}".format(datetime.utcfromtimestamp(ts_test[0]).strftime('%Y-%m-%d %H:%M:%S')))
            print("Test end: {}".format(datetime.utcfromtimestamp(ts_test[-1]).strftime('%Y-%m-%d %H:%M:%S')))
            for _ in range(0, n):
                model.initialize()
                print(model.model.summary())

                model.train(x_train, y_train)
                val_true = model.ig.scalerY.inverse_transform(np.expand_dims(y_val, 1))
                val_pred = model._predict(x_val)
                test_true = model.ig.scalerY.inverse_transform(np.expand_dims(y_test, 1))
                test_pred = model._predict(x_test)

                mapes_interval_val.append(utils.mape(val_true, val_pred))
                mapes_interval_test.append(utils.mape(test_true, test_pred))
                print(mapes_interval_val)
                print(mapes_interval_test)
            mapes_val.append(np.mean(mapes_interval_val))
            mapes_test.append(np.mean(mapes_interval_test))
        if results_path is not None:
            mean_mape_val = np.mean(mapes_val)
            mean_mape_test = np.mean(mapes_test)
            model_name = utils.model_to_str(model)
            new_results = pd.DataFrame([[model_name, mean_mape_val, mean_mape_test, mapes_val, mapes_test]], columns=[
                'model', 'mean_mape_val', 'mean_mape_test', 'mapes_val', 'mapes_test'])
            if os.path.isfile(results_path):
                results = file_utils.load_from_file(results_path)
                new_results = results.append(new_results)
            file_utils.dump_to_file(new_results, results_path)
            utils.pretty_print(new_results)


def find_cutoff(roc, fpr_target):
    r = roc[roc['fpr'] < fpr_target]

    if r.empty:
        return roc[['tpr', 'fpr', 'threshold']].values[0]
    else:
        return r[['tpr', 'fpr', 'threshold']].values[-1]


def apply_cutoff(predicted, cutoff):
    return predicted > cutoff


def evaluate_multiclass(actual, prediction, tsh):

    np_p = [[1, 0, 0] if x[0] - x[2] > tsh and x[0] - x[1] > tsh else ([0, 1, 0] if x[1] - x[2] > tsh and x[1] - x[0] > tsh else [0, 0, 1]) for x in prediction]

    tp_0 = np.logical_and([x[0] == 1 for x in np_p], [x[0] == 1 for x in actual])
    fn_0 = np.logical_and([x[0] == 0 for x in np_p], [x[0] == 1 for x in actual])
    tp_1 = np.logical_and([x[1] == 1 for x in np_p], [x[1] == 1 for x in actual])
    fn_1 = np.logical_and([x[1] == 0 for x in np_p], [x[1] == 1 for x in actual])

    fp_0 = np.logical_and([x[0] == 1 for x in np_p], [x[0] == 0 for x in actual])
    fp_1 = np.logical_and([x[1] == 1 for x in np_p], [x[1] == 0 for x in actual])

    print("TP (H): {}".format(sum(tp_0)))
    print("TP (L): {}".format(sum(tp_1)))

    print("FP (H): {}".format(sum(fp_0)))
    print("FP (L): {}".format(sum(fp_1)))

    print("FN (H): {}".format(sum(fn_0)))
    print("FN (L): {}".format(sum(fn_1)))

    print("P: {}\n".format(sum(tp_0) + sum(tp_1) - sum(fp_0) - sum(fp_1)))
    return sum(tp_0), sum(tp_1), sum(fp_0), sum(fp_1), sum(tp_0) + sum(tp_1) - sum(fp_0) - sum(fp_1)


def evaluate_classification(actual, prediction, roc, fpr_target):

    _, _, cutoff = find_cutoff(roc, fpr_target)
    pred = apply_cutoff(prediction, cutoff)
    AUC = roc_auc_score(actual, pred)
    F1 = f1_score(actual, pred)

    TP = np.sum(np.logical_and(pred == 1, actual == 1))
    TN = np.sum(np.logical_and(pred == 0, actual == 0))
    FP = np.sum(np.logical_and(pred == 1, actual == 0))
    FN = np.sum(np.logical_and(pred == 0, actual == 1))

    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)

    accuracy = (TP + TN) / (TP + TN + FP + FN)

    return accuracy, TPR, FPR, AUC, F1, int(TP), int(FP), int(TN), int(FN)
