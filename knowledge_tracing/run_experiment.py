import os
from datetime import datetime
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

from knowledge_tracing import path_algos


DATA_DIR = os.path.join(path_algos, "data/RoboMission")


def load(name):
    """ Helper function that load RoboMission files by name """
    path = os.path.join(DATA_DIR, f"{name}.csv")
    df = pd.read_csv(path, index_col="id")
    return df


def encode_afm_bg(task_sessions, qmatrix):
    """ Transforms task_sessions and qmatrix into features X, and labels y
    that allow training an AFM-BG model as a logistic regression.

    Parameters
    -----------
    task_sessions: pd.DataFrame with columns [student, start, task, solved]
        contains info about student & learning-plateform interactions
    qmatrix: pd.DataFrame
        links every task to corresponding knowledge components

    Returns
    -------
    X: features
        DataFrame of shape (n_samples, n_features)
    y: labels
        DataFrame of shape (n_samples, )

    """

    # order by student & start_date to allow easily counting of kc-exposure
    task_sessions_ord = task_sessions.sort_values(by=["student", "start"])

    # initialize features
    _, nb_kc = qmatrix.shape
    X = np.zeros((len(task_sessions_ord), 2 * nb_kc))

    # fill features matrix
    student = None
    for i, (index, row) in enumerate(task_sessions_ord.iterrows()):
        if row["student"] != student:
            kc_cnt = np.zeros(nb_kc)
            student = row["student"]
        q = qmatrix.loc[row["task"]].values
        X[i, :] = np.r_[q, q * kc_cnt]
        kc_cnt += qmatrix.loc[row["task"]].values
    X = pd.DataFrame(X, index=task_sessions_ord.index).reindex(task_sessions.index)

    # initialize labels
    y = task_sessions["solved"]

    return X, y


def encode_afm_bgt(task_sessions, qmatrix):
    """ Transforms task_sessions and qmatrix into features X, and labels y
    that allow training an AFM-BGT model as a logistic regression.

    Parameters
    -----------
    task_sessions: pd.DataFrame with columns [student, start, task, solved]
        contains info about student & learning-plateform interactions
    qmatrix: pd.DataFrame
        links every task to corresponding knowledge components

    Returns
    -------
    X: features
        DataFrame of shape (n_samples, n_features)
    y: labels
        DataFrame of shape (n_samples, )

    """

    # order by student & start_date to allow easily counting of kc-exposure
    task_sessions_ord = task_sessions.sort_values(by=["student", "start"])

    # initialize features
    nb_items, nb_kc = qmatrix.shape
    X = np.zeros((len(task_sessions_ord), 2 * nb_kc + nb_items))
    item_to_idx = {item: i for i, item in enumerate(qmatrix.index)}
    Id = np.eye(nb_items)

    # fill features matrix
    student = None
    for i, (index, row) in enumerate(task_sessions_ord.iterrows()):
        if row["student"] != student:
            kc_cnt = np.zeros(nb_kc)
            student = row["student"]
        q = qmatrix.loc[row["task"]].values
        one_hot_item = Id[item_to_idx[row["task"]], :]
        X[i, :] = np.r_[q, q * kc_cnt, one_hot_item]
        kc_cnt += qmatrix.loc[row["task"]].values
    X = pd.DataFrame(X, index=task_sessions_ord.index).reindex(task_sessions.index)

    # initialize labels
    y = task_sessions["solved"]

    return X, y


def encode_pfa(task_sessions, qmatrix):
    """ """
    """ Transforms task_sessions and qmatrix into features X, and labels y
    that allow training an PFA model as a logistic regression.

    Parameters
    -----------
    task_sessions: pd.DataFrame with columns [student, start, task, solved]
        contains info about student & learning-plateform interactions
    qmatrix: pd.DataFrame
        links every task to corresponding knowledge components

    Returns
    -------
    X: features
        DataFrame of shape (n_samples, n_features)
    y: labels
        DataFrame of shape (n_samples, )

    """

    # order by student & start_date to allow easily counting of kc-exposure
    task_sessions_ord = task_sessions.sort_values(by=["student", "start"])

    # initialize features
    nb_items, nb_kc = qmatrix.shape
    X = np.zeros((len(task_sessions_ord), 3 * nb_kc))

    # fill features matrix
    student = None
    for i, (index, row) in enumerate(task_sessions_ord.iterrows()):
        if row["student"] != student:
            kc_cnt_w = np.zeros(nb_kc)
            kc_cnt_f = np.zeros(nb_kc)
            student = row["student"]
        q = qmatrix.loc[row["task"]].values
        X[i, :] = np.r_[q, q * kc_cnt_w, q * kc_cnt_f]
        if row["solved"] > 0:
            kc_cnt_w += qmatrix.loc[row["task"]].values
        else:
            kc_cnt_f += qmatrix.loc[row["task"]].values
    X = pd.DataFrame(X, index=task_sessions_ord.index).reindex(task_sessions.index)

    # initialize labels
    y = task_sessions["solved"]

    return X, y


def item_avg_predictor(task_sessions):
    """ Returns a item average predictor

    Parameters
    -----------
    task_sessions: pd.DataFrame with columns [task, solved]
        contains info about student & learning-plateform interactions

    Returns
    -------
    item_avg_: function
        for a given task, returns the resolution rate

    """

    df_item_solved_avg = task_sessions[["task", "solved"]].groupby("task").mean()
    global_avg = df_item_solved_avg["solved"].mean()

    def item_avg_(task):
        """ Returns how well, on average, a given task was solved """
        if task in df_item_solved_avg.index:
            return df_item_solved_avg.loc[task].values[0]
        else:
            return global_avg

    return item_avg_


def compute_metrics(y_test, y_test_pred_probas):
    y_test_pred = np.round(y_test_pred_probas)
    return {
        "accuracy": accuracy_score(y_test, y_test_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_test_pred),
        "auc": roc_auc_score(y_test, y_test_pred_probas),
        "f1_score": f1_score(y_test, y_test_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test.astype(int), y_test_pred_probas)),
    }


def scale_features(X_train, X_test, method=None):
    """ Scales features according to method """
    if method is None:
        return X_train, X_test
    elif method == "MaxAbsScaler":
        scaler = MaxAbsScaler()
        scaler.fit(X_train)
        X_train_ = scaler.transform(X_train)
        X_test_ = scaler.transform(X_test)
        return X_train_, X_test_
    elif method == "StandardScaler":
        scaler = StandardScaler(with_mean=False)
        scaler.fit(X_train)
        X_train_ = scaler.transform(X_train)
        X_test_ = scaler.transform(X_test)
        return X_train_, X_test_
    else:
        raise ValueError(f"Unknown method {method}")


def compute_samples_weight(task_sessions, method=None):
    """ Computes samples weight to remove dataset bias """
    y = task_sessions["solved"]
    if method is None:
        return [1] * len(task_sessions)
    elif method == "per_answer":
        c = Counter(y)
        return [1 / c[y_] for y_ in y]
    elif method == "per_item_and_answer":
        c = task_sessions.groupby(["task", "solved"]).size()
        return [1 / c[task][y_] for y_, task in zip(y, task_sessions["task"])]
    else:
        raise ValueError(f"Unknown method {method}")


def change_qmatrix(qmatrix, method=None):
    """ Modify qmatrix
    If method is add_random_kc, it adds a random knowledge component
    """
    if method is None or method == "Q1":
        return qmatrix
    elif method == "Q2":
        return qmatrix[
            [
                "teleport",
                "collect",
                "obstacle",
                "destruct",
                "sequences",
                "while",
                "repeat",
                "nested-loops",
                "if",
                "else",
            ]
        ]
    elif method == "Q3":
        return qmatrix[["sequences", "while", "repeat", "nested-loops", "if", "else"]]
    elif method == "Q4":
        return qmatrix[["sequences", "loop", "nested-loops", "test"]]
    elif method == "add_random_kc":
        nb_items, nb_kc = qmatrix.shape
        qmatrix["random"] = np.random.binomial(size=nb_items, n=1, p=0.2)
        return qmatrix
    else:
        raise ValueError(f"Unknown method {method}")


def average_metrics(list_of_dict_metrics):
    """ Average metrics"""
    return {
        "accuracy": np.mean([d["accuracy"] for d in list_of_dict_metrics]),
        "balanced_accuracy": np.mean(
            [d["balanced_accuracy"] for d in list_of_dict_metrics]
        ),
        "auc": np.mean([d["auc"] for d in list_of_dict_metrics]),
        "f1_score": np.mean([d["f1_score"] for d in list_of_dict_metrics]),
        "RMSE": np.mean([d["RMSE"] for d in list_of_dict_metrics]),
    }


if __name__ == "__main__":

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold
    from sklearn.metrics import (
        accuracy_score,
        roc_auc_score,
        f1_score,
        mean_squared_error,
        balanced_accuracy_score,
    )
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--qmatrix_method", type=str, default=None)
    parser.add_argument("-s", "--scaling_method", type=str, default="MaxAbsScaler")
    parser.add_argument("-n", "--n_splits", type=int, default=5)
    parser.add_argument("-w", "--sample_weight_method", type=str, default=None)
    args = parser.parse_args()

    params = {
        "change_qmatrix_method": args.qmatrix_method,
        "compute_samples_weight_method": args.sample_weight_method,
        "scale_features_method": args.scaling_method,
        "n_splits": args.n_splits,
    }

    # loading q matrix
    qmatrix = load("qmatrix")

    # load task_sessions
    path_to_task_sessions = os.path.join(DATA_DIR, "task_sessions.csv")
    parse_dates = ["start", "end"]
    task_sessions = pd.read_csv(
        path_to_task_sessions,
        index_col="id",
        parse_dates=parse_dates,
        infer_datetime_format=True,
    )
    task_sessions = task_sessions.sort_values(by=["student", "start"])

    # try different q-matrix
    qmatrix = change_qmatrix(qmatrix, method=params["change_qmatrix_method"])
    print(f"[INFO] {qmatrix.shape[0]} questions and {qmatrix.shape[1]} items !")

    # filter some non-representative students
    cnt_attempted_tasks = task_sessions.groupby("student").count()["solved"]
    durations = (task_sessions["end"] - task_sessions["start"]) / np.timedelta64(1, "s")
    cnt_solved_tasks = task_sessions.groupby("student").sum()["solved"]
    # filter on students
    students_filtered = cnt_solved_tasks[
        (cnt_solved_tasks > 5) & (cnt_attempted_tasks > 10) & (cnt_attempted_tasks < 50)
    ].index
    # filter task_sessions
    task_sessions = task_sessions[
        task_sessions["student"].isin(students_filtered)
        & (~task_sessions.isna()["program"])
        & (durations < 300)
    ]

    # we are going to log some informations in the process
    logs = defaultdict(dict)
    logs["params"] = params

    # Let's do some Kfolds and CV along student axis
    n_splits = params["n_splits"]
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    students = task_sessions["student"].unique()
    print(f"[INFO] {len(students)} students in the dataset !")

    for i, (idx_students_train, idx_students_test) in enumerate(kfold.split(students)):

        students_train = np.array(students)[idx_students_train]
        students_test = np.array(students)[idx_students_test]

        # logs[f"fold{i}"]["train"] = students_train
        # logs[f"fold{i}"]["test"] = students_test

        # train / test
        task_sessions_train = task_sessions[
            task_sessions["student"].isin(students_train)
        ]
        task_sessions_test = task_sessions[task_sessions["student"].isin(students_test)]

        # weight samples
        sample_weight = compute_samples_weight(
            task_sessions_train, method=params["compute_samples_weight_method"]
        )

        # AFM-BG
        X_train_afm_bg, y_train_afm_bg = encode_afm_bg(task_sessions_train, qmatrix)
        X_test_afm_bg, y_test_afm_bg = encode_afm_bg(task_sessions_test, qmatrix)
        X_train_, X_test_ = scale_features(
            X_train_afm_bg, X_test_afm_bg, method=params["scale_features_method"]
        )
        lr = LogisticRegression(max_iter=1000, solver="liblinear")
        lr.fit(X_train_, y_train_afm_bg, sample_weight=sample_weight)
        # metrics test
        y_test_pred_probas_afm_bg = lr.predict_proba(X_test_)[:, 1]
        logs[f"fold{i}"]["afm_bg"] = compute_metrics(
            y_test_afm_bg, y_test_pred_probas_afm_bg
        )
        # metrics train
        y_train_pred_probas_afm_bg = lr.predict_proba(X_train_)[:, 1]
        logs[f"fold{i}"]["afm_bg_train"] = compute_metrics(
            y_train_afm_bg, y_train_pred_probas_afm_bg
        )

        # AFM-BGT
        X_train_afm_bgt, y_train_afm_bgt = encode_afm_bgt(task_sessions_train, qmatrix)
        X_test_afm_bgt, y_test_afm_bgt = encode_afm_bgt(task_sessions_test, qmatrix)
        X_train_, X_test_ = scale_features(
            X_train_afm_bgt, X_test_afm_bgt, method=params["scale_features_method"]
        )
        lr = LogisticRegression(max_iter=1000, solver="liblinear")
        lr.fit(X_train_, y_train_afm_bgt, sample_weight=sample_weight)
        # metrics test
        y_test_pred_probas_afm_bgt = lr.predict_proba(X_test_)[:, 1]
        logs[f"fold{i}"]["afm_bgt"] = compute_metrics(
            y_test_afm_bgt, y_test_pred_probas_afm_bgt
        )
        # metrics train
        y_train_pred_probas_afm_bgt = lr.predict_proba(X_train_)[:, 1]
        logs[f"fold{i}"]["afm_bgt_train"] = compute_metrics(
            y_train_afm_bgt, y_train_pred_probas_afm_bgt
        )

        # PFA
        X_train_pfa, y_train_pfa = encode_pfa(task_sessions_train, qmatrix)
        X_test_pfa, y_test_pfa = encode_pfa(task_sessions_test, qmatrix)
        X_train_, X_test_ = scale_features(
            X_train_pfa, X_test_pfa, method=params["scale_features_method"]
        )
        lr = LogisticRegression(max_iter=1000, solver="liblinear")
        lr.fit(X_train_, y_train_pfa, sample_weight=sample_weight)
        # metrics test
        y_test_pred_probas_pfa = lr.predict_proba(X_test_)[:, 1]
        logs[f"fold{i}"]["pfa"] = compute_metrics(y_test_pfa, y_test_pred_probas_pfa)
        # metrics train
        y_train_pred_probas_pfa = lr.predict_proba(X_train_)[:, 1]
        logs[f"fold{i}"]["pfa_train"] = compute_metrics(
            y_train_pfa, y_train_pred_probas_pfa
        )

        # item-avg
        item_avg_train = item_avg_predictor(task_sessions_train)
        # metrics test
        y_test_pred_item_avg_probas = [
            item_avg_train(item) for item in task_sessions_test["task"]
        ]
        logs[f"fold{i}"]["item_avg"] = compute_metrics(
            task_sessions_test["solved"], y_test_pred_item_avg_probas
        )
        # metrics train
        y_train_pred_item_avg_probas = [
            item_avg_train(item) for item in task_sessions_train["task"]
        ]
        logs[f"fold{i}"]["item_avg_train"] = compute_metrics(
            task_sessions_train["solved"], y_train_pred_item_avg_probas
        )

        # global-avg
        global_avg_train = task_sessions_train["solved"].mean()
        # metrics test
        y_test_pred_global_avg_probas = [global_avg_train] * len(task_sessions_test)
        logs[f"fold{i}"]["global_avg"] = compute_metrics(
            task_sessions_test["solved"], y_test_pred_global_avg_probas
        )
        # metrics train
        y_train_pred_global_avg_probas = [global_avg_train] * len(task_sessions_train)
        logs[f"fold{i}"]["global_avg_train"] = compute_metrics(
            task_sessions_train["solved"], y_train_pred_global_avg_probas
        )

    # average folds statistics
    logs["afm_bgt"] = average_metrics(
        [logs[f"fold{i}"]["afm_bgt"] for i in range(n_splits)]
    )
    logs["afm_bg"] = average_metrics(
        [logs[f"fold{i}"]["afm_bg"] for i in range(n_splits)]
    )
    logs["pfa"] = average_metrics([logs[f"fold{i}"]["pfa"] for i in range(n_splits)])
    logs["item_avg"] = average_metrics(
        [logs[f"fold{i}"]["item_avg"] for i in range(n_splits)]
    )
    logs["global_avg"] = average_metrics(
        [logs[f"fold{i}"]["global_avg"] for i in range(n_splits)]
    )

    # save log file
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path_to_results = os.path.join(path_algos, f"data/results_{date_str}.json")

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    with open(path_to_results, "w") as f:
        json.dump(logs, f, indent=4, cls=NumpyEncoder)
