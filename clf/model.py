import warnings

import numpy as np
import pandas as pd
import sklearn.svm
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


def classifier(sf):
    warnings.filterwarnings("ignore")
    data = pd.read_csv(r'../data/99-10percent-label.csv').values
    feat = np.asarray(data[:, 0:-1])
    label = np.asarray(data[:, -1] - 1)

    # split data into train & validation (70 -- 30)
    xtrain, xtest, ytrain, ytest = train_test_split(feat, label, test_size=0.3, stratify=label, random_state=1)
    fold = {'xt': xtrain, 'yt': ytrain, 'xv': xtest, 'yv': ytest}

    # model with selected features
    num_train = np.size(xtrain, 0)
    num_valid = np.size(xtest, 0)
    x_train = xtrain[:, sf]
    y_train = ytrain.reshape(num_train)  # Solve bug
    x_valid = xtest[:, sf]
    y_valid = ytest.reshape(num_valid)  # Solve bug

    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_valid)
    # accuracy
    print()
    print("LR")
    print("Accuracy: ", metrics.accuracy_score(y_valid, y_pred))
    print("macro Precision: ", metrics.precision_score(y_valid, y_pred, average="macro"), "  micro Precision: ",
          metrics.precision_score(y_valid, y_pred, average="micro"))
    print("macro Recall: ", metrics.recall_score(y_valid, y_pred, average="macro"), "  micro Recall: ",
          metrics.recall_score(y_valid, y_pred, average="micro"))
    print("macro F1-score: ", metrics.f1_score(y_valid, y_pred, average="macro"), "  micro F1-score: ",
          metrics.f1_score(y_valid, y_pred, average="micro"))
    print()

    knn = KNeighborsClassifier(n_neighbors=1, weights="distance")
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_valid)
    # accuracy
    print("KNN")
    print("Accuracy: ", metrics.accuracy_score(y_valid, y_pred))
    print("macro Precision: ", metrics.precision_score(y_valid, y_pred, average="macro"), "  micro Precision: ",
          metrics.precision_score(y_valid, y_pred, average="micro"))
    print("macro Recall: ", metrics.recall_score(y_valid, y_pred, average="macro"), "  micro Recall: ",
          metrics.recall_score(y_valid, y_pred, average="micro"))
    print("macro F1-score: ", metrics.f1_score(y_valid, y_pred, average="macro"), "  micro F1-score: ",
          metrics.f1_score(y_valid, y_pred, average="micro"))
    print()

    svc = sklearn.svm.SVC()
    svc.fit(x_train, y_train)
    y_pred = svc.predict(x_valid)
    # accuracy
    print()
    print("SVC")
    print("Accuracy: ", metrics.accuracy_score(y_valid, y_pred))
    print("macro Precision: ", metrics.precision_score(y_valid, y_pred, average="macro"), "  micro Precision: ",
          metrics.precision_score(y_valid, y_pred, average="micro"))
    print("macro Recall: ", metrics.recall_score(y_valid, y_pred, average="macro"), "  micro Recall: ",
          metrics.recall_score(y_valid, y_pred, average="micro"))
    print("macro F1-score: ", metrics.f1_score(y_valid, y_pred, average="macro"), "  micro F1-score: ",
          metrics.f1_score(y_valid, y_pred, average="micro"))
    print()

    dtree = DecisionTreeClassifier()
    dtree.fit(x_train, y_train)
    y_pred = dtree.predict(x_valid)
    # accuracy
    print("decision tree")
    print("Accuracy: ", metrics.accuracy_score(y_valid, y_pred))
    print("macro Precision: ", metrics.precision_score(y_valid, y_pred, average="macro"), "  micro Precision: ",
          metrics.precision_score(y_valid, y_pred, average="micro"))
    print("macro Recall: ", metrics.recall_score(y_valid, y_pred, average="macro"), "  micro Recall: ",
          metrics.recall_score(y_valid, y_pred, average="micro"))
    print("macro F1-score: ", metrics.f1_score(y_valid, y_pred, average="macro"), "  micro F1-score: ",
          metrics.f1_score(y_valid, y_pred, average="micro"))

    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_valid)
    # accuracy
    print()
    print("random forest")
    print("Accuracy: ", metrics.accuracy_score(y_valid, y_pred))
    print("macro Precision: ", metrics.precision_score(y_valid, y_pred, average="macro"), "  micro Precision: ",
          metrics.precision_score(y_valid, y_pred, average="micro"))
    print("macro Recall: ", metrics.recall_score(y_valid, y_pred, average="macro"), "  micro Recall: ",
          metrics.recall_score(y_valid, y_pred, average="micro"))
    print("macro F1-score: ", metrics.f1_score(y_valid, y_pred, average="macro"), "  micro F1-score: ",
          metrics.f1_score(y_valid, y_pred, average="micro"))

    gbdt = GradientBoostingClassifier()
    gbdt.fit(x_train, y_train)
    y_pred = gbdt.predict(x_valid)
    # accuracy
    print()
    print("GBDT")
    print("Accuracy: ", metrics.accuracy_score(y_valid, y_pred))
    print("macro Precision: ", metrics.precision_score(y_valid, y_pred, average="macro"), "  micro Precision: ",
          metrics.precision_score(y_valid, y_pred, average="micro"))
    print("macro Recall: ", metrics.recall_score(y_valid, y_pred, average="macro"), "  micro Recall: ",
          metrics.recall_score(y_valid, y_pred, average="micro"))
    print("macro F1-score: ", metrics.f1_score(y_valid, y_pred, average="macro"), "  micro F1-score: ",
          metrics.f1_score(y_valid, y_pred, average="micro"))

    xgb = XGBClassifier(verbosity=0)
    xgb.fit(x_train, y_train)
    y_pred = xgb.predict(x_valid)
    # accuracy
    print()
    print("XGBoost")
    print("Accuracy: ", metrics.accuracy_score(y_valid, y_pred))
    print("macro Precision: ", metrics.precision_score(y_valid, y_pred, average="macro"), "  micro Precision: ",
          metrics.precision_score(y_valid, y_pred, average="micro"))
    print("macro Recall: ", metrics.recall_score(y_valid, y_pred, average="macro"), "  micro Recall: ",
          metrics.recall_score(y_valid, y_pred, average="micro"))
    print("macro F1-score: ", metrics.f1_score(y_valid, y_pred, average="macro"), "  micro F1-score: ",
          metrics.f1_score(y_valid, y_pred, average="micro"))

    lgb = LGBMClassifier()
    lgb.fit(x_train, y_train)
    y_pred = lgb.predict(x_valid)
    # accuracy
    print()
    print("LightGBM")
    print("Accuracy: ", metrics.accuracy_score(y_valid, y_pred))
    print("macro Precision: ", metrics.precision_score(y_valid, y_pred, average="macro"), "  micro Precision: ",
          metrics.precision_score(y_valid, y_pred, average="micro"))
    print("macro Recall: ", metrics.recall_score(y_valid, y_pred, average="macro"), "  micro Recall: ",
          metrics.recall_score(y_valid, y_pred, average="micro"))
    print("macro F1-score: ", metrics.f1_score(y_valid, y_pred, average="macro"), "  micro F1-score: ",
          metrics.f1_score(y_valid, y_pred, average="micro"))

    catb = CatBoostClassifier(silent=1 == 1)
    catb.fit(x_train, y_train)
    y_pred = catb.predict(x_valid)
    # accuracy
    print()
    print()
    print("CatBoost")
    print("Accuracy: ", metrics.accuracy_score(y_valid, y_pred))
    print("macro Precision: ", metrics.precision_score(y_valid, y_pred, average="macro"), "  micro Precision: ",
          metrics.precision_score(y_valid, y_pred, average="micro"))
    print("macro Recall: ", metrics.recall_score(y_valid, y_pred, average="macro"), "  micro Recall: ",
          metrics.recall_score(y_valid, y_pred, average="micro"))
    print("macro F1-score: ", metrics.f1_score(y_valid, y_pred, average="macro"), "  micro F1-score: ",
          metrics.f1_score(y_valid, y_pred, average="micro"))

    mlp = MLPClassifier()
    mlp.fit(x_train, y_train)
    y_pred = mlp.predict(x_valid)
    # accuracy
    print()
    print("MLP")
    print("Accuracy: ", metrics.accuracy_score(y_valid, y_pred))
    print("macro Precision: ", metrics.precision_score(y_valid, y_pred, average="macro"), "  micro Precision: ",
          metrics.precision_score(y_valid, y_pred, average="micro"))
    print("macro Recall: ", metrics.recall_score(y_valid, y_pred, average="macro"), "  micro Recall: ",
          metrics.recall_score(y_valid, y_pred, average="micro"))
    print("macro F1-score: ", metrics.f1_score(y_valid, y_pred, average="macro"), "  micro F1-score: ",
          metrics.f1_score(y_valid, y_pred, average="micro"))
