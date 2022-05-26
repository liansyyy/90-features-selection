import catboost
import lightgbm
import numpy as np
import pandas as pd
import sklearn.svm
import xgboost
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

#
# # load data
# data = pd.read_csv(r'../data/90-10percent-label.csv').values
# feat = np.asarray(data[:, 0:-1])
# label = np.asarray(data[:, -1] - 1)
# print(label)
#
# # split data into train & validation (70 -- 30)
# xtrain, xtest, ytrain, ytest = train_test_split(feat, label, test_size=0.3, stratify=label, random_state=1)
# fold = {'xt': xtrain, 'yt': ytrain, 'xv': xtest, 'yv': ytest}
#
# # # parameter
# k = 1  # k-value in KNN
#
# # model with selected features
# num_train = np.size(xtrain, 0)
# num_valid = np.size(xtest, 0)
# x_train = xtrain[:, :]
# y_train = ytrain.reshape(num_train)  # Solve bug
# x_valid = xtest[:, :]
# y_valid = ytest.reshape(num_valid)  # Solve bug
#
# knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
# knn.fit(x_train, y_train)
# y_pred = knn.predict(x_valid)
# # accuracy
# print("KNN")
# print("Accuracy: ", metrics.accuracy_score(y_valid, y_pred))
# print("macro Precision: ", metrics.precision_score(y_valid, y_pred, average="macro"), "  micro Precision: ",
#       metrics.precision_score(y_valid, y_pred, average="micro"))
# print("macro Recall: ", metrics.recall_score(y_valid, y_pred, average="macro"), "  micro Recall: ",
#       metrics.recall_score(y_valid, y_pred, average="micro"))
# print("macro F1-score: ", metrics.f1_score(y_valid, y_pred, average="macro"), "  micro F1-score: ",
#       metrics.f1_score(y_valid, y_pred, average="micro"))
# print()
#
# svc = sklearn.svm.SVC()
# svc.fit(x_train, y_train)
# y_pred = svc.predict(x_valid)
# # accuracy
# print("SVC")
# print("Accuracy: ", metrics.accuracy_score(y_valid, y_pred))
# print("macro Precision: ", metrics.precision_score(y_valid, y_pred, average="macro"), "  micro Precision: ",
#       metrics.precision_score(y_valid, y_pred, average="micro"))
# print("macro Recall: ", metrics.recall_score(y_valid, y_pred, average="macro"), "  micro Recall: ",
#       metrics.recall_score(y_valid, y_pred, average="micro"))
# print("macro F1-score: ", metrics.f1_score(y_valid, y_pred, average="macro"), "  micro F1-score: ",
#       metrics.f1_score(y_valid, y_pred, average="micro"))
# print()
#
# decisionTree = DecisionTreeClassifier()
# decisionTree.fit(x_train, y_train)
# y_pred = decisionTree.predict(x_valid)
# # accuracy
# print("decision tree")
# print("Accuracy: ", metrics.accuracy_score(y_valid, y_pred))
# print("macro Precision: ", metrics.precision_score(y_valid, y_pred, average="macro"), "  micro Precision: ",
#       metrics.precision_score(y_valid, y_pred, average="micro"))
# print("macro Recall: ", metrics.recall_score(y_valid, y_pred, average="macro"), "  micro Recall: ",
#       metrics.recall_score(y_valid, y_pred, average="micro"))
# print("macro F1-score: ", metrics.f1_score(y_valid, y_pred, average="macro"), "  micro F1-score: ",
#       metrics.f1_score(y_valid, y_pred, average="micro"))
#
# rf = RandomForestClassifier()
# rf.fit(x_train, y_train)
# y_pred = rf.predict(x_valid)
# # accuracy
# print()
# print("random forest")
# print("Accuracy: ", metrics.accuracy_score(y_valid, y_pred))
# print("macro Precision: ", metrics.precision_score(y_valid, y_pred, average="macro"), "  micro Precision: ",
#       metrics.precision_score(y_valid, y_pred, average="micro"))
# print("macro Recall: ", metrics.recall_score(y_valid, y_pred, average="macro"), "  micro Recall: ",
#       metrics.recall_score(y_valid, y_pred, average="micro"))
# print("macro F1-score: ", metrics.f1_score(y_valid, y_pred, average="macro"), "  micro F1-score: ",
#       metrics.f1_score(y_valid, y_pred, average="micro"))
#
# gbdt = GradientBoostingClassifier()
# gbdt.fit(x_train, y_train)
# y_pred = gbdt.predict(x_valid)
# # accuracy
# print()
# print("GBDT")
# print("Accuracy: ", metrics.accuracy_score(y_valid, y_pred))
# print("macro Precision: ", metrics.precision_score(y_valid, y_pred, average="macro"), "  micro Precision: ",
#       metrics.precision_score(y_valid, y_pred, average="micro"))
# print("macro Recall: ", metrics.recall_score(y_valid, y_pred, average="macro"), "  micro Recall: ",
#       metrics.recall_score(y_valid, y_pred, average="micro"))
# print("macro F1-score: ", metrics.f1_score(y_valid, y_pred, average="macro"), "  micro F1-score: ",
#       metrics.f1_score(y_valid, y_pred, average="micro"))
#
# lightgbm = lightgbm.LGBMClassifier()
# lightgbm.fit(x_train, y_train)
# y_pred = lightgbm.predict(x_valid)
# # accuracy
# print()
# print("LightGBM")
# print("Accuracy: ", metrics.accuracy_score(y_valid, y_pred))
# print("macro Precision: ", metrics.precision_score(y_valid, y_pred, average="macro"), "  micro Precision: ",
#       metrics.precision_score(y_valid, y_pred, average="micro"))
# print("macro Recall: ", metrics.recall_score(y_valid, y_pred, average="macro"), "  micro Recall: ",
#       metrics.recall_score(y_valid, y_pred, average="micro"))
# print("macro F1-score: ", metrics.f1_score(y_valid, y_pred, average="macro"), "  micro F1-score: ",
#       metrics.f1_score(y_valid, y_pred, average="micro"))
#
# xgboost = xgboost.XGBClassifier()
# xgboost.fit(x_train, y_train)
# y_pred = xgboost.predict(x_valid)
# # accuracy
# print()
# print("XGBoost")
# print("Accuracy: ", metrics.accuracy_score(y_valid, y_pred))
# print("macro Precision: ", metrics.precision_score(y_valid, y_pred, average="macro"), "  micro Precision: ",
#       metrics.precision_score(y_valid, y_pred, average="micro"))
# print("macro Recall: ", metrics.recall_score(y_valid, y_pred, average="macro"), "  micro Recall: ",
#       metrics.recall_score(y_valid, y_pred, average="micro"))
# print("macro F1-score: ", metrics.f1_score(y_valid, y_pred, average="macro"), "  micro F1-score: ",
#       metrics.f1_score(y_valid, y_pred, average="micro"))
#
# catBoost = catboost.CatBoostClassifier()
# catBoost.fit(x_train, y_train)
# y_pred = catBoost.predict(x_valid)
# # accuracy
# print()
# print("CatBoost")
# print("Accuracy: ", metrics.accuracy_score(y_valid, y_pred))
# print("macro Precision: ", metrics.precision_score(y_valid, y_pred, average="macro"), "  micro Precision: ",
#       metrics.precision_score(y_valid, y_pred, average="micro"))
# print("macro Recall: ", metrics.recall_score(y_valid, y_pred, average="macro"), "  micro Recall: ",
#       metrics.recall_score(y_valid, y_pred, average="micro"))
# print("macro F1-score: ", metrics.f1_score(y_valid, y_pred, average="macro"), "  micro F1-score: ",
#       metrics.f1_score(y_valid, y_pred, average="micro"))

from clf.model import classifier

classifier([i for i in range(7, 99)])
