'''
    @author Mr Dk.
    @version 2019.01.05
        https://scikit-learn.org/stable/modules/tree.html
        https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html#sphx-glr-auto-examples-tree-plot-tree-regression-py
        https://scikit-learn.org/stable/modules/model_persistence.html
'''

import pandas as pd
import os
from sklearn import tree
from joblib import dump, load

features = ["assists", "DBNOs", "headshotKills", "killPoints", "kills", "killStreaks", "walkDistance"]
# modes = ["solo", "duo", "squad", "solo-fpp", "duo-fpp", "squad-fpp"]

data = pd.read_csv("data/sorted.csv")
print(data.columns)
print(data.shape)
lines = data.shape[0]

cur = 0
featureArray = []
labelArray = []
matchId = ""
matchType = ""

if not os.path.isdir("model"):
    os.makedirs("model")

while cur < lines:
    rowMatchId = data.loc[cur, ["matchId"]][0]
    rowMatchType = data.loc[cur, ["matchType"]][0]

    # Enter an new match
    # 1. Train and save the old model
    # 2. Reinitialize
    if matchId != "" and rowMatchId != matchId:
        clf = tree.DecisionTreeRegressor()
        clf = clf.fit(featureArray, labelArray)

        if not os.path.isdir("model/" + rowMatchType):
            os.makedirs("model/" + rowMatchType)
        dump(clf, "model/" + rowMatchType + "/" + matchId + ".joblib")
        print("Dump: the model of match " + str(matchId) + " saved.")
        print("Model trained by " + str(len(labelArray)) + " rows of data.")

        # newClf = load("model/" + matchId + ".joblib")
        # print(newClf.predict([[0, 0, 0, 0, 2, 1, 2017]]))
        # break

        featureArray = []
        labelArray = []
        print("New match: " + rowMatchId + " ,type: " + rowMatchType)

    matchId = rowMatchId
    matchType = rowMatchType
    feature = data.loc[cur, features].tolist()
    label = data.loc[cur, ["winPlacePerc"]][0]
    featureArray.append(feature)
    labelArray.append(label)

    cur += 1
    if cur % 5000 == 0:
        print("Process: " + str(cur) + "/" + str(lines) + " (" + str(cur/lines*100) + "%)")

