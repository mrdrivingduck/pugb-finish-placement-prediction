'''
    @author Mr Dk.
    @version 2019.01.05
        https://scikit-learn.org/stable/modules/tree.html
        https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html#sphx-glr-auto-examples-tree-plot-tree-regression-py
        https://scikit-learn.org/stable/modules/model_persistence.html
'''


import os
import time
import pandas as pd
from sklearn import tree
from joblib import dump

# features = ["assists", "DBNOs", "headshotKills", "killPoints", "kills", "killStreaks", "walkDistance"]


def model_generate(features):
    start_time = time.time()

    data = pd.read_csv("data/sorted_train.csv")
    print(data.columns)
    print(data.shape)
    lines = data.shape[0]

    cur = 0
    feature_array = []
    label_array = []
    match_id = ""
    match_type = ""

    if not os.path.isdir("model"):
        os.makedirs("model")

    while cur < lines:
        row_match_id = data.loc[cur, ["matchId"]][0]
        row_match_type = data.loc[cur, ["matchType"]][0]

        # Enter an new match
        # 1. Train and save the old model
        # 2. Reinitialize
        if match_id != "" and row_match_id != match_id and len(label_array) > 0:
            clf = tree.DecisionTreeRegressor()
            clf = clf.fit(feature_array, label_array)

            if not os.path.isdir("model/" + match_type):
                os.makedirs("model/" + match_type)
            dump(clf, "model/" + match_type + "/" + match_id + ".joblib")
            print("Dump: the model of match " + str(match_id) + " saved.")
            print("Model trained by " + str(len(label_array)) + " rows of data.")

            # newClf = load("model/" + matchId + ".joblib")
            # print(newClf.predict([[0, 0, 0, 0, 2, 1, 2017]]))
            # break

            feature_array = []
            label_array = []
            print("New match: " + row_match_id + " ,type: " + row_match_type)

        match_id = row_match_id
        match_type = row_match_type
        feature = data.loc[cur, features].tolist()
        label = data.loc[cur, ["winPlacePerc"]][0]
        if pd.isnull(label):  # Dealing with NaN
            cur += 1
            continue
        feature_array.append(feature)
        label_array.append(label)

        cur += 1
        if cur % 5000 == 0:
            print("Process: " + str(cur) + "/" + str(lines) + " (" + str(cur/lines*100) + "%)")

    end_time = time.time()
    print("Finished in " + str(end_time - start_time) + " seconds.")
