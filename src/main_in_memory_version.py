

import os
import time
import random
import pandas as pd
from sklearn import tree


# train_data: train_V2.csv
def pre_process_sort(train_data):
    start_time = time.time()

    if not os.path.isdir("data"):
        os.makedirs("data")

    data = pd.read_csv(train_data)
    print(data.shape)

    data = data.sort_values(by=["matchType", "matchId", "groupId"], ascending=(False, True, True))
    data.to_csv("data/sorted_train.csv", header=True, index=False)

    end_time = time.time()
    print("Finished in " + str(end_time - start_time) + " seconds.")


models = {}


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

    while cur < lines:
        row_match_id = data.loc[cur, ["matchId"]][0]
        row_match_type = data.loc[cur, ["matchType"]][0]

        # Enter an new match
        # 1. Train and save the old model
        # 2. Reinitialize
        if match_id != "" and row_match_id != match_id and len(label_array) > 0:
            clf = tree.DecisionTreeRegressor()
            clf = clf.fit(feature_array, label_array)

            if match_type not in models:
                models[match_type] = []
            models[match_type].append(clf)

            # print("Model trained by " + str(len(label_array)) + " rows of data.")
            # print("New match: " + row_match_id + " ,type: " + row_match_type)
            feature_array = []
            label_array = []

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


def predict(features, max_predictor, test_data_path, out_name):
    ids = []
    win_per = []

    data = pd.read_csv(test_data_path)
    print(data.shape)

    cur = start_index = 0
    end_index = data.shape[0]

    while cur < end_index:
        feature = data.loc[cur, features].tolist()
        row_match_type = data.loc[cur, ["matchType"]][0]
        row_id = data.loc[cur, ["Id"]][0]

        # print("Predicting a " + row_match_type + " type match record...")

        count = 0
        res = 0.0

        if len(models.get(row_match_type)) > max_predictor:
            for clf in random.sample(models.get(row_match_type)):
                res += clf.predict([feature])
                count += 1
        else:
            for clf in models.get(row_match_type):
                res += clf.predict([feature])
                count += 1

        res = res / count
        # print("Result predict by " + str(count) + " models, the result is: " + str(res))
        ids.append(row_id)
        win_per.append(res[0])

        # break  # for debug purpose

        cur += 1
        if cur % 2000 == 0:
            print("Process: " + str(cur-start_index) + "/" + str(end_index-start_index) +
                  " (" + str((cur-start_index)/(end_index-start_index)*100) + "%)")

    df = pd.DataFrame(data={"Id": ids, "winPlacePerc": win_per})
    df.to_csv(out_name + ".csv", header=True, index=False)


# main

f = ["assists", "DBNOs", "headshotKills", "killPoints", "kills", "killStreaks", "walkDistance"]
pre_process_sort("data/train_V2.csv")
model_generate(f)
predict(f, 2000, "data/test_V2.csv", "data/submit.csv")
