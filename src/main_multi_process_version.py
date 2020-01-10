'''
    @author Mr Dk.
    @version 2020/01/08
'''


import os
import time
import random
import pickle
import pandas as pd
from multiprocessing import Process
from sklearn import tree


# Sort the CSV file by columns ["matchType", "matchId", "groupId"]
# Easier for model training
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


# Generate model by the sorted data
# Store the whole model after training
def model_generate(features):
    start_time = time.time()
    models = {}

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

    # save the models
    if not os.path.isdir("model"):
        os.makedirs("model")
    pickle_file = open("model/models.pkl", "wb")
    pickle.dump(models, pickle_file)
    pickle_file.close()

    end_time = time.time()
    print("Finished in " + str(end_time - start_time) + " seconds.")
    return models


# Use the model to predict data in the specific range (each process)
def predict(features, max_predictor, data, out_name, start_index, end_index,
            models_dict, process_index):
    ids = []
    win_per = []

    cur = start_index
    print_time = time.time()

    while cur < end_index:
        feature = data.loc[cur, features].tolist()
        row_match_type = data.loc[cur, ["matchType"]][0]
        row_id = data.loc[cur, ["Id"]][0]

        # print("Predicting a " + row_match_type + " type match record...")

        count = 0
        res = 0.0

        if len(models_dict.get(row_match_type)) > max_predictor:
            for clf in random.sample(models_dict.get(row_match_type), max_predictor):
                res += clf.predict([feature])
                count += 1
        else:
            for clf in models_dict.get(row_match_type):
                res += clf.predict([feature])
                count += 1

        res = res / count
        # print("Result predict by " + str(count) + " models, the result is: " + str(res))
        ids.append(row_id)
        win_per.append(res[0])

        cur += 1
        if cur % 2000 == 0:
            print("Process" + str(process_index) + ": " +
                  str(cur-start_index) + "/" + str(end_index-start_index) +
                  " (" + str((cur-start_index)/(end_index-start_index)*100) + "%)")
            cur_time = time.time()
            print(cur_time - print_time)
            print_time = cur_time

    df = pd.DataFrame(data={"Id": ids, "winPlacePerc": win_per})
    df.to_csv(out_name + ".csv", header=True, index=False)


# Schedule the job to multiple processes
def predict_main(source, features, max_predictor, out_name, max_processes):
    start_time = time.time()

    pickle_file = open("model/models.pkl", "rb")
    models = pickle.load(pickle_file)
    pickle_file.close()

    data = pd.read_csv(source)
    print(data.shape)
    lines = data.shape[0]

    processes = []

    for i in range(max_processes):
        start_idx = int(lines * i / max_processes)
        end_idx = int(lines * (i + 1) / max_processes)
        p = Process(target=predict, args=(features, max_predictor, data, out_name + str(i), start_idx, end_idx, models, i))
        processes.append(p)

    for i in range(max_processes):
        processes[i].start()

    for i in range(max_processes):
        processes[i].join()

    end_time = time.time()
    print("Finished in " + str(end_time - start_time) + " seconds.")


# Combine the data set generated by each process
def combine(in_name, processes, out_name):
    df = pd.DataFrame()
    for i in range(processes):
        data = pd.read_csv(in_name + str(i) + ".csv")
        df = df.append(data)
    df.to_csv(out_name, header=True, index=False)


# main
if __name__ == "__main__":
    core = 11
    max_predictor = 2000
    f = ["walkDistance", "killStreaks", "rideDistance", "kills", "heals", "boosts",
         "damageDealt", "weaponsAcquired", "headshotKills", "teamKills", "roadKills", "swimDistance", "revives",
         "assists", "killPlace", "longestKill", "vehicleDestroys",
         "rankPoints", "killPoints", "DBNOs"]
         
    # pre_process_sort("data/train_V2.csv")
    model = model_generate(f)

    # predict_main("data/test_V2.csv", f, max_predictor, "data/submit", core)
    # combine("data/submit", core, "data/final.csv")
