'''
    @author Mr Dk.
    @version 2019.01.06
'''

import os
import time
import random
import pandas as pd
from multiprocessing import Process
from joblib import load


def predict(features, max_predictor, data, out_name, start_index, end_index):
    ids = []
    win_per = []

    cur = start_index

    while cur < end_index:
        feature = data.loc[cur, features].tolist()
        row_match_type = data.loc[cur, ["matchType"]][0]
        row_id = data.loc[cur, ["Id"]][0]

        print("Predicting a " + row_match_type + " type match record...")

        dir_name = "model/" + row_match_type
        count = 0
        res = 0.0
        # for model_name in random.sample(os.listdir(dir_name), max_predictor):
        if len(os.listdir(dir_name)) > max_predictor:
            for model_name in random.sample(os.listdir(dir_name), max_predictor):
                model_loc = os.path.join(dir_name, model_name)
                clf = load(model_loc)
                res += clf.predict([feature])
                count += 1
        else:
            for model_name in os.listdir(dir_name):
                model_loc = os.path.join(dir_name, model_name)
                clf = load(model_loc)
                res += clf.predict([feature])
                count += 1

        res = res / count
        print("Result predict by " + str(count) + " models, the result is: " + str(res))
        ids.append(row_id)
        win_per.append(res[0])

        # break  # for debug purpose

        cur += 1
        print(cur)
        if cur % 500 == 0:
            print("Process: " + str(cur-start_index) + "/" + str(end_index-start_index) +
                  " (" + str((cur-start_index)/(end_index-start_index)*100) + "%)")

    df = pd.DataFrame(data={"Id": ids, "winPlacePerc": win_per})
    df.to_csv(out_name + ".csv", header=True, index=False)


def predict_main(source, features, max_predictor, out_name, max_processes):
    start_time = time.time()

    data = pd.read_csv(source)
    print(data.shape)
    lines = data.shape[0]

    processes = []

    for i in range(max_processes):
        start_idx = int(lines * i / max_processes)
        end_idx = int(lines * (i + 1) / max_processes)
        p = Process(target=predict, args=(features, max_predictor, data, out_name + str(i), start_idx, end_idx))
        processes.append(p)

    for i in range(max_processes):
        processes[i].start()

    for i in range(max_processes):
        processes[i].join()

    end_time = time.time()
    print("Finished in " + str(end_time - start_time) + " seconds.")


if __name__ == "__main__":
    f = ["assists", "DBNOs", "headshotKills", "killPoints", "kills", "killStreaks", "walkDistance"]
    max_predict = 500
    predict_main("data/test_V2.csv", f, max_predict, "data/out", 6)
