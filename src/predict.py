'''
    @author Mr Dk.
    @version 2019.01.05
'''

import os
import time
import random
import pandas as pd
from joblib import load

start_time = time.time()

features = ["assists", "DBNOs", "headshotKills", "killPoints", "kills", "killStreaks", "walkDistance"]
max_predictor = 500

data = pd.read_csv("data/test_V2.csv")
print(data.shape)
lines = data.shape[0]

ids = []
win_per = []

cur = 0

while cur < lines:
    feature = data.loc[cur, features].tolist()
    row_match_type = data.loc[cur, ["matchType"]][0]
    row_id = data.loc[cur, ["Id"]][0]

    print("Predicting a " + row_match_type + " type match record...")

    dir_name = "model/" + row_match_type
    count = 0
    res = 0.0
    for model_name in random.sample(os.listdir(dir_name), max_predictor):
    # for model_name in os.listdir(dir_name):
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
    if cur % 5000 == 0:
        print("Process: " + str(cur) + "/" + str(lines) + " (" + str(cur/lines*100) + "%)")

df = pd.DataFrame(data={"Id": ids, "winPlacePerc": win_per})
df.to_csv("data/res.csv", header=True, index=False)

end_time = time.time()
print("Finished in " + str(end_time - start_time) + " seconds.")
