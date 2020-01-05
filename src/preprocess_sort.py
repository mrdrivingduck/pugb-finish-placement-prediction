'''
    @author Mr Dk.
    @version 2019.01.05
'''

import os
import time
import pandas as pd


def preprocess_sort(train_data):
    start_time = time.time()

    if not os.path.isdir("data"):
        os.makedirs("data")

    data = pd.read_csv(train_data)
    print(data.shape)

    data = data.sort_values(by=["matchType", "matchId", "groupId"], ascending=(False, True, True))
    data.to_csv("data/sorted_train.csv", header=True, index=False)

    end_time = time.time()
    print("Finished in " + str(end_time - start_time) + " seconds.")
