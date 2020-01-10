import time
import pickle
import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor


def train(features, predictor):
    data = pd.read_csv("data/train_V2.csv").dropna(axis=0)
    feature_array = data[features].values
    labels = data["winPlacePerc"].values
    lines = data.shape[0]

    new_feature_array = feature_array

    print(labels)
    print(new_feature_array)
    print("Training model...")

    if not os.path.isdir("model"):
        os.mkdir("model")

    for i in range(predictor):
        start_time = time.time()
        model = RandomForestRegressor()
        model = model.fit(new_feature_array[int(lines * i / predictor):int(lines * (i+1) / predictor)],
                          labels[int(lines * i / predictor):int(lines * (i+1) / predictor)])
        end_time = time.time()
        print("Complete in " + str(end_time - start_time) + " seconds.")
        print("Processed " + str(int(lines * (i+1) / predictor) - int(lines * i / predictor)) + " rows.")

        print("Dumping the model into file...")
        model_file = open("model/model" + str(i) + ".pkl", "wb")
        pickle.dump(model, model_file)
        model_file.close()


def predict(features, predictor):
    print("Decomposition of testing data")
    data = pd.read_csv("data/test_V2.csv")
    feature_array = data[features].values
    new_feature = feature_array

    final = np.zeros(len(feature_array))

    for i in range(predictor):
        start_time = time.time()
        print("Loading model" + str(i) + "...")
        pickle_file = open("model/model" + str(i) + ".pkl", "rb")
        model = pickle.load(pickle_file)
        pickle_file.close()

        print("Predicting...")
        predic_res = model.predict(new_feature)

        for j in range(len(final)):
            final[j] += predic_res[j]
        end_time = time.time()
        print("Prediction: " + str(end_time - start_time) + "s.")

    for i in range(len(final)):
        final[i] /= predictor

    # print(res)
    df = pd.DataFrame(data={"Id": data["Id"].values, "winPlacePerc": final})
    df.to_csv("data/submission.csv", header=True, index=False)


if __name__ == '__main__':
    f = ["walkDistance", "killStreaks", "rideDistance", "kills", "heals", "boosts",
         "damageDealt", "weaponsAcquired", "headshotKills", "teamKills", "roadKills", "swimDistance", "revives",
         "assists", "killPlace", "longestKill", "vehicleDestroys",
         "rankPoints", "killPoints", "DBNOs"]
    train(f, 100)
    predict(f, 100)
