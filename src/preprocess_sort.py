'''
    @author Mr Dk.
    @version 2019.01.05
'''

import time
import pandas as pd

start_time = time.time()

data = pd.read_csv("data/train_V2.csv")
print(data.shape)

data = data.sort_values(by=["matchType", "matchId", "groupId"], ascending=(False, True, True))
data.to_csv("data/temp.csv", header=True, index=False)

end_time = time.time()
print("Finished in " + str(end_time - start_time) + " seconds.")
