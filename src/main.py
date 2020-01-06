'''
    @author Mr Dk.
    @version 2019.01.06
'''


from . import preprocess_sort
from . import model_generator
from . import predict

features = ["assists", "DBNOs", "headshotKills", "killPoints", "kills", "killStreaks", "walkDistance"]
max_predictor = 2000

preprocess_sort.preprocess_sort("data/train_V2.csv")
model_generator.model_generate(features)
predict.predict_main("data/test_V1.csv", features, max_predictor, "data/out", 4)
