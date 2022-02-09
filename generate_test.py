import pandas as pd
import configparser
import pickle
import json
import utilities
import csv

config = configparser.ConfigParser()
config.read("config.ini")

dataFrame = pd.read_csv(config["DIR_PATH"]["TEST_DATA_PATH"])
logs = json.load(open(config["DIR_PATH"]["MODEL_LOG"], "r"))

best_model_id = logs["best_model"]

list_of_training_features = dict(config["TRAINING_FEATURES"])
dataFrame_selected = dataFrame[list(list_of_training_features.keys())]

########################################
# Transform the data
########################################
dataFrame_selected = utilities.standard_normalization(
    dataFrame_selected, list_of_training_features
)
one_hot_encoding_training, _ = utilities.label_encode_columns(
    dataFrame_selected, list_of_training_features
)
########################################
# Load best model
########################################
clf = pickle.load(open(config["DIR_PATH"]["MODELS_STORED"] + best_model_id, "rb"))

predictions = clf.predict(one_hot_encoding_training)
# print(predictions)
# print(predictions.shape)
final_output = utilities.inverse_transform_fedas(predictions)

print(predictions)
print(final_output)
with open(config["DIR_PATH"]["GENERATED_OUTPUT"], "w") as f:

    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerows(final_output)