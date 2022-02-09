import pandas as pd
import configparser
import os

# from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import cross_val_score
import pickle
import json
import numpy as np
import uuid
import utilities

config = configparser.ConfigParser()
config.read("config.ini")

dataFrame = pd.read_csv(config["DIR_PATH"]["TRAIN_DATA_PATH"])

list_of_training_features = dict(config["TRAINING_FEATURES"])
list_of_target_labels = dict(config["TARGET_LABELS"])

print(list_of_training_features)

dataFrame_selected = dataFrame[list(list_of_training_features.keys())]

target_labels = dataFrame[list(list_of_target_labels)]

target_labels = utilities.transform_fedas(target_labels, col_label="correct_fedas_code")

######################################
# Scale the numerical features
######################################
dataFrame_selected = utilities.standard_normalization(
    dataFrame_selected, list_of_training_features
)
######################################
# One hot encoding for categorical data
######################################

encoded_input_features, _ = utilities.label_encode_columns(
    dataFrame_selected, list_of_training_features
)

######################################
# Model work - just a fixed type and parameters of model for now
######################################
rf = RandomForestClassifier()
clf = MultiOutputClassifier(rf)


scores = cross_val_score(clf, encoded_input_features, target_labels, cv=5)
print(f"Scores: {scores}")
mean_cv_score = np.mean(scores)

clf.fit(encoded_input_features, target_labels)

########################################
# Check with previous models performance.
# Store the model
########################################
model_unique_id = str(uuid.uuid4())
logs = json.load(open(config["DIR_PATH"]["MODEL_LOG"], "r"))

logs["all_models"][model_unique_id] = {"cv_score": mean_cv_score}

best_model = True
best_model_id = logs["best_model"]
if logs["all_models"][best_model_id]["cv_score"] >= mean_cv_score:
    best_model = False
    print(f"THIS MODEL DOESN'T OUTPERFORM MODEL {best_model_id}")
    print(
        f"Model {best_model_id} has {logs['all_models'][best_model_id]['cv_score']} while this model has {mean_cv_score}"
    )


if best_model:
    logs["best_model"] = model_unique_id

json.dump(logs, open(config["DIR_PATH"]["MODEL_LOG"], "w"))


os.mkdir(config["DIR_PATH"]["MODELS_STORED"])
pickle.dump(
    clf,
    open(config["DIR_PATH"]["MODELS_STORED"] + model_unique_id, "wb"),
    protocol=pickle.HIGHEST_PROTOCOL,
)
