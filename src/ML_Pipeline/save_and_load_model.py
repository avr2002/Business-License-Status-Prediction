import os
import keras
import pickle
from ML_Pipeline.constants import MODEL_SAVE_PATH
# import time

def save_model(model, columns):
    # file_name = 'dnn_model_' + time.strftime("%Y%m%d-%H%M%S") + '.h5'
    # save_path = "..\..\output\saved_models\neural_network_models" 
    # model_save_path = os.path.join(save_path, file_name)

    model_save_path = os.path.join(MODEL_SAVE_PATH,"dnn_model")
    model.save(model_save_path)
    # model.save("output\saved_models\\neural_network_models\dnn_model")

    # Save column mappings
    column_mapping_save_path = os.path.join(MODEL_SAVE_PATH, "columns.mapping")
    file = open(column_mapping_save_path, "wb")
    # file = open("output\saved_models\\neural_network_models\columns.mapping", "wb")
    pickle.dump(columns, file)
    file.close()

    return True

def load_model(model_path):
    model=None
    try:
        model = keras.models.load_model(model_path)
    except:
        print("Please Enter the Correct Saved Model Path!")
        exit(0)

    column_mapping_save_path = os.path.join(MODEL_SAVE_PATH, "columns.mapping")
    file = open(column_mapping_save_path, "rb")

    columns = pickle.load(file)
    file.close()

    return model, columns
    