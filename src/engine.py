# import sys
# sys.path.append(r"..")
# sys.path.append(r"../src")
# sys.path.append(r"../src/ML_Pipeline")

import os
import subprocess
import pandas as pd
from ML_Pipeline.preprocessing import apply
from ML_Pipeline import train_model
from ML_Pipeline.save_and_load_model import save_model, load_model
from ML_Pipeline import predict
from ML_Pipeline.constants import MODEL_SAVE_PATH, INPUT_DATA_FILE_PATH, TEST_DATA_FILE_PATH

import warnings
warnings.filterwarnings("ignore")


val = int(input("Train - 0\nPredict - 1\nDeploy - 2\nEnter your value: "))

if val == 0:
    data_file_path = INPUT_DATA_FILE_PATH 
    data = pd.read_csv(data_file_path, low_memory=False)
    data = data.drop_duplicates().reset_index(drop=True)

    print("\nData Successfully Loaded into Pandas DataFrame")

    processed_df = apply(data)

    ml_model, columns = train_model.fit(processed_df)
    print("\nModel Training Completed")

    model_saved = save_model(ml_model, columns)
    print(f"Model Saved at '{MODEL_SAVE_PATH}' in dnn_model\n")
elif val==1:
    # Load the saved model
    SAVED_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, "dnn_model")
    ml_model, columns = load_model(SAVED_MODEL_PATH)

    # Load the test data
    test_data = pd.read_csv(TEST_DATA_FILE_PATH)
    test_data = test_data.drop_duplicates().reset_index(drop=True)

    # Pre-Process the test data
    processed_test_data = apply(test_data)

    # Make Predictions
    predictions = predict.predict(processed_test_data, ml_model, columns)
    print("\n",predictions,"\n")
    print("Model Predictions on Test Data is Completed!\n")
else:
    # sys.path.append(r"../src/ML_Pipeline")
    # For production deployment
    '''process = subprocess.Popen(['sh', 'ML_Pipeline/wsgi.sh'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               universal_newlines=True)'''

    # For dev deployment
    process = subprocess.Popen(['python', "src/ML_Pipeline/deploy.py"],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                universal_newlines=True)
    # print(os.getcwd())
    
    for stdout_line in process.stdout:
        print(stdout_line)
    
    stdout, stderr = process.communicate()
    print(stdout, stderr)