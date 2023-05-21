import sys
sys.path.append("C:\\Users\\Amit Vikram Raj\\#Self_Learning\\#ProjectPro\\Projects\\Business-License-Status-Prediction\\src\\ML_Pipeline")

import json
import pandas as pd
import preprocessing
import save_and_load_model
import predict
from ML_Pipeline.constants import MODEL_SAVE_PATH
from flask import Flask
from flask import request


app = Flask(__name__)

model_path = MODEL_SAVE_PATH + "\dnn_model"
ml_model, columns = save_and_load_model.load_model(model_path)

@app.post("/get_license_status")
async def get_license_status():
    items = json.load(request.data)
    
    test_df = pd.DataFrame([items], columns=items.keys())
    processed_df = preprocessing.apply(test_df)

    prediction = predict.predict(processed_df, ml_model, columns)

    print(prediction[0])
    return prediction[0]

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)