import pandas as pd
from ML_Pipeline.constants import TARGET

def predict(test_data, model, columns):
    # Preprocess test data to do categorical encoding and match columns expected
    # by the model
    new_cols = [x for x in columns if x not in test_data.columns]

    new_df = pd.DataFrame(columns=new_cols, index=range(test_data.shape[0]))
    new_df.fillna(0, inplace=True)
    test_data = pd.concat([test_data, new_df.reindex(test_data.index)], axis=1)
    test_data = test_data[columns]
    test_data = test_data.loc[:, ~test_data.columns.duplicated()]

    for col in TARGET:
        try:
            test_data = test_data.drop(col, axis=1)
        except:
            continue

    X_test = test_data.values
    predict = model.predict(X_test)

    predict = match_predictions(predict)
    return predict


def match_predictions(predictions):
    mappings = {0:"AAC (license was cancelled during term)", 1:"AAI (license was issued)", 2:"REV (license was revoked)"}
    l = []
    for prediction in predictions:
        max_value = max(prediction)
        l.append(mappings[list(prediction).index(max_value)])
    return l