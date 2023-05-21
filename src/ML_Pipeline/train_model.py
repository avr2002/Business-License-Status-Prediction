import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from ML_Pipeline.constants import TARGET, LEARNING_RATE


def train(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train,
              epochs=5,
              batch_size=32,
              validation_data=(X_test, y_test),
              workers=6)
    
    return model

def fit(data):
    """
    Function to initiate the model and training data
    """
    columns = data.columns
    X_train, y_train, X_test, y_test = data_train_test_split(data)
    
    # Create the model
    model = keras.Sequential([layers.InputLayer(input_shape=X_train.shape[1]),
                              layers.Dense(32, activation="relu"),
                              layers.Dense(32, activation="relu"),
                              layers.Dense(3, activation="softmax")])
    
    # Compile the model
    model.compile(loss="categorical_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  metrics=['accuracy'])
    
    print("\n", model.summary(), "\n")

    model = train(model, X_train, y_train, X_test, y_test)

    return model, columns


def data_train_test_split(data):
    X = data.drop(TARGET, axis=1)
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1,
                                                        random_state=42, stratify=y)


    # Converting into numpy array
    X_train, X_test, y_train, y_test = X_train.values, X_test.values,\
                                    y_train.values, y_test.values
    
    print(f"X_train Shape: {X_train.shape}, y_train Shape:{y_train.shape}")
    print(f"X_test Shape: {X_test.shape}, y_train Shape:{y_test.shape}\n")
    
    return X_train, y_train, X_test, y_test
