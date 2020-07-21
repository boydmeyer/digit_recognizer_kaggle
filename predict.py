from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np

# load data
test_data = pd.read_csv('data/test.csv')

# make 0.1 numbers
test_data = test_data / 255.0

# reshape to 3 dimension
X_final = test_data.values.reshape(-1,28,28,1)

# load my model
model = load_model('data/mymodel.model')

# do prediction
results = model.predict(X_final)
results = np.argmax(results, axis=1)

# create 'final.csv'
def create_csv(prediction):
    pred_data = pd.Series(prediction, name="Label")
    submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), pred_data], axis=1)
    submission.to_csv("final.csv", index=False)

create_csv(results)

