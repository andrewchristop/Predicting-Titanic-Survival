import pandas as pd
import numpy as np
import os
import tensorflow as tf
import tensorflow_decision_forests as tfdf
from sklearn.model_selection import train_test_split

train = pd.read_csv("./data/train.csv")
predict = pd.read_csv("./data/test.csv")

#x = features, y= labels

x = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin']] #Elimination of useless features
y = train[['Survived']]

predict_ds = predict[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin']] #Columns to be selected by test data

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3) #30% of the training data dedicated for evaluation purposes

train_ds = pd.concat([x_train, y_train], axis=1) #Combining labels with features into distinct train-test dataframes
test_ds = pd.concat([x_test, y_test], axis=1)

train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds, label='Survived') #Conversion to TensorFlow datatype
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_ds, label='Survived')
predict_ds = tfdf.keras.pd_dataframe_to_tf_dataset(predict_ds)

path = os.listdir('./model')

if(len(path) == 0): #Model is only trained when the model directory is empty
  model_1 = tfdf.keras.RandomForestModel(verbose=2)
  model_1.fit(train_ds)
  model_1.compile(metrics=["accuracy"])
  evaluation = model_1.evaluate(test_ds, return_dict=True)
  print()
  model_1.save("./model/")

saved = tf.keras.models.load_model('./model') #Loads saved model

results = saved.predict(predict_ds) #Runs inference/prediction against the model
results = np.round(results).astype(int) #Converts output to binary data
results = pd.DataFrame(results, columns=['Survived']) 
final = pd.concat([predict[['PassengerId']], results], axis=1) #Displays the results so that PassengerId and Survived columns are side by side
final.to_csv('./prediction/prediction_result.csv', index=False) #Saves results as CSV
