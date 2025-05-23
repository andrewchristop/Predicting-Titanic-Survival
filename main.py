import pandas as pd
import numpy as np
import os
import tensorflow as tf
import tensorflow_decision_forests as tfdf
from sklearn.model_selection import train_test_split

train = pd.read_csv("./data/train.csv")
predict = pd.read_csv("./data/test.csv")

#x = features, y= labels

x = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin']]
y = train[['Survived']]

predict_ds = predict[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

train_ds = pd.concat([x_train, y_train], axis=1)
test_ds = pd.concat([x_test, y_test], axis=1)

#print(train[['Survived', 'Name']])

train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds, label='Survived')
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_ds, label='Survived')
predict_ds = tfdf.keras.pd_dataframe_to_tf_dataset(predict_ds)

path = os.listdir('./model')

if(len(path) == 0):
  model_1 = tfdf.keras.RandomForestModel(verbose=2)
  model_1.fit(train_ds)
  model_1.compile(metrics=["accuracy"])
  evaluation = model_1.evaluate(test_ds, return_dict=True)
  print()
  
  for name, value in evaluation.items():
   # print(f"{name}: {value:.4f}")
   if (value >= 0.85):
     model_1.save("./model/")
     print(f"Model with {value:.4f} accuracy saved")

saved = tf.keras.models.load_model('./model')

results = saved.predict(predict_ds)
results = np.round(results).astype(int)
results = pd.DataFrame(results, columns=['Survived'])
final = pd.concat([predict[['PassengerId']], results], axis=1)
print(final)
#print(final.info())
#print(predict[['PassengerId']].info())
#print(train['Survived'].unique().tolist().sort())

#print(train_ds.info())
#print(test_ds.info())
#print(train.info())
#print(test.info())
