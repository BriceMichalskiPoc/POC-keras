import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tabulate import tabulate

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

np.set_printoptions(precision=3, suppress=True)
tf.random.set_seed(20051995)
np.random.seed(20051995)


url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
column_names = [
    "MPG",
    "Cylinders",
    "Displacement",
    "Horsepower",
    "Weight",
    "Acceleration",
    "Model Year",
    "Origin",
]

raw_dataset = pd.read_csv(
    "data/auto.csv",
    names=column_names,
    na_values="?",
    comment="\t",
    sep=" ",
    skipinitialspace=True,
)
dataset = raw_dataset.copy()
dataset = dataset.dropna()
dataset["Origin"] = dataset["Origin"].map({1: "USA", 2: "Europe", 3: "Japan"})
dataset = pd.get_dummies(dataset, prefix="", prefix_sep="")

# TRAIN
train_dataset = dataset.sample(frac=0.9, random_state=1)
test_dataset = dataset.drop(train_dataset.index)
sns_plot = sns.pairplot(
    train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde"
)
sns_plot.savefig("output2.png")


train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop("MPG")
test_labels = test_features.pop("MPG")


## MODEL
horsepower = np.array(train_features["Horsepower"])
horsepower_normalizer = preprocessing.Normalization(
    input_shape=[
        1,
    ],
    name="LAST",
)
horsepower_normalizer.adapt(horsepower)

horsepower_model = tf.keras.Sequential([horsepower_normalizer, layers.Dense(units=1)])


summary = []

horsepower_model.summary(print_fn=summary.append(x))

with open("model.md", "w") as f:
    f.write(
        """
# Model Layers Visualisation

## Current model:

```python
{summary}
```
    """.format(
            summary="\n".join(summary)
        )
    )
