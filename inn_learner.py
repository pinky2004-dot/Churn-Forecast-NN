# -*- coding: utf-8 -*-
"""

<center><font size=6> Bank Churn Prediction - Neural Networks </font></center>

## Problem Statement

### Context

Businesses like banks which provide service have to worry about problem of 'Customer Churn' i.e. customers leaving and joining another service provider. It is important to understand which aspects of the service influence a customer's decision in this regard. Management can concentrate efforts on improvement of service, keeping in mind these priorities.

### Objective

You as a Data scientist with the  bank need to  build a neural network based classifier that can determine whether a customer will leave the bank  or not in the next 6 months.

### Data Dictionary

* CustomerId: Unique ID which is assigned to each customer

* Surname: Last name of the customer

* CreditScore: It defines the credit history of the customer.
  
* Geography: A customer’s location
   
* Gender: It defines the Gender of the customer
   
* Age: Age of the customer
    
* Tenure: Number of years for which the customer has been with the bank

* NumOfProducts: refers to the number of products that a customer has purchased through the bank.

* Balance: Account balance

* HasCrCard: It is a categorical variable which decides whether the customer has credit card or not.

* EstimatedSalary: Estimated salary

* isActiveMember: Is is a categorical variable which decides whether the customer is active member of the bank or not ( Active member in the sense, using bank products regularly, making transactions etc )

* Exited : whether or not the customer left the bank within six month. It can take two values
** 0=No ( Customer did not leave the bank )
** 1=Yes ( Customer left the bank )
"""

"""## Importing necessary libraries"""

# Libraries to help with reading and manipulating data
import numpy as np
import pandas as pd
import time

# Libraries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Removes the limit for the number of displayed columns
pd.set_option("display.max_columns", None)
# Sets the limit for the number of displayed rows
pd.set_option("display.max_rows", 200)

from imblearn.over_sampling import SMOTE #importing SMOTE

# to split the data into train and test
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score,accuracy_score, recall_score, precision_score, classification_report

import tensorflow as tf #An end-to-end open source machine learning platform
from tensorflow import keras  # High-level neural networks API for deep learning.
from keras import backend   # Abstraction layer for neural network backend engines.
from keras.models import Sequential  # Model for building NN sequentially.
from keras.layers import Dense
from tensorflow.keras.layers import Dense, Input, Dropout,BatchNormalization
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.initializers import he_normal

# to suppress warnings
import warnings
warnings.filterwarnings("ignore")

"""## Loading the dataset"""

data = pd.read_csv('bank-1.csv')

df = data.copy()

"""## Data Overview"""

df.head()

df.tail()

df.shape

"""- There are a total of 10,000 rows and 14 columns"""

df.info()

"""- As expected, the surname is a string/object.
- Geography and Gender are also string/object.
- It looks like there are no missing values, but let me look a bit deeper into the data.
"""

df.duplicated().sum()

"""- No duplicated values"""

round(df.isnull().sum() / df.isnull().count() * 100, 2)

"""- There are no missing values."""

df['Exited'].value_counts(1)

"""- Only 20% of the customers left the bank
- 79% of the customers remained with the bank
"""

df.describe().T

"""- The minimum age of the customer is 18 years old, which makes sense because some banks require that the customer is atleast 18 years old.
- The average age in the bank is around 38 years.
- The oldest customer has an age of 92 years.
- The maximum tenure that a customer has been with the bank is 10 years.
- The minimum salary of a customer is 11.58 dollars and the maximum is around 200 thousand dollars.
- The average estimated salary is around 100 thousand dollars.
- The account balances range from 0 dollars to 251 thousand dollars, that is a huge difference.
"""

df.nunique()

"""- The 'RowNumber', 'CustomerId', 'Surname' are unique identifier for a customer, so we can drop this column because it will not be of use."""

df.sample(n=10, random_state=1) #random rows of data

for i in df.describe(include=['object']).columns:
  print("Unique values in", i, "are: ")
  print(df[i].value_counts())
  print("-" * 50)

"""###Convert to categorical variable"""

cols = df.select_dtypes(['object'])
cols.columns

for i in cols.columns:
  df[i] = df[i].astype('category')

df['Exited'] = df['Exited'].astype('category')

df['HasCrCard'] = df['HasCrCard'].astype('category')
df['IsActiveMember'] = df['IsActiveMember'].astype('category')

df['NumOfProducts'] = df['NumOfProducts'].astype('category')

df.info()

df.describe(include=['category']).T

df

df = df.drop(['CustomerId', 'RowNumber', 'Surname'], axis=1)
df.head()

"""## Exploratory Data Analysis"""

# function to create labeled barplots


def labeled_barplot(data, feature, perc=False, n=None):
    """
    Barplot with percentage at the top

    data: dataframe
    feature: dataframe column
    perc: whether to display percentages instead of count (default is False)
    n: displays the top n category levels (default is None, i.e., display all levels)
    """

    total = len(data[feature])  # length of the column
    count = data[feature].nunique()
    if n is None:
        plt.figure(figsize=(count + 1, 5))
    else:
        plt.figure(figsize=(n + 1, 5))

    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=data,
        x=feature,
        palette="Paired",
        order=data[feature].value_counts().index[:n].sort_values(),
    )

    for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(
                100 * p.get_height() / total
            )  # percentage of each class of the category
        else:
            label = p.get_height()  # count of each level of the category

        x = p.get_x() + p.get_width() / 2  # width of the plot
        y = p.get_height()  # height of the plot

        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        )  # annotate the percentage

    plt.show()  # show the plot

# function to plot a boxplot and a histogram along the same scale.


def histogram_boxplot(data, feature, figsize=(12, 7), kde=False, bins=None):
    """
    Boxplot and histogram combined

    data: dataframe
    feature: dataframe column
    figsize: size of figure (default (12,7))
    kde: whether to the show density curve (default False)
    bins: number of bins for histogram (default None)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,  # Number of rows of the subplot grid= 2
        sharex=True,  # x-axis will be shared among all subplots
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )  # creating the 2 subplots
    sns.boxplot(
        data=data, x=feature, ax=ax_box2, showmeans=True, color="violet"
    )  # boxplot will be created and a star will indicate the mean value of the column
    sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins, palette="winter"
    ) if bins else sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2
    )  # For histogram
    ax_hist2.axvline(
        data[feature].mean(), color="green", linestyle="--"
    )  # Add mean to the histogram
    ax_hist2.axvline(
        data[feature].median(), color="black", linestyle="-"
    )  # Add median to the histogram

"""### Univariate Analysis

####CreditScore
"""

histogram_boxplot(df, 'CreditScore')

"""- The credit score is a bit left skewed.
- Atleast 300 records have the highest credit scores.
- 50% of credit scores is between 600 and 700.
- If you look closely you can see there is a bell shaped curve, if you ignore the few records to the left.

####Geography
"""

labeled_barplot(df, 'Geography')

"""- The majority of the customers are from France.
- Germany and Spain are almost equal to each other.

####Gender
"""

labeled_barplot(df, 'Gender')

"""- The majority of the customers are male.
- Females are not so bad they are almost catching up to the males.

####Age
"""

histogram_boxplot(df, 'Age')

"""- There is an outlier to the far right that is the highest age of the customer.
- The age is right skewed, which means there are a lot of customers who are young or middle aged.
- There are 800+ records where the age is around 38-41 year olds, this is the highest age group.
- The second highest age group is 25-29 year olds.

####Tenure
"""

labeled_barplot(df, 'Tenure')

"""- 413 of the customers spend 0 years with the bank. I think they recently joined the bank or they left before the first year.
- 10 years was how long they spent with the bank, which is the longest time period.

####Balance
"""

histogram_boxplot(df, 'Balance')

"""- 3,500 customers have 0 dollars.
- If you look in the middle, you will see a bell curve.
- As the balance increases after 150K there are less customers who have a bank balance that have 150K+ money.

####Number of Products
"""

labeled_barplot(df, 'NumOfProducts')

"""- There are 5,084 customers who purchased 1 product from the bank.
- Only 60 customers purchased 4 products from the bank.

####Has Credit Card
"""

labeled_barplot(df, 'HasCrCard')

"""- 7,055 customers have credit cards.
- 2,945 customers do not have credit cards.

####Is Active Number
"""

labeled_barplot(df, 'IsActiveMember')

"""- 4,849 customers who are not an active members.
- 5,151 customers who are active members.

####Estimated Salary
"""

histogram_boxplot(df, 'EstimatedSalary')

"""- 50% of the customers have an estimated salary of 100,000 dollars.
- 450+ and less than 500 sustomers have an estimated salary of 200,000 dollars.

####Exited
"""

labeled_barplot(df, 'Exited')

"""- There are 7,963 records who did not leave the bank.
- There are 2,036 records who left the bank.
- There is a huge imbalance in the target variables. We will have to fix it with SMOTE or Under sampling.

### Bivariate Analysis

#### CreditScore vs Gender
"""

plt.title('Credit Score vs Gender')
sns.histplot(data=df, x='CreditScore', hue='Gender', palette='rainbow');

g = sns.FacetGrid(df, col='Gender')
g.map(sns.histplot, 'CreditScore', kde=True);

"""- The credit score for the females is a bit left skewed

####Age vs Gender
"""

sns.histplot(data=df, x='Age', hue='Gender', palette='rainbow');

a = sns.FacetGrid(df, col='Gender')
a.map(sns.histplot, 'Age', kde=True);

"""- The histogram is right skewed.
- Majority of the customer are in their 30-45 age group.

####Age vs Tenure
"""

sns.histplot(data=df, x='Age', hue='Tenure', palette='rainbow');

t = sns.FacetGrid(df, col='Tenure', row='Gender')
t.map(sns.histplot, 'Age', kde=True);

"""- Right skewed.

####EstimatedSalary vs Gender
"""

plt.figure(figsize=(8, 6))
sns.boxplot(x='Gender', y='EstimatedSalary', data=df)
plt.title('EstimatedSalary vs Gender')
plt.show()

sns.histplot(data=df, x='EstimatedSalary', hue='Gender', palette='rainbow');

"""- Females have lower estimated salary.
- The 50% range for the females is lower.

####Balance vs Gender
"""

sns.histplot(data=df, x='Balance', hue='Gender', palette='rainbow');

"""- Almost 2,000 male customers have $0 in their balance.
- The customers who have something in their balance is normally distributed.

####NumOfProducts vs EstimatedSalary
"""

plt.figure(figsize=(8, 6))
sns.boxplot(x='NumOfProducts', y='EstimatedSalary', data=df)
plt.title('NumOfProducts vs EstimatedSalary')
plt.show()

"""####HasCrCard vs EstimatedSalary"""

sns.boxplot(x='HasCrCard', y='EstimatedSalary', data=df)
plt.title('HasCrCard vs EstimatedSalary')
plt.show()

"""####Gender vs NumOfProducts"""

sns.boxplot(x='Gender', y='NumOfProducts', data=df)
plt.title('Gender vs NumOfProducts')
plt.show()

"""####Tenure vs EstimatedSalary"""

sns.boxplot(x='Tenure', y='EstimatedSalary', data=df)
plt.title('Tenure vs EstimatedSalary')
plt.show()

sns.barplot(x='Tenure', y='EstimatedSalary', data=df)
plt.title('Tenure vs EstimatedSalary')
plt.show()

"""- Customers who stayed with the bank for 10 years have a slightly higher estimated salary.

####IsActiveMember vs Balance
"""

sns.histplot(data=df, x='Balance', hue='IsActiveMember', palette='rainbow');

"""- Customers who are not active members have $0 in their balance."""

sns.boxplot(x='IsActiveMember', y='Balance', data=df)
plt.title('IsActiveMember vs Balance')
plt.show();

"""####IsActiveMember vs EstimatedSalary"""

sns.histplot(data=df, x='EstimatedSalary', hue='IsActiveMember', palette='rainbow');

sns.boxplot(x='IsActiveMember', y='EstimatedSalary', data=df)
plt.title('IsActiveMember vs EstimatedSalary')
plt.show();

"""####Balance vs target(Exited)"""

sns.histplot(data=df, x='Balance', hue='Exited');

sns.boxplot(x='Exited', y='Balance', data=df)
plt.title('Exited vs Balance')
plt.show();

"""####EstimatedSalary vs target(Exited)"""

sns.histplot(data=df, x='EstimatedSalary', hue='Exited');

sns.boxplot(x='Exited', y='EstimatedSalary', data=df)
plt.title('Exited vs EstimatedSalary')
plt.show();

"""####Age vs target(Exited)"""

sns.histplot(data=df, x='Age', hue='Exited');

"""- Majority of the customers who did not leave the bank are around 25 - 45 years old.

####Geography vs target(Exited)
"""

sns.histplot(data=df, x='Geography', hue='Exited');

"""####CreditScore vs target(Exited)"""

sns.histplot(data=df, x='CreditScore', hue='Exited');

"""####Gender vs target(Exited)"""

sns.histplot(data=df, x='Gender', hue='Exited');

"""####Tenure vs target(Exited)"""

sns.histplot(data=df, x='Tenure', hue='Exited');

"""####Correlation Check"""

plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(numeric_only = True), annot=True, vmin=-1, vmax=1, fmt=".2f", cmap='coolwarm')
plt.show()

"""## Data Preprocessing

### Outlier Detection and Treatment
"""

numeric_features = df.select_dtypes(include=np.number).columns.tolist()

plt.figure(figsize=(15,12))

for i, variable in enumerate(numeric_features):
    plt.subplot(6, 3, i + 1)
    plt.boxplot(df[variable], whis=1.5)
    plt.tight_layout()
    plt.title(variable)

plt.show()

"""- The outliers are present but they don't have to be treated.
- They are part of the data.

### Data Preparation for model building
"""

X = df.drop(['Exited'], axis=1)
y = df['Exited']

X

"""### Splitting Training, Validation, and Testing Datasets to prevent data leaks"""

X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=1,
    stratify=y, #ensures that the class distribution in the test set matches the original dataset
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,
    random_state=1,
    stratify=y_temp,
)

print(f"X_train shape: {X_train.shape},{y_train.shape}")
print(f"X_val shape: {X_val.shape},{y_val.shape}")
print(f"X_test shape: {X_test.shape},{y_test.shape}")

y.value_counts(1)

y_train.value_counts(1)

y_val.value_counts(1)

y_test.value_counts(1)

"""### Encoding categorical variables"""

categorical_features = X_train.select_dtypes(include='category').columns.tolist() #categorical features
encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore') #initializing encoder, i am trying OneHotEncoder() instead of .get_dummies()
encoder.fit(X_train[categorical_features]) #fit encoder on train data
encoded_columns = list(encoder.get_feature_names_out(categorical_features)) #get new features after encoding

# Transform categorical features into a DataFrame
X_train_encoded = pd.DataFrame(encoder.transform(X_train[categorical_features]), columns=encoded_columns, index=X_train.index)
X_val_encoded = pd.DataFrame(encoder.transform(X_val[categorical_features]), columns=encoded_columns, index=X_val.index)
X_test_encoded = pd.DataFrame(encoder.transform(X_test[categorical_features]), columns=encoded_columns, index=X_test.index)

# Drop original categorical columns
X_train = X_train.drop(columns=categorical_features)
X_val = X_val.drop(columns=categorical_features)
X_test = X_test.drop(columns=categorical_features)

# Concatenate with the original DataFrame
X_train = pd.concat([X_train, X_train_encoded], axis=1)
X_val = pd.concat([X_val, X_val_encoded], axis=1)
X_test = pd.concat([X_test, X_test_encoded], axis=1)

X_train

X_val

X_test

"""### Data Normalization"""

transformer = StandardScaler()

X_train[['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']] = transformer.fit_transform(X_train[['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']])
X_val[['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']] = transformer.transform(X_val[['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']])
X_test[['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']] = transformer.transform(X_test[['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']])

X_train

X_val

X_test

# Convert target variable to numerical
y_train = y_train.astype(int)
y_val = y_val.astype(int)
y_test = y_test.astype(int)

"""## Model Building

### Model Evaluation Criterion

Write down the logic for choosing the metric that would be the best metric for this business scenario:

The nature of predictions made by the classification model will translate as follows:

- True positives (TP) are failures correctly predicted by the model.
- False negatives (FN) are real failures in a generator where there is no detection by model.
- False positives (FP) are failure detections in a generator where there is no failure.

**Which metric to optimize?**

* We need to choose the metric which will ensure that the maximum number of generator failures are predicted correctly by the model.
* We would want `Recall` to be maximized as greater the `Recall`, the higher the chances of minimizing false negatives.
* We want to minimize false negatives because if a model predicts that a machine will have no failure when there will be a failure, it will increase the maintenance cost.
"""

def plot(history, name):
    """
    Function to plot loss/accuracy

    history: an object which stores the metrics and losses.
    name: can be one of Loss or Accuracy
    """
    fig, ax = plt.subplots() #Creating a subplot with figure and axes.
    plt.plot(history.history[name]) #Plotting the train accuracy or train loss
    plt.plot(history.history['val_'+name]) #Plotting the validation accuracy or validation loss

    plt.title('Model ' + name.capitalize()) #Defining the title of the plot.
    plt.ylabel(name.capitalize()) #Capitalizing the first letter.
    plt.xlabel('Epoch') #Defining the label for the x-axis.
    fig.legend(['Train', 'Validation'], loc="outside right upper") #Defining the legend, loc controls the position of the legend.

# defining a function to compute different metrics to check performance of a classification model built using statsmodels
def model_performance_classification(
    model, predictors, target, threshold=0.5
):
    """
    Function to compute different metrics to check classification model performance

    model: classifier
    predictors: independent variables
    target: dependent variable
    threshold: threshold for classifying the observation as class 1
    """

    # checking which probabilities are greater than threshold
    pred = model.predict(predictors) > threshold
    # pred_temp = model.predict(predictors) > threshold
    # # rounding off the above values to get classes
    # pred = np.round(pred_temp)

    acc = accuracy_score(target, pred)  # to compute Accuracy
    recall = recall_score(target, pred, average='weighted')  # to compute Recall
    precision = precision_score(target, pred, average='weighted')  # to compute Precision
    f1 = f1_score(target, pred, average='weighted')  # to compute F1-score

    # creating a dataframe of metrics
    df_perf = pd.DataFrame(
        {"Accuracy": acc, "Recall": recall, "Precision": precision, "F1 Score": f1,},
        index=[0],
    )

    return df_perf

def confusion_matrix_sklearn(model, predictors, target):
    """
    To plot the confusion_matrix with percentages

    model: classifier
    predictors: independent variables
    target: dependent variable
    """
    y_pred = model.predict(predictors)
    # Convert predicted probabilities to class labels (0 or 1)
    y_pred = (y_pred > 0.5).astype(int) # added this line
    cm = confusion_matrix(target, y_pred)
    labels = np.asarray(
        [
            ["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / cm.flatten().sum())]
            for item in cm.flatten()
        ]
    ).reshape(2, 2)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=labels, fmt="")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

#Defining the columns of the dataframe which are nothing but the hyper parameters and the metrics.
columns = ["# hidden layers","# neurons - hidden layer","activation function - hidden layer ","# epochs","batch size","optimizer","learning rate, momentum","weight initializer","regularization","train loss","validation loss","train accuracy","validation accuracy", "time (secs)"]

#Creating a pandas dataframe.
results = pd.DataFrame(columns=columns)

"""### Model 0 - Neural Network with SGD Optimizer"""

tf.keras.backend.clear_session() #clears current session, resetting all layers and model, freeing up memory

"""- Fully connected layers"""

model0 = Sequential()

model0.add(Dense(128, activation="relu", input_dim=X_train.shape[1])) #input & hidden layer(128)
model0.add(Dense(256, activation="relu")) #hidden layer(256)
model0.add(Dense(1, activation="sigmoid")) #output layer(1 neuron)

model0.summary()

"""**How did I get those parameters?**

```
1st layer (input/hidden layer):
(X_train(13 features/columns) * 128 neurons) + 128 neurons = 1,792 parameters
```

```
2nd layer (hidden layer):
(128 neurons(from previous layer) * 256 neurons) + 256 neurons = 33,024 parameters
```

```
3rd layer (output layer):
(256 neurons(from previous layer) * 1 neuron) + 1 neuron = 257 parameters
```

```
Total Parameters:
1,792 parameters + 33,024 parameters + 257 parameters = 35,073 parameters
```
"""

optimizer = tf.keras.optimizers.SGD() #SGD optimizer
model0.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

batch_size = 64 #batch size
epochs = 10 #epochs

start = time.time() #starting time

history = model0.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    epochs=epochs
)

end = time.time() #ending time

print("Time taken in seconds ",end-start)

plot(history, "loss")

plot(history, "accuracy")

results.loc[0] = [2,[128,256],['relu','relu'],10,64,optimizer,"-","-","-",history.history["loss"][-1],history.history["val_loss"][-1],history.history["accuracy"][-1],history.history["val_accuracy"][-1],round(end-start,2)]

results

model_0_train_perf = model_performance_classification(model0, X_train, y_train)
model_0_train_perf

confusion_matrix_sklearn(model0, X_train, y_train)

model_0_val_perf = model_performance_classification(model0, X_val, y_val)
model_0_val_perf

confusion_matrix_sklearn(model0, X_val, y_val)

"""- This model has 2 hidden layers.
- Both the hidden layers use relu as activation function.
- I will try increasing the number of epochs to see if the the loss goes down and recall, accuracy improves.

### Model 1 - Neural Network with SGD Optimizer (continued)
"""

tf.keras.backend.clear_session() #clears current session, resetting all layers and model, freeing up memory

model1 = Sequential()

model1.add(Dense(128, activation="relu", input_dim=X_train.shape[1])) #input & hidden layer(128)
model1.add(Dense(256, activation="relu")) #hidden layer(256)
model1.add(Dense(1, activation="sigmoid")) #output layer(1 neuron)

model1.summary()

optimizer = tf.keras.optimizers.SGD() #SGD optimizer
model1.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

batch_size = 64 #batch size
epochs = 50 #epochs

start = time.time() #starting time

history = model1.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    epochs=epochs
)

end = time.time() #ending time

print("Time taken in seconds ",end-start)

plot(history, "loss")

plot(history, "accuracy")

results.loc[1] = [2,[128,256],['relu','relu'],50,64,optimizer,"-","-","-",history.history["loss"][-1],history.history["val_loss"][-1],history.history["accuracy"][-1],history.history["val_accuracy"][-1],round(end-start,2)]

results

model_1_train_perf = model_performance_classification(model1, X_train, y_train)
model_1_train_perf

confusion_matrix_sklearn(model1, X_train, y_train)

model_1_val_perf = model_performance_classification(model1, X_val, y_val)
model_1_val_perf

confusion_matrix_sklearn(model1, X_val, y_val)

"""- This model is slightly better than model0.
- However, it can still improve it's performance.
- Next, i'll increase only the number of epochs again to see if the metrics improve.
- The model is slightly overfitting to the training data.

### Model 2 - Neural Network with SGD Optimizer (continued)
"""

tf.keras.backend.clear_session() #clears current session, resetting all layers and model, freeing up memory

model2 = Sequential()

model2.add(Dense(128, activation="relu", input_dim=X_train.shape[1])) #input & hidden layer(128)
model2.add(Dense(256, activation="relu")) #hidden layer(256)
model2.add(Dense(1, activation="sigmoid")) #output layer(1 neuron)

model2.summary()

optimizer = tf.keras.optimizers.SGD() #SGD optimizer
model2.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

batch_size = 64 #batch size
epochs = 100 #epochs

start = time.time() #starting time

history = model2.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    epochs=epochs
)

end = time.time() #ending time

print("Time taken in seconds ",end-start)

plot(history, "loss")

plot(history, "accuracy")

results.loc[2] = [2,[128,256],['relu','relu'],100,64,optimizer,"-","-","-",history.history["loss"][-1],history.history["val_loss"][-1],history.history["accuracy"][-1],history.history["val_accuracy"][-1],round(end-start,2)]

results

model_2_train_perf = model_performance_classification(model2, X_train, y_train)
model_2_train_perf

confusion_matrix_sklearn(model2, X_train, y_train)

model_2_val_perf = model_performance_classification(model2, X_val, y_val)
model_2_val_perf

confusion_matrix_sklearn(model2, X_val, y_val)

"""- Hmmm, the metrics did not change that much and the loss did not decrease that much either.
- I will try to use tanh for the 2nd hidden layer.
- The model is slightly overfitting to the training data.

### Model 3 - Neural Network with SGD Optimizer (continued)
"""

tf.keras.backend.clear_session() #clears current session, resetting all layers and model, freeing up memory

model3 = Sequential()

model3.add(Dense(128, activation="relu", input_dim=X_train.shape[1])) #input & hidden layer(128)
model3.add(Dense(256, activation="tanh")) #hidden layer(256)
model3.add(Dense(1, activation="sigmoid")) #output layer(1 neuron)

model3.summary()

optimizer = tf.keras.optimizers.SGD() #SGD optimizer
model3.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

batch_size = 64 #batch size
epochs = 100 #epochs

start = time.time() #starting time

history = model3.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    epochs=epochs
)

end = time.time() #ending time

print("Time taken in seconds ",end-start)

plot(history, "loss")

plot(history, "accuracy")

results.loc[3] = [2,[128,256],['relu','tanh'],100,64,optimizer,"-","-","-",history.history["loss"][-1],history.history["val_loss"][-1],history.history["accuracy"][-1],history.history["val_accuracy"][-1],round(end-start,2)]

results

model_3_train_perf = model_performance_classification(model3, X_train, y_train)
model_3_train_perf

confusion_matrix_sklearn(model3, X_train, y_train)

model_3_val_perf = model_performance_classification(model3, X_val, y_val)
model_3_val_perf

confusion_matrix_sklearn(model3, X_val, y_val)

"""- The performance did not improve at all.
- So I will use different hyperparameters to try to increase the accuracy, recall metrics and also decrease the loss.
- I will now use Adam optimizer instead of SGD optimizer.

### Model 4 - Neural Network with Adam Optimizer
"""

tf.keras.backend.clear_session() #clears current session, resetting all layers and model, freeing up memory

model4 = Sequential()

model4.add(Dense(128, activation="relu", input_dim=X_train.shape[1])) #input & hidden layer(128)
model4.add(Dense(256, activation="tanh")) #hidden layer(256)
model4.add(Dense(1, activation="sigmoid")) #output layer(1 neuron)

model4.summary()

optimizer = tf.keras.optimizers.Adam() #Adam optimizer
model4.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

batch_size = 64 #batch size
epochs = 100 #epochs

start = time.time() #starting time

history = model4.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    epochs=epochs
)

end = time.time() #ending time

print("Time taken in seconds ",end-start)

plot(history, "loss")

plot(history, "accuracy")

results.loc[4] = [2,[128,256],['relu','tanh'],100,64,optimizer,"-","-","-",history.history["loss"][-1],history.history["val_loss"][-1],history.history["accuracy"][-1],history.history["val_accuracy"][-1],round(end-start,2)]

results

model_4_train_perf = model_performance_classification(model4, X_train, y_train)
model_4_train_perf

confusion_matrix_sklearn(model4, X_train, y_train)

model_4_val_perf = model_performance_classification(model4, X_val, y_val)
model_4_val_perf

confusion_matrix_sklearn(model4, X_val, y_val)

"""- The training loss decreased a little but the accuracy validation is not good.
- The model is overfitting to the training data.
- Next, I will try to use dropout technique.

### Model 5 - Neural Network with Adam Optimizer and Dropout
"""

tf.keras.backend.clear_session() #clears current session, resetting all layers and model, freeing up memory

model5 = Sequential()

model5.add(Dense(128, activation="relu", input_dim=X_train.shape[1])) #input & hidden layer(128)
model5.add(Dropout(0.4)) #dropout layer
model5.add(Dense(256, activation="tanh")) #hidden layer(256)
model5.add(Dropout(0.4)) #dropout layer
model5.add(Dense(1, activation="sigmoid")) #output layer(1 neuron)

model5.summary()

optimizer = tf.keras.optimizers.Adam() #Adam optimizer
model5.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

batch_size = 64 #batch size
epochs = 100 #epochs

start = time.time() #starting time

history = model5.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    epochs=epochs
)

end = time.time() #ending time

print("Time taken in seconds ",end-start)

plot(history, "loss")

plot(history, "accuracy")

results.loc[5] = [4,[128,256],['relu','tanh'],100,64,optimizer,"-","-","-",history.history["loss"][-1],history.history["val_loss"][-1],history.history["accuracy"][-1],history.history["val_accuracy"][-1],round(end-start,2)]

results

model_5_train_perf = model_performance_classification(model5, X_train, y_train)
model_5_train_perf

confusion_matrix_sklearn(model5, X_train, y_train)

model_5_val_perf = model_performance_classification(model5, X_val, y_val)
model_5_val_perf

confusion_matrix_sklearn(model5, X_val, y_val)

"""- The model is not good.
- The best model right now out of the all the previous ones is model 4.
- I tried the dropout but it did not improve.
- I will now try the learning rate and momentum.

### Model 6 - Neural Network with Adam Optimizer (continued)
"""

tf.keras.backend.clear_session() #clears current session, resetting all layers and model, freeing up memory

model6 = Sequential()

model6.add(Dense(128, activation="relu", input_dim=X_train.shape[1])) #input & hidden layer(128)
model6.add(Dense(256, activation="tanh")) #hidden layer(256)
model6.add(Dense(1, activation="sigmoid")) #output layer(1 neuron)

model6.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01) #Adam optimizer
model6.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

batch_size = 64 #batch size
epochs = 100 #epochs

start = time.time() #starting time

history = model6.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    epochs=epochs
)

end = time.time() #ending time

print("Time taken in seconds ",end-start)

plot(history, "loss")

plot(history, "accuracy")

results.loc[6] = [4,[128,256],['relu','tanh'],100,64,optimizer,[0.01,"-"],"-","-",history.history["loss"][-1],history.history["val_loss"][-1],history.history["accuracy"][-1],history.history["val_accuracy"][-1],round(end-start,2)]

results

model_6_train_perf = model_performance_classification(model6, X_train, y_train)
model_6_train_perf

confusion_matrix_sklearn(model6, X_train, y_train)

model_6_val_perf = model_performance_classification(model6, X_val, y_val)
model_6_val_perf

confusion_matrix_sklearn(model6, X_val, y_val)

"""- The train loss decreased quite a bit, but the validation loss is increasing.
- This model is overfitting to the training data.
- I will remove the dropout layer and try to add the batch normalization layers.
- I will use tanh for the first hidden layer and relu for the second hidden layer.

### Model 7 - Neural Network with Adam Optimizer (continued)
"""

tf.keras.backend.clear_session() #clears current session, resetting all layers and model, freeing up memory

model7 = Sequential()

model7.add(Dense(128, activation="relu", input_dim=X_train.shape[1], kernel_regularizer=l2(0.001))) #input & hidden layer(128)
model7.add(BatchNormalization()) #batch normalization layer
model7.add(Dropout(0.4)) #dropout layer
model7.add(Dense(256, activation="tanh", kernel_regularizer=l2(0.001))) #hidden layer(256)
model7.add(BatchNormalization()) #batch normalization layer
model7.add(Dropout(0.4)) #dropout layer
model7.add(Dense(1, activation="sigmoid")) #output layer(1 neuron)

model7.summary()

optimizer = tf.keras.optimizers.Adam() #Adam optimizer
model7.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

batch_size = 64 #batch size
epochs = 100 #epochs

start = time.time() #starting time

history = model7.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    epochs=epochs
)

end = time.time() #ending time

print("Time taken in seconds ",end-start)

plot(history, "loss")

plot(history, "accuracy")

results.loc[7] = [6,[128,256],['tanh','relu'],100,64,optimizer,"-","-","-",history.history["loss"][-1],history.history["val_loss"][-1],history.history["accuracy"][-1],history.history["val_accuracy"][-1],round(end-start,2)]

results

model_7_train_perf = model_performance_classification(model7, X_train, y_train)
model_7_train_perf

confusion_matrix_sklearn(model7, X_train, y_train)

model_7_val_perf = model_performance_classification(model7, X_val, y_val)
model_7_val_perf

confusion_matrix_sklearn(model7, X_val, y_val)

"""- The performance decreased slightly.
- I will now try to use the he weight initialization.

### Model 8 - Neural Network with Adam Optimizer (continued)
"""

tf.keras.backend.clear_session() #clears current session, resetting all layers and model, freeing up memory

model8 = Sequential()

model8.add(Dense(128, activation="relu", input_dim=X_train.shape[1], kernel_initializer="he_normal", kernel_regularizer=l2(0.001))) #input & hidden layer(128), Xavier initialization
model8.add(BatchNormalization()) #batch normalization layer
model8.add(Dropout(0.4)) #dropout layer
model8.add(Dense(256, activation="relu", kernel_initializer="he_normal", kernel_regularizer=l2(0.001))) #hidden layer(256), Xavier initialization
model8.add(BatchNormalization()) #batch normalization layer
model8.add(Dropout(0.4)) #dropout layer
model8.add(Dense(1, activation="sigmoid")) #output layer(1 neuron)

model8.summary()

optimizer = tf.keras.optimizers.Adam() #Adam optimizer
model8.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

batch_size = 64 #batch size
epochs = 100 #epochs

start = time.time() #starting time

history = model8.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    epochs=epochs
)

end = time.time() #ending time

print("Time taken in seconds ",end-start)

plot(history, "loss")

plot(history, "accuracy")

results.loc[8] = [4,[128,256],['relu','tanh'],100,64,optimizer,"-","he_normal","-",history.history["loss"][-1],history.history["val_loss"][-1],history.history["accuracy"][-1],history.history["val_accuracy"][-1],round(end-start,2)]

results

model_8_train_perf = model_performance_classification(model8, X_train, y_train)
model_8_train_perf

confusion_matrix_sklearn(model8, X_train, y_train)

model_8_val_perf = model_performance_classification(model8, X_val, y_val)
model_8_val_perf

confusion_matrix_sklearn(model8, X_val, y_val)

"""- Did not improve at all.
- I will try to use SGD optimizer with learning rate of 0.01

### Model 9 - Neural Network with SGD Optimizer (continued)
"""

tf.keras.backend.clear_session() #clears current session, resetting all layers and model, freeing up memory

model9 = Sequential()

model9.add(Dense(128, activation="relu", input_dim=X_train.shape[1])) #input & hidden layer(128)
model9.add(Dense(256, activation="relu")) #hidden layer(256)
model9.add(Dense(1, activation="sigmoid")) #output layer(1 neuron)

model9.summary()

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01) #SGD optimizer with learning rate of 0.01 (higher than default)
model9.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

batch_size = 64 #batch size
epochs = 100 #epochs

start = time.time() #starting time

history = model9.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    epochs=epochs
)

end = time.time() #ending time

print("Time taken in seconds ",end-start)

plot(history, "loss")

plot(history, "accuracy")

results.loc[9] = [2,[128,256],['relu','relu'],100,64,optimizer,[0.01,"-"],"-","-",history.history["loss"][-1],history.history["val_loss"][-1],history.history["accuracy"][-1],history.history["val_accuracy"][-1],round(end-start,2)]

results

model_9_train_perf = model_performance_classification(model9, X_train, y_train)
model_9_train_perf

confusion_matrix_sklearn(model9, X_train, y_train)

model_9_val_perf = model_performance_classification(model9, X_val, y_val)
model_9_val_perf

confusion_matrix_sklearn(model9, X_val, y_val)

"""- Did not improve from the model 3 which I ran previously.
- I will now try to add another hidden layer still with just SGD optimizer.

### Model 10 - Neural Network with SGD Optimizer (continued)
"""

tf.keras.backend.clear_session() #clears current session, resetting all layers and model, freeing up memory

model10 = Sequential()

model10.add(Dense(128, activation="relu", input_dim=X_train.shape[1])) #input & hidden layer(128)
model10.add(Dense(256, activation="relu")) #hidden layer(256)
model10.add(Dense(512, activation="relu")) #hidden layer(512)
model10.add(Dense(1, activation="sigmoid")) #output layer(1 neuron)

model10.summary()

optimizer = tf.keras.optimizers.SGD() #Adam optimizer
model10.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

batch_size = 64 #batch size
epochs = 100 #epochs

start = time.time() #starting time

history = model10.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    epochs=epochs
)

end = time.time() #ending time

print("Time taken in seconds ",end-start)

plot(history, "loss")

plot(history, "accuracy")

results.loc[10] = [3,[128,256,512],['relu','relu','relu'],100,64,optimizer,"-","-","-",history.history["loss"][-1],history.history["val_loss"][-1],history.history["accuracy"][-1],history.history["val_accuracy"][-1],round(end-start,2)]

results

model_10_train_perf = model_performance_classification(model10, X_train, y_train)
model_10_train_perf

confusion_matrix_sklearn(model10, X_train, y_train)

model_10_val_perf = model_performance_classification(model10, X_val, y_val)
model_10_val_perf

confusion_matrix_sklearn(model10, X_val, y_val)

"""- Did not improve the metrics.
- Still overfitting.
- How can I improve the accuracy as well as decreasing the loss?
  - I can try to add a new hidden layer to the latest improved model for Adam optimizer.
  - I will also add Batch Normalization and Dropout layer.

### Model 11 - Neural Network with Adam Optimizer (continued)
"""

tf.keras.backend.clear_session() #clears current session, resetting all layers and model, freeing up memory

model11 = Sequential()

model11.add(Dense(128, activation="relu", input_dim=X_train.shape[1])) #input & hidden layer(128)
model11.add(BatchNormalization()) #batch normalization layer
model11.add(Dropout(0.4)) #dropout
model11.add(Dense(256, activation="tanh")) #hidden layer(256)
model11.add(BatchNormalization()) #batch normalization layer
model11.add(Dropout(0.4)) #dropout
model11.add(Dense(512, activation="relu")) #hidden layer(512)
model11.add(BatchNormalization()) #batch normalization layer
model11.add(Dropout(0.4)) #dropout
model11.add(Dense(1, activation="sigmoid")) #output layer(1 neuron)

model11.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.005) #Adam optimizer
model11.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

batch_size = 64 #batch size
epochs = 100 #epochs

start = time.time() #starting time

history = model11.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    epochs=epochs
)

end = time.time() #ending time

print("Time taken in seconds ",end-start)

plot(history, "loss")

plot(history, "accuracy")

results.loc[11] = [9,[128,256,512],['relu','tanh','relu'],100,64,optimizer,["0.005","-"],"-","-",history.history["loss"][-1],history.history["val_loss"][-1],history.history["accuracy"][-1],history.history["val_accuracy"][-1],round(end-start,2)]

results

model_11_train_perf = model_performance_classification(model11, X_train, y_train)
model_11_train_perf

confusion_matrix_sklearn(model11, X_train, y_train)

model_11_val_perf = model_performance_classification(model11, X_val, y_val)
model_11_val_perf

confusion_matrix_sklearn(model11, X_val, y_val)

"""- The metrics did not improve much.
- I will now use regularization techniques to prevent overfitting.

### Model 12 - Neural Network with Balanced Data (by applying SMOTE) and SGD Optimizer
"""

tf.keras.backend.clear_session() #clears current session, resetting all layers and model, freeing up memory

sm = SMOTE(sampling_strategy=1, k_neighbors=5, random_state=1)
X_train_over, y_train_over = sm.fit_resample(X_train, y_train)

model12 = Sequential()

model12.add(Dense(128, activation="relu", input_dim=X_train_over.shape[1])) #input & hidden layer(128)
model12.add(BatchNormalization()) #batch normalization layer
model12.add(Dropout(0.3)) #dropout
model12.add(Dense(256, activation="sigmoid")) #hidden layer(256)
model12.add(BatchNormalization()) #batch normalization layer
model12.add(Dropout(0.3)) #dropout
model12.add(Dense(512, activation="relu")) #hidden layer(512)
model12.add(BatchNormalization()) #batch normalization layer
model12.add(Dense(1, activation="sigmoid")) #output layer(1 neuron)

model12.summary()

optimizer = tf.keras.optimizers.SGD() #Adam optimizer
model12.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

batch_size = 64 #batch size
epochs = 100 #epochs

start = time.time() #starting time

history = model12.fit(
    X_train_over,
    y_train_over,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    epochs=epochs
)

end = time.time() #ending time

print("Time taken in seconds ",end-start)

plot(history, "loss")

plot(history, "accuracy")

results.loc[12] = [8,[128,256,512],['relu','sigmoid','relu'],100,64,optimizer,"-","-","-",history.history["loss"][-1],history.history["val_loss"][-1],history.history["accuracy"][-1],history.history["val_accuracy"][-1],round(end-start,2)]

results

model_12_train_over_perf = model_performance_classification(model12, X_train_over, y_train_over)
model_12_train_over_perf

confusion_matrix_sklearn(model12, X_train_over, y_train_over)

model_12_val_perf = model_performance_classification(model12, X_val, y_val)
model_12_val_perf

confusion_matrix_sklearn(model12, X_val, y_val)

"""### Model 13 - Neural Network with Balanced Data (by applying SMOTE) and Adam Optimizer"""

tf.keras.backend.clear_session() #clears current session, resetting all layers and model, freeing up memory

sm = SMOTE(sampling_strategy=1, k_neighbors=5, random_state=1)
X_train_over, y_train_over = sm.fit_resample(X_train, y_train)

model13 = Sequential()

model13.add(Dense(128, activation="relu", input_dim=X_train_over.shape[1])) #input & hidden layer(128)
model13.add(Dense(256, activation="tanh")) #hidden layer(256)
model13.add(Dense(1, activation="sigmoid")) #output layer(1 neuron)

model13.summary()

optimizer = tf.keras.optimizers.Adam() #Adam optimizer
model13.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

batch_size = 64 #batch size
epochs = 100 #epochs

start = time.time() #starting time

history = model13.fit(
    X_train_over,
    y_train_over,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    epochs=epochs
)

end = time.time() #ending time

print("Time taken in seconds ",end-start)

plot(history, "loss")

plot(history, "accuracy")

results.loc[13] = [2,[128,256],['relu','tanh'],100,64,optimizer,"-","-","-",history.history["loss"][-1],history.history["val_loss"][-1],history.history["accuracy"][-1],history.history["val_accuracy"][-1],round(end-start,2)]

results

model_13_train_over_perf = model_performance_classification(model13, X_train_over, y_train_over)
model_13_train_over_perf

confusion_matrix_sklearn(model13, X_train_over, y_train_over)

model_13_val_perf = model_performance_classification(model13, X_val, y_val)
model_13_val_perf

confusion_matrix_sklearn(model13, X_val, y_val)

"""### Model 14 - Neural Network with Balanced Data (by applying SMOTE), Adam Optimizer, and Dropout"""

tf.keras.backend.clear_session() #clears current session, resetting all layers and model, freeing up memory

sm = SMOTE(sampling_strategy=1, k_neighbors=5, random_state=1)
X_train_over, y_train_over = sm.fit_resample(X_train, y_train)

model14 = Sequential()

model14.add(Dense(128, activation="relu", input_dim=X_train_over.shape[1])) #input & hidden layer(17)
model14.add(BatchNormalization()) #batch normalization layer
model14.add(Dropout(0.4)) #dropout
model14.add(Dense(256, activation="relu")) #hidden layer(8)
model14.add(Dense(1, activation="sigmoid")) #output layer(1 neuron)

model14.summary()

optimizer = tf.keras.optimizers.Adam() #Adam optimizer
model14.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

batch_size = 64 #batch size
epochs = 100 #epochs

start = time.time() #starting time

history = model14.fit(
    X_train_over,
    y_train_over,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    epochs=epochs
)

end = time.time() #ending time

print("Time taken in seconds ",end-start)

plot(history, "loss")

plot(history, "accuracy")

results.loc[14] = [4,[128,256],['relu','relu'],100,64,optimizer,"-","-","-",history.history["loss"][-1],history.history["val_loss"][-1],history.history["accuracy"][-1],history.history["val_accuracy"][-1],round(end-start,2)]

results

model_14_train_over_perf = model_performance_classification(model14, X_train_over, y_train_over)
model_14_train_over_perf

confusion_matrix_sklearn(model14, X_train_over, y_train_over)

model_14_val_perf = model_performance_classification(model14, X_val, y_val)
model_14_val_perf

confusion_matrix_sklearn(model14, X_val, y_val)

"""## Model Performance Comparison and Final Model Selection"""

models_train_comp_df = pd.concat([
    model_0_train_perf.T,
    model_1_train_perf.T,
    model_2_train_perf.T,
    model_3_train_perf.T,
    model_4_train_perf.T,
    model_5_train_perf.T,
    model_6_train_perf.T,
    model_7_train_perf.T,
    model_8_train_perf.T,
    model_9_train_perf.T,
    model_10_train_perf.T,
    model_11_train_perf.T,
    model_12_train_over_perf.T,
    model_13_train_over_perf.T,
    model_14_train_over_perf.T
], axis=1)

models_train_comp_df.columns = [
    "Model 0",
    "Model 1",
    "Model 2",
    "Model 3",
    "Model 4",
    "Model 5",
    "Model 6",
    "Model 7",
    "Model 8",
    "Model 9",
    "Model 10",
    "Model 11",
    "Model 12",
    "Model 13",
    "Model 14"
]

print("Training performance comparison:")
models_train_comp_df

models_validation_comp_df = pd.concat([
    model_0_val_perf.T,
    model_1_val_perf.T,
    model_2_val_perf.T,
    model_3_val_perf.T,
    model_4_val_perf.T,
    model_5_val_perf.T,
    model_6_val_perf.T,
    model_7_val_perf.T,
    model_8_val_perf.T,
    model_9_val_perf.T,
    model_10_val_perf.T,
    model_11_val_perf.T,
    model_12_val_perf.T,
    model_13_val_perf.T,
    model_14_val_perf.T
], axis=1)

models_validation_comp_df.columns = [
    "Model 0",
    "Model 1",
    "Model 2",
    "Model 3",
    "Model 4",
    "Model 5",
    "Model 6",
    "Model 7",
    "Model 8",
    "Model 9",
    "Model 10",
    "Model 11",
    "Model 12",
    "Model 13",
    "Model 14"
]

print("Testing performance comparison:")
models_validation_comp_df

"""- Models 12 - 14 are not a good choice for the final selection. Those models are clearly highly overfitted to the training data.
- Models 1, 3, 9 are good models to select. Even though, they are slightly overfitting it is a good choice.

- Model 9 is my final selection for picking the model.
- It has pretty good metrics.

###Test Set Final Performance
"""

model_9_test_perf = model_performance_classification(model9, X_test, y_test)
model_9_test_perf

confusion_matrix_sklearn(model9, X_test, y_test)

"""- Model 9 has a 86% on the recall test set.

## Actionable Insights and Business Recommendations

- The credit score is a bit left skewed.
- Atleast 300 records have the highest credit scores.
- 50% of credit scores is between 600 and 700.
- If you look closely you can see there is a bell shaped curve, if you ignore the few records to the left.
- The majority of the customers are from France.
- Germany and Spain are almost equal to each other.
- The majority of the customers are male.
- Females are not so bad they are almost catching up to the males.
- There is an outlier to the far right that is the highest age of the customer.
- The age is right skewed, which means there are a lot of customers who are young or middle aged.
- There are 800+ records where the age is around 38-41 year olds, this is the highest age group.
- The second highest age group is 25-29 year olds.
- 413 of the customers spend 0 years with the bank. I think they recently joined the bank or they left before the first year.
- 10 years was how long they spent with the bank, which is the longest time period.
- 3,500 customers have 0 dollars.
- If you look in the middle, you will see a bell curve.
- As the balance increases after 150K there are less customers who have a bank balance that have 150K+ money.
- There are 5,084 customers who purchased 1 product from the bank.
- Only 60 customers purchased 4 products from the bank.
- 7,055 customers have credit cards.
- 2,945 customers do not have credit cards.
- 4,849 customers who are not an active members.
- 5,151 customers who are active members.
- 50% of the customers have an estimated salary of 100,000 dollars.
- 450+ and less than 500 sustomers have an estimated salary of 200,000 dollars.
- There are 7,963 records who did not leave the bank.
- There are 2,036 records who left the bank.
- There is a huge imbalance in the target variables. We will have to fix it with SMOTE or Under sampling.
- Model 9 is my final selection for picking the model.
- It has pretty good metrics.
- Focus more on the younger population rather than 60+.
- Younger customers are more likely to stay with a bank since they might have some responsibilities in their lives.

<font size=6 color='blue'>Power Ahead</font>
___
"""

#convert to html
!jupyter nbconvert --to html /content/INN_Learner_Notebook_Full_code.ipynb