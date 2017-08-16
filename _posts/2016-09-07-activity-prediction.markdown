---
title:  "Predicting Human Activity from Smartphone Accelerometer and Gyroscope Data"
date:   2016-09-07
tags: [machine learning]

header:
  image: "smartphone_accelerometer.jpg"
  caption: "Photo credit: [**MathWorks**](http://www.mathworks.com/hardware-support/android-sensor.html)"

excerpt: "Human Activity Classification, Accelerometers, Support Vector Machines"
---

How does my Fitbit track my steps? I always assumed it was pretty accurate, but I never actually knew how it worked.

So I googled it.

My Fitbit uses a 3-axial accelerometer to track my motion, according to the company's [website](https://help.fitbit.com/articles/en_US/Help_article/1143). The tracker then uses an algorithm to determine whether I'm walking or running (as opposed to standing still or driving in a car).

Cool. But how does it actually do that? And how good is it?

Since Fitbit probably won't release its code anytime soon, I decided to find out how well I could design an algorithm to differentiate walking from staying still (as well as from other activities). After a quick google search, I found a dataset of 3-axial accelerometer signals from an academic experiment on the UC Irvine Machine Learning Repository. The data originally come from [SmartLab](www.smartlab.ws) at the University of Genova. You can download the data [here](https://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions).

# Loading the Accelerometer and Gyroscope Data

The data are pre-split into training and test sets, so we'll read them in separately. To get the feature names and the activity labels, we have to read separate files too.

```python
import pandas as pd
from scipy.spatial.distance import cosine
import numpy as np
from scipy.sparse import csr_matrix
from __future__ import division
import matplotlib.pyplot as plt
%matplotlib inline

# display pandas results to 3 decimal points, not in scientific notation
pd.set_option('display.float_format', lambda x: '%.3f' % x)
```

```python
with open('/Users/nickbecker/Downloads/HAPT Data Set/features.txt') as handle:
    features = handle.readlines()
    features = map(lambda x: x.strip(), features)

with open('/Users/nickbecker/Downloads/HAPT Data Set/activity_labels.txt') as handle:
    activity_labels = handle.readlines()
    activity_labels = map(lambda x: x.strip(), activity_labels)

activity_df = pd.DataFrame(activity_labels)
activity_df = pd.DataFrame(activity_df[0].str.split(' ').tolist(),
                           columns = ['activity_id', 'activity_label'])
activity_df
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>activity_id</th>
      <th>activity_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>WALKING</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>WALKING_UPSTAIRS</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>WALKING_DOWNSTAIRS</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>SITTING</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>STANDING</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>LAYING</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>STAND_TO_SIT</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>SIT_TO_STAND</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>SIT_TO_LIE</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>LIE_TO_SIT</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>STAND_TO_LIE</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>LIE_TO_STAND</td>
    </tr>
  </tbody>
</table>
</div>

So we have 12 activities, ranging from sitting down to walking up the stairs. The SmartLab researchers created 561 features from 17 3-axial accelerometer and gyroscope signals from the smartphone. These features capture descriptive statistics and moments of the 17 signal distributions (mean, standard deviation, max, min, skewness, etc.). They also did some reasonable pre-processing to de-noise and filter the sensor data (you can read about it [here](https://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions) -- it's pretty reasonable). Let's look at a few features.


```python
x_train = pd.read_table('/Users/nickbecker/Downloads/HAPT Data Set/Train/X_train.txt',
             header = None, sep = " ", names = features)
x_train.iloc[:10, :10].head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tBodyAcc-Mean-1</th>
      <th>tBodyAcc-Mean-2</th>
      <th>tBodyAcc-Mean-3</th>
      <th>tBodyAcc-STD-1</th>
      <th>tBodyAcc-STD-2</th>
      <th>tBodyAcc-STD-3</th>
      <th>tBodyAcc-Mad-1</th>
      <th>tBodyAcc-Mad-2</th>
      <th>tBodyAcc-Mad-3</th>
      <th>tBodyAcc-Max-1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.044</td>
      <td>-0.006</td>
      <td>-0.035</td>
      <td>-0.995</td>
      <td>-0.988</td>
      <td>-0.937</td>
      <td>-0.995</td>
      <td>-0.989</td>
      <td>-0.953</td>
      <td>-0.795</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.039</td>
      <td>-0.002</td>
      <td>-0.029</td>
      <td>-0.998</td>
      <td>-0.983</td>
      <td>-0.971</td>
      <td>-0.999</td>
      <td>-0.983</td>
      <td>-0.974</td>
      <td>-0.803</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.040</td>
      <td>-0.005</td>
      <td>-0.023</td>
      <td>-0.995</td>
      <td>-0.977</td>
      <td>-0.985</td>
      <td>-0.996</td>
      <td>-0.976</td>
      <td>-0.986</td>
      <td>-0.798</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.040</td>
      <td>-0.012</td>
      <td>-0.029</td>
      <td>-0.996</td>
      <td>-0.989</td>
      <td>-0.993</td>
      <td>-0.997</td>
      <td>-0.989</td>
      <td>-0.993</td>
      <td>-0.798</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.039</td>
      <td>-0.002</td>
      <td>-0.024</td>
      <td>-0.998</td>
      <td>-0.987</td>
      <td>-0.993</td>
      <td>-0.998</td>
      <td>-0.986</td>
      <td>-0.994</td>
      <td>-0.802</td>
    </tr>
  </tbody>
</table>
</div>


```python
y_train = pd.read_table('/Users/nickbecker/Downloads/HAPT Data Set/Train/y_train.txt',
             header = None, sep = " ", names = ['activity_id'])
y_train.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>activity_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>


```python
x_test = pd.read_table('/Users/nickbecker/Downloads/HAPT Data Set/Test/X_test.txt',
             header = None, sep = " ", names = features)
y_test = pd.read_table('/Users/nickbecker/Downloads/HAPT Data Set/Test/y_test.txt',
             header = None, sep = " ", names = ['activity_id'])
```

Now that we've got the train and test data loaded into memory, we can start building a model to predict the activity from the acceleration and angular velocity features.

# Building a Human Activity Classifier

Let's build a model and plot the cross-validation accuracy curves for the training data. We'll split `x_train` and `y_train` into training and validation sets since we only want to predict on the test set once. We'll use 5-fold cross validation to get a good sense of the model's accuracy at different values of C. To get the data for the curves, we'll use the `validation_curve` function from sklearn.


```python
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import validation_curve

C_params = np.logspace(-6, 3, 10)
clf_svc = LinearSVC(random_state = 12)

train_scores, test_scores = validation_curve(
    clf_svc, x_train.values, y_train.values.flatten(),
    param_name = "C", param_range = C_params,
    cv = 5, scoring = "accuracy", n_jobs = -1)
```


We can plot the learning curves by following the `learning_curve` documentation. First, I'll calculate the means and standard deviations for the plot shading.

```python
import seaborn as sns

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

y_min = 0.5
y_max = 1.0

sns.set(font_scale = 1.25)
sns.set_style("darkgrid")
f = plt.figure(figsize = (12, 8))
ax = plt.axes()
plt.title("SVM Training and Validation Accuracy")
plt.xlabel("C Value")
plt.ylabel("Accuracy")
plt.ylim(y_min, y_max)
plt.yticks(np.arange(y_min, y_max + .01, .05))
plt.semilogx(C_params, train_scores_mean, label = "CV Training Accuracy", color = "red")
plt.fill_between(C_params, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha = 0.2, color = "red")
plt.semilogx(C_params, test_scores_mean, label = "CV Validation Accuracy",
             color = "green")
plt.fill_between(C_params, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha = 0.2, color = "green")
plt.legend(loc = "best")
plt.show()
```


![](/images/learning_curves.png?raw=true)


From the graph, it looks like the best value of C is at 10<sup>-1</sup>. The validation accuracy begins slowly decreasing after that 10<sup>-1</sup>, indicating we are overfitting. The validation curve looks great, but we only optimized on C. We don't have to use a linear kernel. We could do a grid search on different kernels and C values. With a larger search space, we might get a different set of optimal parameters.

Let's run a parameter grid search to see what the optimal parameters are for the SVM model. We'll switch to the general Support Vector Classifier in `sklearn` so we can have the option to try non-linear kernels.


```python
from sklearn.svm import SVC

Cs = np.logspace(-6, 3, 10)
parameters = [{'kernel': ['rbf'], 'C': Cs},
              {'kernel': ['linear'], 'C': Cs}]

svc = SVC(random_state = 12)

clf = GridSearchCV(estimator = svc, param_grid = parameters, cv = 5, n_jobs = -1)
clf.fit(x_train.values, y_train.values.flatten())

print clf.best_params_
print clf.best_score_
```

    {'kernel': 'rbf', 'C': 1000.0}
    0.925968842539


So the best cross-validated parameters are a `rbf` kernel with `C = 1000`. Let's use the best model to predict on the test data.


```python
clf.score(x_test, y_test)
```




    0.94339025932953824



94% accuracy. That seems pretty high. But how do we know how good that actually is?

# Evaluating the Model

How much better is the model than the no-information rate model? The no-information rate is the accuracy we'd get if we guessed the most common class for every observation. It's the best we could do with no features.


```python
y_test.activity_id.value_counts().values[0] / y_test.activity_id.value_counts().values.sum()
```




    0.17583807716635041



Our model's accuracy of 94% looks even better compared to the baseline of 18%. That's over five times more accurate than the no-information accuracy. But where does our model fail? Let's look at the crosstab.


```python
crosstab = pd.crosstab(y_test.values.flatten(), clf.predict(x_test),
                          rownames=['True'], colnames=['Predicted'],
                          margins=True)
crosstab
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predicted</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>All</th>
    </tr>
    <tr>
      <th>True</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>488</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>496</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20</td>
      <td>449</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>471</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>16</td>
      <td>399</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>420</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>453</td>
      <td>52</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>508</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>19</td>
      <td>536</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>556</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>544</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>545</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>16</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>23</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>20</td>
      <td>0</td>
      <td>11</td>
      <td>0</td>
      <td>32</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>19</td>
      <td>0</td>
      <td>6</td>
      <td>25</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>11</td>
      <td>1</td>
      <td>34</td>
      <td>0</td>
      <td>49</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>3</td>
      <td>15</td>
      <td>27</td>
    </tr>
    <tr>
      <th>All</th>
      <td>513</td>
      <td>472</td>
      <td>404</td>
      <td>476</td>
      <td>588</td>
      <td>545</td>
      <td>20</td>
      <td>12</td>
      <td>32</td>
      <td>29</td>
      <td>50</td>
      <td>21</td>
      <td>3162</td>
    </tr>
  </tbody>
</table>
</div>



We do really well for `activity_ids` 1-3, 5, 6, and 8, but much worse for 4, 7, and 9-12. Why is that?

One possible answer is that we don't have enough data. The `All` column on the right side of the crosstab indicates that we have way fewer observations for activities 7-12. Getting more data may improve the model, depending on whether it's a bias or variance issue.

Additionally, it's clear the model seems to be systematically mistaking some activities for others (activities 4 and 5, 9 and 11, and 10 and 12 are confused for each other more than others). Maybe there's a reason for this. Let's put the labels on the `activity_ids` and see if we notice any patterns.


```python
crosstab_clean = crosstab.iloc[:-1, :-1]
crosstab_clean.columns = activity_df.activity_label.values
crosstab_clean.index = activity_df.activity_label.values
crosstab_clean
```



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th><sub>WALKING</sub></th>
      <th><sub>WALKING<br>_UPSTAIRS</sub></th>
      <th><sub>WALKING<br>_DOWNSTAIRS</sub></th>
      <th><sub>SITTING</sub></th>
      <th><sub>STANDING</sub></th>
      <th><sub>LAYING</sub></th>
      <th><sub>STAND_<br>TO_SIT</sub></th>
      <th><sub>SIT_<br>TO_STAND</sub></th>
      <th><sub>SIT_<br>TO_LIE</sub></th>
      <th><sub>LIE_<br>TO_SIT</sub></th>
      <th><sub>STAND_<br>TO_LIE</sub></th>
      <th><sub>LIE_<br>TO_STAND</sub></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th><sub>WALKING</sub></th>
      <td>488</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th><sub>WALKING_<br>UPSTAIRS</sub></th>
      <td>20</td>
      <td>449</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th><sub>WALKING_<br>DOWNSTAIRS</sub></th>
      <td>5</td>
      <td>16</td>
      <td>399</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th><sub>SITTING</sub></th>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>453</td>
      <td>52</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th><sub>STANDING</sub></th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>19</td>
      <td>536</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th><sub>LAYING</sub></th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>544</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th><sub>STAND_<br>TO_SIT</sub></th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>16</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th><sub>SIT_<br>TO_STAND</sub></th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th><sub>SIT_<br>TO_LIE</sub></th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>20</td>
      <td>0</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <th><sub>LIE_<br>TO_SIT</sub></th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>19</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th><sub>STAND_<br>TO_LIE</sub></th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>11</td>
      <td>1</td>
      <td>34</td>
      <td>0</td>
    </tr>
    <tr>
      <th><sub>LIE_<br>TO_STAND</sub></th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>3</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>



That makes it way more clear. The model struggles to classify those `activity_ids` because they are similar actions. 4 and 5 are both stationary (sitting and standing), 9 and 11 both involving lying down (sit_to_lie and stand_to_lie), and 10 and 12 both involve standing up from a resting position (lie_to_sit and lie_to_stand). It makes sense that accelerometer and gyroscope data on these actions would be similar.

So with 94% accuracy in this activity classifier scenario, can we start a $3 Billion dollar fitness tracker company? Maybe.

For the simple Fitbit bracelet, we don't really need to distinguish between the 12 different activities themselves--only whether or not the person is walking. If we could predict whether someone is walking or not from the smartphone with near perfect accuracy, we'd be in business.

## Predicting Walking vs. Not Walking

So how do we classify walking? First we need to convert the labels to a binary to indicate whether they represent walking or staying in place. From the activity labels, we know that 1-3 are walking and everything other activity involves staying in place.


```python
y_train['walking_flag'] = y_train.activity_id.apply(lambda x: 1 if x <= 3 else 0)
y_test['walking_flag'] = y_test.activity_id.apply(lambda x: 1 if x <= 3 else 0)
```

Now, we'll train a new SVM model and find the optimal parameters with cross-validation.


```python
from sklearn.svm import SVC

Cs = np.logspace(-6, 3, 10)
parameters = [{'kernel': ['rbf'], 'C': Cs},
              {'kernel': ['linear'], 'C': Cs}]

svc = SVC(random_state = 12)

clf_binary = GridSearchCV(estimator = svc, param_grid = parameters, cv = 5, n_jobs = -1)
clf_binary.fit(x_train.values, y_train.walking_flag.values.flatten())

print clf_binary.best_params_
print clf_binary.best_score_
```


    {'kernel': 'linear', 'C': 0.10000000000000001}
    0.999098751127



```python
clf_binary.score(x_test, y_test.walking_flag)
```




    0.99905123339658441



99.9% accuracy! Let's look at the crosstab.


```python
crosstab_binary = pd.crosstab(y_test.walking_flag.values.flatten(), clf_binary.predict(x_test),
                          rownames=['True'], colnames=['Predicted'],
                          margins=True)
crosstab_binary
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predicted</th>
      <th>0</th>
      <th>1</th>
      <th>All</th>
    </tr>
    <tr>
      <th>True</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1774</td>
      <td>1</td>
      <td>1775</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1385</td>
      <td>1387</td>
    </tr>
    <tr>
      <th>All</th>
      <td>1776</td>
      <td>1386</td>
      <td>3162</td>
    </tr>
  </tbody>
</table>
</div>

<br>

That's beautiful! We can almost perfectly tell when someone is walking from smartphone accelerometer and gyroscope data. Now we just need a second model to estimate the distance traveled and we'll be ready to compete with Fitbit.


***

For those interested, the Jupyter Notebook with all the code can be found in the [Github repository](https://github.com/beckernick/music_recommender) for this post.
