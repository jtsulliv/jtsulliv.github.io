---
title: "Customer Churn Prediction with R: Logistic Regression and Random Forest"
date: 2017-12-12
tags: [machine learning]

excerpt: "Churn Prediction, Logistic Regression, Random Forest, AUC, Cross-Validation"
---

Subscription based services typically make money in the following three ways:

1) Acquire new customers
2) Upsell customers
3) Retain existing customers

In this article I'm going to focus on customer retention.  To do this, I'm going to build a customer churn predictive model.

The motivation for this model is return on investment (ROI).  If a company interacted with every single customer, the cost would be astronomical.  Focusing retention efforts on a small subset of high risk customers is a much more effective strategy.

## Wrangling the Data
The [dataset](https://www.ibm.com/communities/analytics/watson-analytics-blog/guide-to-sample-datasets/) I'm going to be working with can be found on the IBM Watson Analytics website.

This is a sample dataset for a telecommunications company.  We can start by taking a look at the dimensions of the data, as well as the different features.

```r
library(tidyverse)
library(miscset)

# Setting the working directory
path_loc <- "C:/Users/jsullivan/Desktop/Storage/Data Science/Portfolio/Projects/Churn Prediction"
setwd(path_loc)

# reading in the data
df <- read_csv("Telco data.csv")


# dimensions of the data
dim_desc(df)

# names of the data
names(df)
```

This is a test, take two...

```python
library(tidyverse)
library(miscset)

# Setting the working directory
path_loc <- "C:/Users/jsullivan/Desktop/Storage/Data Science/Portfolio/Projects/Churn Prediction"
setwd(path_loc)

# reading in the data
df <- read_csv("Telco data.csv")


# dimensions of the data
dim_desc(df)

# names of the data
names(df)
```
