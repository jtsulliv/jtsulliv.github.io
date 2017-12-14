---
title: "Churn Prediction with R: Logistic Regression and Random Forest"
date: 2017-12-14
tage: [machine learning]

header:
  image: "churn/churn_image.jpg"


excerpt: "Churn Prediction, Logistic Regression, Random Forest, AUC, Cross-Validation"
---
**R Code:** [Churn Prediction with R](tbd)

## Introduction
Subscription based services typically make money in the following three ways:

1. Acquire new customers
2. Upsell customers
3. Retain existing customers

In this article I'm going to focus on customer retention.  To do this, I'm going to build a customer churn predictive model.

The motivation for this model is return on investment (ROI).  If a company interacted with every single customer, the cost would be astronomical.  Focusing retention efforts on a small subset of high risk customers is a much more effective strategy.

## Wrangling the Data
The dataset I'm going to be working with can be found on the [IBM Watson Analytics website](https://www.ibm.com/communities/analytics/watson-analytics-blog/guide-to-sample-datasets/).

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

## [1] "[7,043 x 21]"
```

```r
# names of the data
names(df)

##  [1] "customerID"       "gender"           "SeniorCitizen"   
##  [4] "Partner"          "Dependents"       "tenure"          
##  [7] "PhoneService"     "MultipleLines"    "InternetService"
## [10] "OnlineSecurity"   "OnlineBackup"     "DeviceProtection"
## [13] "TechSupport"      "StreamingTV"      "StreamingMovies"
## [16] "Contract"         "PaperlessBilling" "PaymentMethod"   
## [19] "MonthlyCharges"   "TotalCharges"     "Churn"
```

Taking a look we see that there are 21 features, and 7043 rows of observances.  The features are named pretty well, such as "PhoneService" and "TechSupport."  The target feature we'll be attempting to predict is "Churn".  We can dig a little deeper and take a look at the data types of the features.

```r
# data types
glimpse(df)

## Observations: 7,043
## Variables: 21
## $ customerID       <chr> "7590-VHVEG", "5575-GNVDE", "3668-QPYBK", "77...
## $ gender           <chr> "Female", "Male", "Male", "Male", "Female", "...
## $ SeniorCitizen    <int> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
## $ Partner          <chr> "Yes", "No", "No", "No", "No", "No", "No", "N...
## $ Dependents       <chr> "No", "No", "No", "No", "No", "No", "Yes", "N...
## $ tenure           <int> 1, 34, 2, 45, 2, 8, 22, 10, 28, 62, 13, 16, 5...
## $ PhoneService     <chr> "No", "Yes", "Yes", "No", "Yes", "Yes", "Yes"...
## $ MultipleLines    <chr> "No phone service", "No", "No", "No phone ser...
## $ InternetService  <chr> "DSL", "DSL", "DSL", "DSL", "Fiber optic", "F...
## $ OnlineSecurity   <chr> "No", "Yes", "Yes", "Yes", "No", "No", "No", ...
## $ OnlineBackup     <chr> "Yes", "No", "Yes", "No", "No", "No", "Yes", ...
## $ DeviceProtection <chr> "No", "Yes", "No", "Yes", "No", "Yes", "No", ...
## $ TechSupport      <chr> "No", "No", "No", "Yes", "No", "No", "No", "N...
## $ StreamingTV      <chr> "No", "No", "No", "No", "No", "Yes", "Yes", "...
## $ StreamingMovies  <chr> "No", "No", "No", "No", "No", "Yes", "No", "N...
## $ Contract         <chr> "Month-to-month", "One year", "Month-to-month...
## $ PaperlessBilling <chr> "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes"...
## $ PaymentMethod    <chr> "Electronic check", "Mailed check", "Mailed c...
## $ MonthlyCharges   <dbl> 29.85, 56.95, 53.85, 42.30, 70.70, 99.65, 89....
## $ TotalCharges     <dbl> 29.85, 1889.50, 108.15, 1840.75, 151.65, 820....
## $ Churn            <chr> "No", "No", "Yes", "No", "Yes", "Yes", "No", ...
```

The data contains various categorical features that are character types, such as the "Dependents" feature, that has values of either "Yes" or "No".  There's also numeric types, which includes "MonthlyCharges" and "TotalCharges".  The "SeniorCitizen" variable is an integer type, but it really represents "Yes" and "No" so we'll convert that to a factor.  We'll investigate the "tenure" variable, which is also an integer, later on.

For now, let's start by transforming the character variables, as well as the "SeniorCitizen"" variable, to factor types.

```r
df <- df %>% mutate_if(is.character, as.factor)
df$SeniorCitizen <- as.factor(df$SeniorCitizen)
glimpse(df)
```
```r
## Observations: 7,043
## Variables: 21
## $ customerID       <fctr> 7590-VHVEG, 5575-GNVDE, 3668-QPYBK, 7795-CFO...
## $ gender           <fctr> Female, Male, Male, Male, Female, Female, Ma...
## $ SeniorCitizen    <fctr> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...
## $ Partner          <fctr> Yes, No, No, No, No, No, No, No, Yes, No, Ye...
## $ Dependents       <fctr> No, No, No, No, No, No, Yes, No, No, Yes, Ye...
## $ tenure           <int> 1, 34, 2, 45, 2, 8, 22, 10, 28, 62, 13, 16, 5...
## $ PhoneService     <fctr> No, Yes, Yes, No, Yes, Yes, Yes, No, Yes, Ye...
## $ MultipleLines    <fctr> No phone service, No, No, No phone service, ...
## $ InternetService  <fctr> DSL, DSL, DSL, DSL, Fiber optic, Fiber optic...
## $ OnlineSecurity   <fctr> No, Yes, Yes, Yes, No, No, No, Yes, No, Yes,...
## $ OnlineBackup     <fctr> Yes, No, Yes, No, No, No, Yes, No, No, Yes, ...
## $ DeviceProtection <fctr> No, Yes, No, Yes, No, Yes, No, No, Yes, No, ...
## $ TechSupport      <fctr> No, No, No, Yes, No, No, No, No, Yes, No, No...
## $ StreamingTV      <fctr> No, No, No, No, No, Yes, Yes, No, Yes, No, N...
## $ StreamingMovies  <fctr> No, No, No, No, No, Yes, No, No, Yes, No, No...
## $ Contract         <fctr> Month-to-month, One year, Month-to-month, On...
## $ PaperlessBilling <fctr> Yes, No, Yes, No, Yes, Yes, Yes, No, Yes, No...
## $ PaymentMethod    <fctr> Electronic check, Mailed check, Mailed check...
## $ MonthlyCharges   <dbl> 29.85, 56.95, 53.85, 42.30, 70.70, 99.65, 89....
## $ TotalCharges     <dbl> 29.85, 1889.50, 108.15, 1840.75, 151.65, 820....
## $ Churn            <fctr> No, No, Yes, No, Yes, Yes, No, No, Yes, No, ...
```

Now lets look for missing values.  

```r
df %>% map(~ sum(is.na(.)))

## $customerID
## [1] 0
##
## $gender
## [1] 0
##
## $SeniorCitizen
## [1] 0
##
## $Partner
## [1] 0
##
## $Dependents
## [1] 0
##
## $tenure
## [1] 0
##
## $PhoneService
## [1] 0
##
## $MultipleLines
## [1] 0
##
## $InternetService
## [1] 0
##
## $OnlineSecurity
## [1] 0
##
## $OnlineBackup
## [1] 0
##
## $DeviceProtection
## [1] 0
##
## $TechSupport
## [1] 0
##
## $StreamingTV
## [1] 0
##
## $StreamingMovies
## [1] 0
##
## $Contract
## [1] 0
##
## $PaperlessBilling
## [1] 0
##
## $PaymentMethod
## [1] 0
##
## $MonthlyCharges
## [1] 0
##
## $TotalCharges
## [1] 11
##
## $Churn
## [1] 0
```
It looks like "TotalCharges" is the only feature with missing values.  Lets go ahead and impute the 11 missing values using the median value.

```r
# imputing with the median
df <- df %>%
  mutate(TotalCharges = replace(TotalCharges,
                                is.na(TotalCharges),
                                median(TotalCharges, na.rm = T)))

# checking that the imputation worked
sum(is.na(df$TotalCharges))

## [1] 0
```

Now that we've imported the data and done some cleaning, lets start to explore the data.

## Exploring the Data

Let's start by taking a look at the unique values of the factor variables.

```r
df_tbl <- df %>%
  select_if(is.factor) %>%
  summarise_all(n_distinct)


df_tbl[1:8] %>%
  print(width = Inf)

df_tbl[9:15] %>%
  print(width = Inf)

df_tbl[16:18] %>%
  print(width = Inf)
```

There's a unique value for each "customerID" so we probably won't be able to gain much information there.  All of the other factors have four or fewer unique values, so they will all be pretty manageable.

To guide the analysis, I'm going to try and answer the following questions about my customer segments:

1. Are men more likely to churn than women?  
2. Are senior citizens more like to churn?
3. Do individuals with a partner churn more than those without a partner?
4. Do people with dependents churn more than people that do not have dependents?

I'll start with gender.  I wouldn't expect one gender to be more likely than another to churn, but lets see what the data shows.

```r
ggplot(df) +
  geom_bar(aes(x = gender, fill = Churn), position = "dodge")
```

![jpg](/images/churn/figure1.jpg?raw=True)

Taking a look, the results are similar.  Roughly one quarter of the male customers churn, and roughly one quarter of the female customers churn.  We can also take a look at exactly how many people from each gender churned.

```r
df %>%
  group_by(gender,Churn) %>%
  summarise(n=n())
```
Next I'll take a look at senior citizens.  

```r
#SeniorCitizen
ggplot(df) +
  geom_bar(aes(x = SeniorCitizen, fill = Churn), position = "dodge")
```

![jpg](/images/churn/figure2.jpg?raw=True)

```r
df %>%
  group_by(SeniorCitizen) %>%
  summarise(n = n()) %>%
  mutate(freq = n / sum(n))
```
```r
## # A tibble: 2 x 3
##   SeniorCitizen     n      freq
##          <fctr> <int>     <dbl>
## 1             0  5901 0.8378532
## 2             1  1142 0.1621468
```

df %>%
  group_by(SeniorCitizen, Churn) %>%
  summarise(n = n()) %>%
  mutate(freq = n / sum(n))
```
This variable shows a much more meaningful relationship.  Roughly 16% of the customers are senior citizens, and roughly 42% of those senior citizens churn.  On the other hand, of the 84% of customers that are not senior citizens, only 24% churn.  These results show that senior citizens are much more likely to churn.

Now I'm going to take a look at people with partners.

```r
#Partner
ggplot(df) +
  geom_bar(aes(x=Partner, fill = Churn), position = "dodge")

df %>%
  group_by(Partner) %>%
  summarise(n = n()) %>%
  mutate(freq = n / sum(n))

df %>%
  group_by(Partner, Churn) %>%
  summarise(n = n()) %>%
  mutate(freq = n / sum(n))
```
Roughly half of the people have partners.  Of the people with partners, 20% churn.  For people without partners, approximately 33% churn.  

Next, I'll take a look at the Dependents category.
```r
ggplot(df) +
  geom_bar(aes_string(x="Dependents", fill="Churn"), position = "dodge")

df %>% group_by(Dependents, Churn) %>%
  summarise(n=n()) %>%
  mutate(freq = n / sum(n))

df %>% group_by(Dependents) %>%
  summarise(n = n()) %>%
  mutate(freq = n / sum(n))
```
Approximately 30% of the people have dependents, of which 15% churn.  For the other 70% that don't have dependents, 31% churn.

Another useful visualization is the box and whisker plot.  This gives us a little bit more compact visual of our data, and helps us identify outliers.  Lets take a look at some box and whisker plots for total charges of the different customer segments.

```r
# Senior Citizens
ggplot(df, aes(x = SeniorCitizen, y = TotalCharges)) +
  geom_boxplot()

# Partner
ggplot(df, aes(x = Partner, y = TotalCharges)) +
  geom_boxplot()

# Dependents
ggplot(df, aes(x = Dependents, y = TotalCharges)) +
  geom_boxplot()
```

After looking at these initial results, we can ask some more questions.  We might want to compare the total charges of senior citizens, people without partners, and people without dependents.  

These seem to be the subsets of people most likely to churn within their respective customer segments.  Lets compare them so that we can identify where we would potentially focus our efforts.

```r
# Total charges and tenure of senior citizens
df %>%
  select(SeniorCitizen, Churn, TotalCharges, tenure) %>%
  filter(SeniorCitizen == 1, Churn == "Yes") %>%
  summarize(n = n(),
            total = sum(TotalCharges),
            avg_tenure = sum(tenure)/n)

# Total charges and tenure of people without a partner
df %>%
  select(Partner, Churn, TotalCharges, tenure) %>%
  filter(Partner == "No", Churn == "Yes") %>%
  summarise(n = n(),
            total = sum(TotalCharges),
            avg_tenure = sum(tenure)/n)

# Total charges and tenure of people without dependents
df %>%
  select(Dependents, Churn, TotalCharges, tenure) %>%
  filter(Dependents == "No", Churn == "Yes") %>%
  summarise(n = n(),
            total = sum(TotalCharges),
            avg_tenure = sum(tenure)/n)
```

Here's a summary of the total charges for each customer segment that churned:

| Customer Segment | Total Charges |
|------------------|---------------|
| Senior Citizens  | 900,000       |
| No Partners      | 1,300,000     |
| No Dependents    | 2,300,000     |

Based on the results, we should focus our efforts on people without dependents.  This customer segment that churned had nearly 2.3MM in total charges compared to 1.3MM for people without partners, and only 900K for senior citizens.

Let's dig a little deeper and see what services that customer segment uses.

```r
dependents <- df %>% filter(Dependents == "No")

ggplotGrid(ncol=2,
lapply(c("PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup",
         "DeviceProtection"),
       function(col){
         ggplot(dependents,aes_string(col)) + geom_bar(aes(fill=Churn),position="dodge")
       }))

ggplotGrid(ncol=2,
lapply(c("TechSupport","StreamingTV","StreamingMovies","Contract",
         "PaperlessBilling"),
       function(col){
         ggplot(dependents,aes_string(col)) + geom_bar(aes(fill=Churn),position="dodge")
       }))

ggplot(dependents) +
  geom_bar(aes(x=PaymentMethod,fill=Churn), position = "dodge")
```

Taking a look at the results, we gain some potential insights:

1. A lot of people with phone service churned.  Maybe these people don't really use the phone service.  Moving them to a plan without phone service to save them some money on their bill might help retain them.
2.  People with fiber optic internet churned much more than people with DSL or no internet at all.  Maybe moving some of those people to DSL or eliminating their internet service would be an option.  Another option could be some sort of price reduction to their fiber optic plan as some sort of a promotion for being a loyal customer.
3. People without online backup, device protection, and online security churn fairly frequently.  Maybe their devices have crashed, causing them to lose valuable files.  They may have also experienced fraud or identity theft that has left them very unhappy.  Moving these people to some of these services may help safeguard their systems, thus preventing a lot of unwanted headaches.
4. Similarly to online backup and security, those without device protection tended to churn more than those that subscribed ot the service.  Adding device protection to their plans may be a good way to prevent churn.
5. Those without tech support tend to churn more frequently than those with tech support.  Moving customers to tech support accounts might be another potential way to prevent churn.

There are a number of other different insights that we could gain from the data, but this would be a good initial list to investigate further if the company had even more detailed data sets.

Now that we've done a basic exploratory analysis, lets jump into making some predictive models.  

## Predictive Models
To make predictions, I'm going to use both logistic regression and random forest.

Logistic regression is a linear classifier, which makes it easier to interpret than non-linear models.  At the same time, because it's a linear model, it has a high bias towards this type of fit, so it may not perform well on non-linear data.

Random forest is another popular classification method.  Unlike logistic regression, random forest is better at fitting non-linear data.  It can also work well even if there are correlated features, which can be a problem for interpreting logistic regression (although shrinkage methods like the Lasso and Ridge Regression can help with correlated features in a logistic regression model).

I'm not really sure whether my data has a linear or non-linear decision boundary, so this is why I'm going to start with logistic regression, and then test out a random forest model.

I'll be using a train/test validation set approach, for my resampling method, as well as k-fold cross-validation.  This is always good practice to prevent over-fitting.

### Logistic Regression
First I'll develop a logistic regression model, I'm going to start by splitting my data into a training set (75%), and test set (25%).  I'm going to remove the customerID feature because it's unique for each observation, and probably won't add valuable information to my model.

```r
library(caret)

# removing customerID; doesn't add any value to the model
df <- df %>% select(-customerID)  

# train/test split; 75%/25%

# setting the seed for reproducibility
set.seed(5)
inTrain <- createDataPartition(y = df$Churn, p=0.75, list=FALSE)

train <- df[inTrain,]
test <- df[-inTrain,]
```

Now that the data is split, I'll fit a logistic regression model using all of the features.  After I fit the model, I'll take a look at the confusion matrix to see how well the model made predictions on the validation set.

```r
# fitting the model
fit <- glm(Churn~., data=train, family=binomial)

# making predictions
churn.probs <- predict(fit, test, type="response")
head(churn.probs)

# converting probabilities to classes; "Yes" or "No"
contrasts(df$Churn)  # Yes = 1, No = 0
glm.pred = rep("No", length(churn.probs))
glm.pred[churn.probs > 0.5] = "Yes"

confusionMatrix(glm.pred, test$Churn, positive = "Yes")
```

Right out of the box, it looks like the model is performing fairly well.  The accuracy is 81%.  If we were to predict that all results in the test set were the majority class (No), the accuracy would be 73%.

Some of the other metrics that are reported are better measures though, because the response classes are slightly imbalanced (~73% = No, ~27% = Yes).  The sensitivity, which is a measure of the true positive rate (TP/(TP+FN)), is 56%.  The specificity, or true negative rate (TN/(TN+FP)), is 90%.  

This tells us that our model is 56% accurate at correctly identifying true positives.  Phrased another way, the model has correctly identified 56% of people that actually churned.  

Another useful metric is AUC.  This is the area under the receiver operating characteristic (ROC) curve.  By default, I used 0.5 as the threshold for making predictions from the probabilities.  Often times this isn't optimal, so the ROC curve is constructed to plot true positive rate vs. the false positive rate (y=TP, x=FP).  

AUC can take on any value between 0 and 1.  The baseline model used is a random predictor, which has a value of 0.5.  The further this value is from 0.5, the better, with an ideal model having an AUC of 1.

Now I'll take a look at the ROC curve and the AUC value.

```r
library(ROCR)
# need to create prediction object from ROCR
pr <- prediction(churn.probs, test$Churn)

# plotting ROC curve
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)

# AUC value
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc
```

The baseline performance from random guessing provides a 45 degree line, so we can see that our model is outperforming random guessing, which is good.  The AUC measure is 0.85, which is greater than 0.5 (baseline model), which is also good.

#### Feature Selection
If we want to try and improve our model, we can take a look at the summary of the fit and identify which features are significant (p-value < 0.05).  We can also use the varImp function, which takes the absolute value of the test statistic (higher number, the better).

```r
# summary of the model
summary(fit)

# feature importance
sort(varImp(fit), decreasing = TRUE)
```

I'm using a p-value of 0.05 as my threshold (95% confidence interval) for the coefficient estimates, which is 1.96 standard deviations from the mean, so this will be my cutoff for which features to include.  Now lets fit a model with those features and see how it compares.

```r
# fitting the model
fit <- glm(Churn~SeniorCitizen + tenure + MultipleLines + InternetService + StreamingTV + Contract + PaperlessBilling + PaymentMethod + TotalCharges
           , data=train,
           family=binomial)

# making predictions
churn.probs <- predict(fit, test, type="response")
head(churn.probs)

# converting probabilities to classes; "Yes" or "No"
contrasts(df$Churn)  # Yes = 1, No = 0
glm.pred = rep("No", length(churn.probs))
glm.pred[churn.probs > 0.5] = "Yes"

confusionMatrix(glm.pred, test$Churn, positive = "Yes")
```
The accuracy has remained virtually unchanged, with a value of 80%.  Similarly, the true positive rate (55%) and true negative rate (89%) haven't changed.  

Likely there is a good amount of multicollinearity in the original model with all of the features.  From a predictive standpoint, we can see that excluding features that aren't significant does not influence our results.  

The true advantage of simplifying down the model and excluding those features is interpretability.  With multicollinearity, the coefficient estimates are unstable, so depending on our sample, they can change drastically.  Simplifying the model down and attempting to exclude some of this multicollinearity makes the estimates more stable.

We can see evidence of this in the standard error.  The InternetService feature has a standard error of 0.9 in the original model, but in the simplified model its reduced to 0.1.  This tells us that our second, more simplified, model has much more stable coefficient estimates.

Now lets also take a look at the ROC curve, and AUC.

```r
library(ROCR)
# need to create prediction object from ROCR
pr <- prediction(churn.probs, test$Churn)

# plotting ROC curve
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)

# AUC value
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc
```
Similarly to the other metrics, the AUC for this model is 0.85, which is the same as the original model.  Next, I'll use a random forest model to make predictions.  

### Random Forest
I'm going to be using a random forest model, because it can deal with non-linearities better than logistic regression.  I'm not really sure if my data has strong non-linear relationships, but if the random forest model outperforms logistic regression, it might.

Random forest uses multiple decision trees to make predictions.  Single decision trees on their own can be very effective at learning non-linear relationships (low bias, but high variance).  Due to their high variance, they can tend to over-fit.  Random forest reduces this variance by averaging many trees (at the sacrifice of a slight increase in the bias).

I'll start by fitting the model to all of the features.

```r
library(randomForest)
churn.rf = randomForest(Churn~., data = train, importance = T)

churn.rf

churn.predict.prob <- predict(churn.rf, test, type="prob")

churn.predict <- predict(churn.rf, test)
confusionMatrix(churn.predict, test$Churn, positive = "Yes")
```

The accuracy of the model is 79%, the true positive rate is 52%, and the true negative rate is 89%.  It looks like the model performed slightly worse than logistic regression, but not by much.  Now I'll take a look at the ROC curve and AUC.  

```r
library(ROCR)
# need to create prediction object from ROCR
pr <- prediction(churn.predict.prob[,2], test$Churn)

# plotting ROC curve
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)

# AUC value
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc
```
The AUC is 0.84, which is pretty close to the logistic regression model.  Now I'll take a look at the feature importance of the variables in the random forest model.

```r
importance(churn.rf)
varImpPlot(churn.rf)
```
There's two measures of feature importance that are reported, mean decrease in accuracy, and mean decrease in gini.  

The first is the decrease in accuracy of out of bag samples when the variable feature is excluded from the model.  

The second is the mean decrease in gini.  This metric has to do with the decrease in node impurity that results from splits over that variable.  The higher the mean decrease in gini, the lower the node impurity.  Basically, this means that the lower the node impurity, the more likely the split will produce a left node that is dedicated to one class, and a right node that is dedicated to another class.  If the split is totally pure, the left node will be 100% of one class, and the right will be 100% of another class.  This is obviously more optimal for making predictions than having two nodes of mixed classes.

Some of the features that were important in the logistic regression model, such as tenure and TotalCharges, are also important to the random forest model.  Other features like TechSupport and MonthlyCharges were not significant in the logistic regression model, but are ranked fairly high for the random forest model.

#### Parameter Tuning
Rather than try a different subset of features, I'll try to tune some of the parameters of the random forest model.  First I'll change the number of variables that are sampled at each split.  Right now the default is 4, so I'll try several other numbers, and use the AUC as the comparison metric.

```r
# changing the number of variables to try at each split
# mtry = 8, 12, 16, 20

# fitting the model
churn.rf = randomForest(Churn~., data = train, mtry = 20, importance = T)

churn.predict.prob <- predict(churn.rf, test, type="prob")

churn.predict <- predict(churn.rf, test)
confusionMatrix(churn.predict, test$Churn, positive = "Yes")

# need to create prediction object from ROCR
pr <- prediction(churn.predict.prob[,2], test$Churn)

# AUC value
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc
# mtry = 8; AUC = 0.83
# mtry = 12; AUC = 0.83
# mtry = 16; AUC = 0.83
# mtry = 20; AUC = 0.82
```

Changing the number of variables to try didn't improve the model.  The resulting AUC decreased slightly to 0.83.  Now I'll try changing the number of trees.  

```r
# changing the number of trees
# ntree = 25, 250, 500, 750

# fitting the model
churn.rf = randomForest(Churn~., data = train, ntree = 750, importance = T)

churn.predict.prob <- predict(churn.rf, test, type="prob")

churn.predict <- predict(churn.rf, test)
confusionMatrix(churn.predict, test$Churn, positive = "Yes")

# need to create prediction object from ROCR
pr <- prediction(churn.predict.prob[,2], test$Churn)

# AUC value
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc
# ntree = 25; AUC = 0.83
# ntree = 250; AUC = 0.84
# ntree = 500; AUC = 0.84
# ntree = 750; AUC = 0.84
```
Changing the number of trees didn't really improve the model either, so I'll stick with the original model (mtry = 4, ntree = 500).

Finally, I'll use K-fold cross-validation with 10 folds, repeated 3 times, to compare the models.

### K-fold Cross Validation
In the previous sections I used a train/test validation procedure for evaluating my models.  In this section, I'll vet the models a little more rigorously using 10-fold cross validation, repeated 3 times.

```r
#k-fold cross val in caret
set.seed(10)

# train control
fitControl <- trainControl(## 10-fold CV
                            method = "repeatedcv",
                            number = 10,
                            ## repeated 3 times
                            repeats = 3,
                            classProbs = TRUE,
                            summaryFunction = twoClassSummary)

# logistic regression model
logreg <- train(Churn ~., df,
                  method = "glm",
                  family = "binomial",
                  trControl = fitControl,
                  metric = "ROC")

  # ROC        Sens       Spec     
  # 0.8453004  0.8960842  0.5519886

# random forest model
rf <- train(Churn ~., df,
                  method = "rf",
                  trControl = fitControl,
                  metric = "ROC")

  # mtry  ROC        Sens       Spec     
  #  2    0.8321455  0.9476230  0.3550946
  # 16    0.8256092  0.8929878  0.4999195
  # 30    0.8193891  0.8894460  0.4986708
```

The results are very similar to the previous results.  The logistic regression model had an AUC of approximately 0.84, and the random forest model had an AUC of approximately 0.83.  

### Cost Evaluation
All of the previous modeling and evaluation metrics were useful, but they don't tell us much about the actual impacts on the business.  In this section, I'll go over the cost implications of implementing, vs. not implementing a predictive model.

To start, I'll make several assumptions related to cost.  Doing a quick search, it looks like the customer acquisition cost in the telecom industry is around $300.  I'll assume that this is the customer acquisition cost in my model as a result of false negative predictions (predicting that a customer was happy, but the customer actually churned).  

Doing another quick search, it looks like customer acquisition cost is approximately five times higher than customer retention costs.  I'll assume that my customer retention costs are $60.  These costs will be incurred during false positives (predicting a customer would churn when they were actually happy), and true positives (predicting unhappy customers correctly).  There will be no cost incurred for true negative predictions (correctly predicting a customer was happy).  

Here's the equation for cost that I'm going to try and minimize:

cost = FN*300 + TP*60 + FP*60 + TN*0

Since the logistic regression model seemed to perform slightly better, I'll use that model.

```r
# fitting the logistic regression model
fit <- glm(Churn~., data=train, family=binomial)

# making predictions
churn.probs <- predict(fit, test, type="response")
head(churn.probs)

# converting probabilities to classes; "Yes" or "No"
contrasts(df$Churn)  # Yes = 1, No = 0
glm.pred = rep("No", length(churn.probs))
glm.pred[churn.probs > 0.5] = "Yes"


x <- confusionMatrix(glm.pred, test$Churn, positive = "Yes")

# threshold values
thresh <- seq(0.1,1.0, length = 10)
t = rep()
for (i in 1:length(thresh)){
  t[[i]] = thresh[[i]]
}
```
