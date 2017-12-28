---
title: "Churn Prediction: Logistic Regression and Random Forest"
date: 2017-12-14
tags: [machine learning]

header:
  image: "churn/churn_image2.jpg"


excerpt: "Churn Prediction, R, Logistic Regression, Random Forest, AUC, Cross-Validation"
---
**R Code:** [Churn Prediction with R](https://github.com/jtsulliv/Machine-Learning/tree/master/churn-prediction)

In the [previous article](https://jtsulliv.github.io/churn-eda/) I performed an exploratory data analysis of a customer churn dataset from the telecommunications industry.  In this article I'm going to be building predictive models.  

The motivation for these models is return on investment (ROI).  If a company interacted with every single customer, the cost would be astronomical.  Focusing retention efforts on a small subset of high risk customers is a much more effective strategy.

At the end of the article I'll present a hypothetical business scenario in which I project a yearly savings of $4MM in customer retention costs.  This cost savings is achieved by optimizing the threshold of a logistic regression model.  I will make some basic assumptions about customer acquisition and customer retention costs, as well as the size of the telecommunications company.

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
```
```r
##          1          2          3          4          5          6
## 0.32756804 0.77302887 0.56592677 0.20112771 0.05152568 0.15085976
```
```r
# converting probabilities to classes; "Yes" or "No"
contrasts(df$Churn)  # Yes = 1, No = 0
```
```r
##     Yes
## No    0
## Yes   1
```
```r
glm.pred = rep("No", length(churn.probs))
glm.pred[churn.probs > 0.5] = "Yes"

confusionMatrix(glm.pred, test$Churn, positive = "Yes")
```
```r
## Confusion Matrix and Statistics
##
##           Reference
## Prediction   No  Yes
##        No  1165  205
##        Yes  128  262
##                                           
##                Accuracy : 0.8108          
##                  95% CI : (0.7917, 0.8288)
##     No Information Rate : 0.7347          
##     P-Value [Acc > NIR] : 4.239e-14       
##                                           
##                   Kappa : 0.4877          
##  Mcnemar's Test P-Value : 3.117e-05       
##                                           
##             Sensitivity : 0.5610          
##             Specificity : 0.9010          
##          Pos Pred Value : 0.6718          
##          Neg Pred Value : 0.8504          
##              Prevalence : 0.2653          
##          Detection Rate : 0.1489          
##    Detection Prevalence : 0.2216          
##       Balanced Accuracy : 0.7310          
##                                           
##        'Positive' Class : Yes             
##
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
```

![jpg](/images/churn/figure11.jpg?raw=True)

```r
# AUC value
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc
```
```r
## [1] 0.8481338
```
The baseline performance from random guessing provides a 45 degree line, so we can see that our model is outperforming random guessing, which is good.  The AUC measure is 0.85, which is greater than 0.5 (baseline model), which is also good.

#### Feature Selection
If we want to try and improve our model, we can take a look at the summary of the fit and identify which features are significant (p-value < 0.05).  We can also use the varImp function, which takes the absolute value of the test statistic (higher number, the better).

```r
# summary of the model
summary(fit)
```
```r
##
## Call:
## glm(formula = Churn ~ ., family = binomial, data = train)
##
## Deviance Residuals:
##     Min       1Q   Median       3Q      Max  
## -1.8663  -0.6833  -0.2922   0.7327   3.4256  
##
## Coefficients: (7 not defined because of singularities)
##                                        Estimate Std. Error z value
## (Intercept)                           1.599e+00  9.238e-01   1.731
## genderMale                           -5.516e-03  7.492e-02  -0.074
## SeniorCitizen1                        2.461e-01  9.774e-02   2.518
## PartnerYes                           -5.320e-02  9.025e-02  -0.589
## DependentsYes                        -1.466e-01  1.038e-01  -1.413
## tenure                               -6.549e-02  7.174e-03  -9.129
## PhoneServiceYes                       4.170e-01  7.361e-01   0.567
## MultipleLinesNo phone service                NA         NA      NA
## MultipleLinesYes                      4.664e-01  2.002e-01   2.330
## InternetServiceFiber optic            2.157e+00  9.059e-01   2.381
## InternetServiceNo                    -2.166e+00  9.157e-01  -2.365
## OnlineSecurityNo internet service            NA         NA      NA
## OnlineSecurityYes                    -1.578e-01  2.023e-01  -0.780
## OnlineBackupNo internet service              NA         NA      NA
## OnlineBackupYes                       1.354e-01  1.995e-01   0.679
## DeviceProtectionNo internet service          NA         NA      NA
## DeviceProtectionYes                   2.020e-01  2.010e-01   1.005
## TechSupportNo internet service               NA         NA      NA
## TechSupportYes                       -1.577e-01  2.049e-01  -0.769
## StreamingTVNo internet service               NA         NA      NA
## StreamingTVYes                        7.295e-01  3.714e-01   1.964
## StreamingMoviesNo internet service           NA         NA      NA
## StreamingMoviesYes                    7.130e-01  3.698e-01   1.928
## ContractOne year                     -5.932e-01  1.241e-01  -4.780
## ContractTwo year                     -1.179e+00  1.931e-01  -6.105
## PaperlessBillingYes                   3.587e-01  8.657e-02   4.144
## PaymentMethodCredit card (automatic) -1.049e-01  1.317e-01  -0.797
## PaymentMethodElectronic check         2.639e-01  1.098e-01   2.404
## PaymentMethodMailed check            -2.143e-02  1.331e-01  -0.161
## MonthlyCharges                       -5.487e-02  3.600e-02  -1.524
## TotalCharges                          3.788e-04  8.114e-05   4.668
##                                      Pr(>|z|)    
## (Intercept)                            0.0834 .  
## genderMale                             0.9413    
## SeniorCitizen1                         0.0118 *  
## PartnerYes                             0.5555    
## DependentsYes                          0.1577    
## tenure                                < 2e-16 ***
## PhoneServiceYes                        0.5710    
## MultipleLinesNo phone service              NA    
## MultipleLinesYes                       0.0198 *  
## InternetServiceFiber optic             0.0173 *  
## InternetServiceNo                      0.0180 *  
## OnlineSecurityNo internet service          NA    
## OnlineSecurityYes                      0.4354    
## OnlineBackupNo internet service            NA    
## OnlineBackupYes                        0.4972    
## DeviceProtectionNo internet service        NA    
## DeviceProtectionYes                    0.3149    
## TechSupportNo internet service             NA    
## TechSupportYes                         0.4416    
## StreamingTVNo internet service             NA    
## StreamingTVYes                         0.0495 *  
## StreamingMoviesNo internet service         NA    
## StreamingMoviesYes                     0.0538 .  
## ContractOne year                     1.76e-06 ***
## ContractTwo year                     1.03e-09 ***
## PaperlessBillingYes                  3.42e-05 ***
## PaymentMethodCredit card (automatic)   0.4255    
## PaymentMethodElectronic check          0.0162 *  
## PaymentMethodMailed check              0.8721    
## MonthlyCharges                         0.1275    
## TotalCharges                         3.04e-06 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
##
## (Dispersion parameter for binomial family taken to be 1)
##
##     Null deviance: 6113.6  on 5282  degrees of freedom
## Residual deviance: 4377.1  on 5259  degrees of freedom
## AIC: 4425.1
##
## Number of Fisher Scoring iterations: 6
```
```r
# feature importance
sort(varImp(fit), decreasing = TRUE)
```
```r
##                                         Overall
## tenure                               9.12928741
## ContractTwo year                     6.10453339
## ContractOne year                     4.77957880
## TotalCharges                         4.66826552
## PaperlessBillingYes                  4.14385172
## SeniorCitizen1                       2.51829778
## PaymentMethodElectronic check        2.40396896
## InternetServiceFiber optic           2.38067561
## InternetServiceNo                    2.36534330
## MultipleLinesYes                     2.33021152
## StreamingTVYes                       1.96415706
## StreamingMoviesYes                   1.92825920
## MonthlyCharges                       1.52412139
## DependentsYes                        1.41284560
## DeviceProtectionYes                  1.00497919
## PaymentMethodCredit card (automatic) 0.79696515
## OnlineSecurityYes                    0.77995571
## TechSupportYes                       0.76944913
## OnlineBackupYes                      0.67884834
## PartnerYes                           0.58947183
## PhoneServiceYes                      0.56655082
## PaymentMethodMailed check            0.16104087
## genderMale                           0.07362484
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
```
```r
##          1          2          3          4          5          6
## 0.36592800 0.74222067 0.61241105 0.25060677 0.04409168 0.19736195
```
```r
# converting probabilities to classes; "Yes" or "No"
contrasts(df$Churn)  # Yes = 1, No = 0
```
```r
# converting probabilities to classes; "Yes" or "No"
contrasts(df$Churn)  # Yes = 1, No = 0
```
```r
glm.pred = rep("No", length(churn.probs))
glm.pred[churn.probs > 0.5] = "Yes"

confusionMatrix(glm.pred, test$Churn, positive = "Yes")
```
```r
## Confusion Matrix and Statistics
##
##           Reference
## Prediction   No  Yes
##        No  1157  209
##        Yes  136  258
##                                           
##                Accuracy : 0.804           
##                  95% CI : (0.7846, 0.8223)
##     No Information Rate : 0.7347          
##     P-Value [Acc > NIR] : 6.588e-12       
##                                           
##                   Kappa : 0.4708          
##  Mcnemar's Test P-Value : 0.000106        
##                                           
##             Sensitivity : 0.5525          
##             Specificity : 0.8948          
##          Pos Pred Value : 0.6548          
##          Neg Pred Value : 0.8470          
##              Prevalence : 0.2653          
##          Detection Rate : 0.1466          
##    Detection Prevalence : 0.2239          
##       Balanced Accuracy : 0.7236          
##                                           
##        'Positive' Class : Yes             
##
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
```

![jpg](/images/churn/figure12.jpg?raw=True)

```r
# AUC value
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc
```
```r
## [1] 0.8454775
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
```
```r
##
## Call:
##  randomForest(formula = Churn ~ ., data = train, importance = T)
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 4
##
##         OOB estimate of  error rate: 20.29%
## Confusion matrix:
##       No Yes class.error
## No  3502 379  0.09765524
## Yes  693 709  0.49429387
```
```r
churn.predict.prob <- predict(churn.rf, test, type="prob")

churn.predict <- predict(churn.rf, test)
confusionMatrix(churn.predict, test$Churn, positive = "Yes")
```
```r
## Confusion Matrix and Statistics
##
##           Reference
## Prediction   No  Yes
##        No  1148  225
##        Yes  145  242
##                                         
##                Accuracy : 0.7898        
##                  95% CI : (0.77, 0.8086)
##     No Information Rate : 0.7347        
##     P-Value [Acc > NIR] : 4.791e-08     
##                                         
##                   Kappa : 0.4296        
##  Mcnemar's Test P-Value : 4.008e-05     
##                                         
##             Sensitivity : 0.5182        
##             Specificity : 0.8879        
##          Pos Pred Value : 0.6253        
##          Neg Pred Value : 0.8361        
##              Prevalence : 0.2653        
##          Detection Rate : 0.1375        
##    Detection Prevalence : 0.2199        
##       Balanced Accuracy : 0.7030        
##                                         
##        'Positive' Class : Yes           
##
```

The accuracy of the model is 79%, the true positive rate is 52%, and the true negative rate is 89%.  It looks like the model performed slightly worse than logistic regression, but not by much.  Now I'll take a look at the ROC curve and AUC.  

```r
library(ROCR)
# need to create prediction object from ROCR
pr <- prediction(churn.predict.prob[,2], test$Churn)

# plotting ROC curve
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)
```

![jpg](/images/churn/figure13.jpg?raw=True)

```r
# AUC value
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc
```
```r
## [1] 0.837611
```

The AUC is 0.84, which is pretty close to the logistic regression model.  Now I'll take a look at the feature importance of the variables in the random forest model.

```r
importance(churn.rf)
```
```r
##                          No        Yes MeanDecreaseAccuracy
## gender           -0.2787850  0.1640010           -0.1070151
## SeniorCitizen    12.5137057  1.0032629           11.7955391
## Partner           0.6681314 -0.2110722            0.4006978
## Dependents       -3.5717006  8.8013663            3.5375971
## tenure           26.7636624 29.5199677           48.7934452
## PhoneService      0.4547738  7.2310984            5.0965859
## MultipleLines     3.7343699  7.4371935            8.0811306
## InternetService  15.2651738 21.6388724           23.4215171
## OnlineSecurity   10.3995975 21.7833665           19.4301151
## OnlineBackup      8.1641774 11.0629272           12.7885313
## DeviceProtection 11.1750628 -4.1130469            9.5983045
## TechSupport       9.4310594 29.4754798           21.5097751
## StreamingTV       9.9877053  0.2214763            9.3273850
## StreamingMovies  12.1806516  0.7345354           11.6250767
## Contract          7.1397800 29.7611523           32.5802482
## PaperlessBilling -2.6548883 16.6188362            9.5044340
## PaymentMethod     4.8829898 11.7248708           11.8284691
## MonthlyCharges   17.6834891 16.0277495           28.3442378
## TotalCharges     28.2673591 16.8527671           41.1659411
##                  MeanDecreaseGini
## gender                  43.348639
## SeniorCitizen           34.916600
## Partner                 37.067003
## Dependents              31.846637
## tenure                 304.225536
## PhoneService             8.170746
## MultipleLines           41.031725
## InternetService         73.350708
## OnlineSecurity          82.194235
## OnlineBackup            45.627216
## DeviceProtection        41.936300
## TechSupport             79.967321
## StreamingTV             34.535218
## StreamingMovies         35.813841
## Contract               146.945538
## PaperlessBilling        42.636922
## PaymentMethod          110.136025
## MonthlyCharges         298.938071
## TotalCharges           339.656579
```
```r
varImpPlot(churn.rf)
```

![jpg](/images/churn/figure14.jpg?raw=True)

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
```
```r
## Confusion Matrix and Statistics
##
##           Reference
## Prediction   No  Yes
##        No  1144  219
##        Yes  149  248
##                                           
##                Accuracy : 0.7909          
##                  95% CI : (0.7711, 0.8097)
##     No Information Rate : 0.7347          
##     P-Value [Acc > NIR] : 2.544e-08       
##                                           
##                   Kappa : 0.4367          
##  Mcnemar's Test P-Value : 0.0003221       
##                                           
##             Sensitivity : 0.5310          
##             Specificity : 0.8848          
##          Pos Pred Value : 0.6247          
##          Neg Pred Value : 0.8393          
##              Prevalence : 0.2653          
##          Detection Rate : 0.1409          
##    Detection Prevalence : 0.2256          
##       Balanced Accuracy : 0.7079          
##                                           
##        'Positive' Class : Yes             
##
```
```r
# need to create prediction object from ROCR
pr <- prediction(churn.predict.prob[,2], test$Churn)

# AUC value
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc
```
```r
## [1] 0.8262105
```
```r
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
```
```r
## Confusion Matrix and Statistics
##
##           Reference
## Prediction   No  Yes
##        No  1150  224
##        Yes  143  243
##                                           
##                Accuracy : 0.7915          
##                  95% CI : (0.7717, 0.8102)
##     No Information Rate : 0.7347          
##     P-Value [Acc > NIR] : 1.844e-08       
##                                           
##                   Kappa : 0.4338          
##  Mcnemar's Test P-Value : 2.967e-05       
##                                           
##             Sensitivity : 0.5203          
##             Specificity : 0.8894          
##          Pos Pred Value : 0.6295          
##          Neg Pred Value : 0.8370          
##              Prevalence : 0.2653          
##          Detection Rate : 0.1381          
##    Detection Prevalence : 0.2193          
##       Balanced Accuracy : 0.7049          
##                                           
##        'Positive' Class : Yes             
##
```
```r
# need to create prediction object from ROCR
pr <- prediction(churn.predict.prob[,2], test$Churn)

# AUC value
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc
```
```r
## [1] 0.8392315
```
```r
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
```
```r
## ROC        Sens       Spec     
## 0.8453004  0.8960842  0.5519886
```
```r
# random forest model
rf <- train(Churn ~., df,
                  method = "rf",
                  trControl = fitControl,
                  metric = "ROC")
```
```r
## mtry  ROC        Sens       Spec     
##  2    0.8321455  0.9476230  0.3550946
## 16    0.8256092  0.8929878  0.4999195
## 30    0.8193891  0.8894460  0.4986708
```

The results are very similar to the previous results.  The logistic regression model had an AUC of approximately 0.84, and the random forest model had an AUC of approximately 0.83.  

### Cost Evaluation
All of the previous modeling and evaluation metrics were useful, but they don't tell us much about the actual impacts on the business.  In this section, I'll go over the cost implications of implementing, vs. not implementing a predictive model.

To start, I'll make several assumptions related to cost.  Doing a quick search, it looks like the customer acquisition cost in the telecom industry is around $300.  I'll assume that this is the customer acquisition cost in my model as a result of false negative predictions (predicting that a customer was happy, but the customer actually churned).  

Doing another quick search, it looks like customer acquisition cost is approximately five times higher than customer retention costs.  I'll assume that my customer retention costs are $60.  These costs will be incurred during false positives (predicting a customer would churn when they were actually happy), and true positives (predicting unhappy customers correctly).  There will be no cost incurred for true negative predictions (correctly predicting a customer was happy).  

Here's the equation for cost that I'm going to try and minimize:

cost = FN(300) + TP(60) + FP(60) + TN(0)

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

# cost as a function of threshold
thresh <- seq(0.1,1.0, length = 10)
cost = rep(0,length(thresh))
for (i in 1:length(thresh)){

  glm.pred = rep("No", length(churn.probs))
  glm.pred[churn.probs > thresh[i]] = "Yes"
  x <- confusionMatrix(glm.pred, test$Churn, positive = "Yes")
  TN <- x$table[1]/1760
  FP <- x$table[2]/1760
  FN <- x$table[3]/1760
  TP <- x$table[4]/1760
  cost[i] = FN*300 + TP*60 + FP*60 + TN*0
}


# simple model - assume threshold is 0.5
glm.pred = rep("No", length(churn.probs))
glm.pred[churn.probs > 0.5] = "Yes"
x <- confusionMatrix(glm.pred, test$Churn, positive = "Yes")
TN <- x$table[1]/1760
FP <- x$table[2]/1760
FN <- x$table[3]/1760
TP <- x$table[4]/1760
cost_simple = FN*300 + TP*60 + FP*60 + TN*0


# putting results in a dataframe for plotting
dat <- data.frame(
  model = c(rep("optimized",10),"simple"),
  cost_thresh = c(cost,cost_simple),
  thresh_plot = c(thresh,0.5)
)

ggplot(dat, aes(x = thresh_plot, y = cost_thresh, group = model, colour = model)) +
  geom_line() +
  geom_point()


# cost savings of optimized model (threshold = 0.2) compared to baseline model (threshold = 0.5)

savings_per_customer = cost_simple - min(cost)

total_savings = 500000*savings_per_customer

## total savings:  4107955
```
![jpg](/images/churn/image20.jpg?raw=True)

If we assume that our baseline model is the logistic regression model with a threshold of 0.5, the cost associated with this model is $48/customer.

If we optimize the model and use a threshold of 0.2, our customer retention cost is reduced to $40/customer.

Assuming a customer base of 500,000 this comes out to a yearly savings of over $4MM.  

This example illustrates the value of optimizing a machine learning model for accuracy, as well as impact on the business.  
