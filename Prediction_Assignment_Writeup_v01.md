---
title: "Prediction Assignment Writeup"
author: "Igor Isaev"
date: "10/29/2020"
output: 
  html_document:
    toc: yes
    keep_md: yes
---

<style type="text/css">
  body{
  font-size: 14pt;
}
</style>


<!-- Rscript -e 'rmarkdown::render("Prediction_Assignment_Writeup_v01.Rmd")' -->

## Overview

Using fitness tracker devices it is now possible to collect a large amount of data about personal activity relatively inexpensively. These types of devices help people regularly quantify how much of a particular activity they do, but they rarely quantify how well they do it. 

This project utilizes data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. The data come from the source: http://groupware.les.inf.puc-rio.br/har. Thanks to these enthusiasts for collecting and providing this data.

The goal of this project is to predict the manner in which the participants did the exercises (the "classe" variable in the Training set), create a report describing which steps were taken to build the final model. And apply the final model to the Test data set to predict 20 different test cases.

The current report provides a short version of the code outputs and some intermediate steps are only described in words to keep the report barely short.


## Exploratory Analysis
Let`s load the necessary libraries, data, and have a look into it.


```r
	library(caret)
	library(doParallel)
	pml_tes <- read.csv('pml-testing.csv', na.strings = c("NA", "#DIV/0!", ""))
	pml_tra <- read.csv('pml-training.csv', na.strings = c("NA", "#DIV/0!", ""))
```
The Training and Test data consist of 160 variables/columns each. 
The first 159 variable names are identical in both data sets, but the last is "classe" in the Training data set and "problem_id" in the Test data set. The "classe" we should predict and the "problem_id" is simply an observation number in the Test data set.  
Here are the last twelve data sets names: 

```r
	names(pml_tra)[-(1:148)]
```

```
##  [1] "stddev_yaw_forearm" "var_yaw_forearm"    "gyros_forearm_x"   
##  [4] "gyros_forearm_y"    "gyros_forearm_z"    "accel_forearm_x"   
##  [7] "accel_forearm_y"    "accel_forearm_z"    "magnet_forearm_x"  
## [10] "magnet_forearm_y"   "magnet_forearm_z"   "classe"
```

```r
	names(pml_tes)[-(1:148)]
```

```
##  [1] "stddev_yaw_forearm" "var_yaw_forearm"    "gyros_forearm_x"   
##  [4] "gyros_forearm_y"    "gyros_forearm_z"    "accel_forearm_x"   
##  [7] "accel_forearm_y"    "accel_forearm_z"    "magnet_forearm_x"  
## [10] "magnet_forearm_y"   "magnet_forearm_z"   "problem_id"
```
Some of the variables in the Training set have a lot of NA values.  
The following code provides detailed information about the variables.

```r
	n_names <- names(pml_tra)
	xc_names = NULL
	xc_val_length = NULL
	xc_uval_length = NULL
	xc_var_claa = NULL
	xc_nofna = NULL
	for(xc1 in 1:length(n_names)){
	  xc_names[xc1] = n_names[xc1]
	  xc_val_length[xc1] = length( pml_tra[, n_names[xc1]] )
	  xc_uval_length[xc1] = length( unique(pml_tra[, n_names[xc1]]) )
	  xc_var_claa[xc1] = class(pml_tra[, n_names[xc1]])
	  xc_nofna[xc1] = sum(is.na( pml_tra[, n_names[xc1]] ))
	}
	variables_summary = data.frame(VarName = xc_names, 
	                               Values = xc_val_length, 
	                               Unique = xc_uval_length,
	                               VarClass = xc_var_claa,
	                               N_of_NA = xc_nofna)
```

```r
	variables_summary[1:17,]
```

```
##                 VarName Values Unique  VarClass N_of_NA
## 1                     X  19622  19622   integer       0
## 2             user_name  19622      6 character       0
## 3  raw_timestamp_part_1  19622    837   integer       0
## 4  raw_timestamp_part_2  19622  16783   integer       0
## 5        cvtd_timestamp  19622     20 character       0
## 6            new_window  19622      2 character       0
## 7            num_window  19622    858   integer       0
## 8             roll_belt  19622   1330   numeric       0
## 9            pitch_belt  19622   1840   numeric       0
## 10             yaw_belt  19622   1957   numeric       0
## 11     total_accel_belt  19622     29   integer       0
## 12   kurtosis_roll_belt  19622    396   numeric   19226
## 13  kurtosis_picth_belt  19622    316   numeric   19248
## 14    kurtosis_yaw_belt  19622      1   logical   19622
## 15   skewness_roll_belt  19622    394   numeric   19225
## 16 skewness_roll_belt.1  19622    337   numeric   19248
## 17    skewness_yaw_belt  19622      1   logical   19622
```
The table shows only the first 17 variables, but the main features are described by them:  

- there are variables of 'character' type that should be converted to 'factor' variables for analysis  
- there are also some variables of class 'logical', which consist only NA values  
- some numeric variables also have a lot of NAs (more than 90%)

The manual about random forest algorithm (the most preferable algorithm) says that it is not handling missing values, that is why the current strategy is to get rid of them.  
There are also some variables that are not related to the data collected directly from the sensors but represent only descriptive information.


## Data preparation and cleaning


```r
	set.seed(15)
	# indexes of variables that consist of only NA
	idx_var_discarded  <- c(5, 6, 14, 17, 26, 89, 92, 101, 127, 130, 139)
	# indexes of variables that consist NA in general
	idx_var_ineligible <- c(12:36, 50:59, 69:83, 87:101, 103:112, 125:139, 141:150)
	# indexes of descriptive variables (one may use them in the data analysis but it is not necessary)
	idx_unnecessary_vars <- c(1,2,3,4,5,6,7)
	#let`s convert our target variable into factor
	pml_tra[,160] <- as.factor(pml_tra[,160])
	# let`s create cleaned data set
	pml_tdc <- pml_tra[, -c(idx_var_discarded, idx_var_ineligible, idx_unnecessary_vars)]
	sum(is.na(pml_tdc))
```

```
## [1] 0
```

```r
	#To preform cross validation of models, we should split our data to training and testing subsets
	idxTestC = createDataPartition(pml_tdc$classe, p = 0.25, list = FALSE)
	pmlTestC = pml_tdc[ idxTestC, ]
	pmlTrainC = pml_tdc[ -idxTestC, ]
	dim(pmlTestC)
```

```
## [1] 4907   53
```

```r
	dim(pmlTrainC)
```

```
## [1] 14715    53
```
## Models training and comparison
In order to obtain a reliable model, let`s compare a few algorithms 'rpart', 'rf', and 'gbm'.
For acceleration of the training process, we use 'doParallel' library to utilize all cores of the processor.


```r
	#Decision Tree
	cl <- makePSOCKcluster(8)
	registerDoParallel(cl)
	  model_DT <- train(classe~., data = pmlTrainC, method = "rpart")
	stopCluster(cl)

	model_DT
```

```
## CART 
## 
## 14715 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 14715, 14715, 14715, 14715, 14715, 14715, ... 
## Resampling results across tuning parameters:
## 
##   cp          Accuracy   Kappa     
##   0.03684710  0.5059724  0.35727708
##   0.06011396  0.4049157  0.19237885
##   0.11604938  0.3284289  0.06908287
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was cp = 0.0368471.
```

```r
	prediction_DT_TestC <- predict(model_DT, newdata = pmlTestC)
	confusionMatrix(prediction_DT_TestC, pmlTestC$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1249  408  394  359  125
##          B   16  311   26  158  121
##          C  102  231  436  287  225
##          D    0    0    0    0    0
##          E   28    0    0    0  431
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4946          
##                  95% CI : (0.4805, 0.5087)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3397          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8953  0.32737  0.50935   0.0000  0.47783
## Specificity            0.6338  0.91888  0.79141   1.0000  0.99301
## Pos Pred Value         0.4927  0.49209  0.34036      NaN  0.93900
## Neg Pred Value         0.9384  0.85053  0.88417   0.8362  0.89411
## Prevalence             0.2843  0.19360  0.17444   0.1638  0.18382
## Detection Rate         0.2545  0.06338  0.08885   0.0000  0.08783
## Detection Prevalence   0.5166  0.12880  0.26106   0.0000  0.09354
## Balanced Accuracy      0.7646  0.62312  0.65038   0.5000  0.73542
```

```r
	#Generalized Boosted Model
	cl <- makePSOCKcluster(8)
	registerDoParallel(cl)
	  model_GBM <- train(classe~., data = pmlTrainC, method = "gbm", verbose = FALSE)
	stopCluster(cl)

	model_GBM
```

```
## Stochastic Gradient Boosting 
## 
## 14715 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 14715, 14715, 14715, 14715, 14715, 14715, ... 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy   Kappa    
##   1                   50      0.7486102  0.6813291
##   1                  100      0.8178923  0.7695217
##   1                  150      0.8502711  0.8105300
##   2                   50      0.8517056  0.8121285
##   2                  100      0.9033678  0.8777131
##   2                  150      0.9282571  0.9092138
##   3                   50      0.8932482  0.8648652
##   3                  100      0.9386017  0.9223052
##   3                  150      0.9574916  0.9462142
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## Accuracy was used to select the optimal model using the largest value.
## The final values used for the model were n.trees = 150, interaction.depth =
##  3, shrinkage = 0.1 and n.minobsinnode = 10.
```

```r
	prediction_GBM_TestC <- predict(model_GBM, newdata = pmlTestC)
	confusionMatrix(prediction_GBM_TestC, pmlTestC$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1368   24    0    4    2
##          B   19  906   32    8   11
##          C    7   16  806   15    7
##          D    1    4   18  769   10
##          E    0    0    0    8  872
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9621          
##                  95% CI : (0.9564, 0.9673)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.952           
##                                           
##  Mcnemar's Test P-Value : 6.787e-05       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9806   0.9537   0.9416   0.9565   0.9667
## Specificity            0.9915   0.9823   0.9889   0.9920   0.9980
## Pos Pred Value         0.9785   0.9283   0.9471   0.9589   0.9909
## Neg Pred Value         0.9923   0.9888   0.9877   0.9915   0.9926
## Prevalence             0.2843   0.1936   0.1744   0.1638   0.1838
## Detection Rate         0.2788   0.1846   0.1643   0.1567   0.1777
## Detection Prevalence   0.2849   0.1989   0.1734   0.1634   0.1793
## Balanced Accuracy      0.9861   0.9680   0.9652   0.9742   0.9824
```

```r
	#Random Forest
	cl <- makePSOCKcluster(8)
	registerDoParallel(cl)
	  model_RF <- train(classe~., data = pmlTrainC, method = "rf")
	stopCluster(cl)

	model_RF
```

```
## Random Forest 
## 
## 14715 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 14715, 14715, 14715, 14715, 14715, 14715, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9896714  0.9869298
##   27    0.9894127  0.9866022
##   52    0.9812526  0.9762769
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 2.
```

```r
	prediction_RF_TestC <- predict(model_RF, newdata = pmlTestC)
	confusionMatrix(prediction_RF_TestC, pmlTestC$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    2    0    0    0
##          B    0  947    8    0    0
##          C    0    1  846    9    0
##          D    0    0    2  795    0
##          E    0    0    0    0  902
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9955          
##                  95% CI : (0.9932, 0.9972)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9943          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9968   0.9883   0.9888   1.0000
## Specificity            0.9994   0.9980   0.9975   0.9995   1.0000
## Pos Pred Value         0.9986   0.9916   0.9883   0.9975   1.0000
## Neg Pred Value         1.0000   0.9992   0.9975   0.9978   1.0000
## Prevalence             0.2843   0.1936   0.1744   0.1638   0.1838
## Detection Rate         0.2843   0.1930   0.1724   0.1620   0.1838
## Detection Prevalence   0.2847   0.1946   0.1744   0.1624   0.1838
## Balanced Accuracy      0.9997   0.9974   0.9929   0.9942   1.0000
```
The obtained accuracies are Decision Tree: 0.4946 , Generalized Boosted Model: 0.9621, Random Forest: 0.9955.  
The best performance shows Random Forest algorithm. 

## Variable Importance studies
In order to avoid the overfitting problem, we may reduce the number of predictors. The applied criteria for the predictor`s selection is to keep as little as possible the most important variables, while the prediction accuracy is still high.

A list of variables importance is:

```r
	varImpRF <- varImp(model_RF)
	varImpRF <- varImpRF$importance[order(varImpRF$importance$Overall,decreasing = TRUE),,drop = FALSE]
	varImp_var_list <- rownames(varImpRF)
	varImp_val <- varImpRF$Overall
```

The tests on 18, 13, 9, 6, 5, 4, 3, 2, and 1 the most important variables resulted in the selection of the 6 most important variables. This choice was made because the model with the 6 most important variables would keep 30% of the importance of the total variables (sum(varImp_val)), it would consist of only 11% of predictors (6 out of 52) and still have an accuracy of 98.5%, which is only 1% less than the complete model.

```r
	varImp_var_list[1:6]
```

```
## [1] "roll_belt"         "yaw_belt"          "magnet_dumbbell_z"
## [4] "pitch_forearm"     "pitch_belt"        "magnet_dumbbell_y"
```

The final model is:

```r
	cl <- makePSOCKcluster(8)
	registerDoParallel(cl)
	  model_RF_final <- train(classe~., data = pmlTrainC[,c(varImp_var_list[1:6],"classe")], method = "rf")
	stopCluster(cl)

	model_RF_final
```

```
## Random Forest 
## 
## 14715 samples
##     6 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 14715, 14715, 14715, 14715, 14715, 14715, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##   2     0.9751721  0.9686165
##   4     0.9719828  0.9645851
##   6     0.9662190  0.9572994
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 2.
```

```r
	prediction_RF_final_TestC <- predict(model_RF_final, newdata = pmlTestC)
	confusionMatrix(prediction_RF_final_TestC, pmlTestC$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1387    6    3    1    1
##          B    3  921    5    0    5
##          C    2   15  838    3    2
##          D    3    8   10  798    3
##          E    0    0    0    2  891
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9853          
##                  95% CI : (0.9816, 0.9885)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9814          
##                                           
##  Mcnemar's Test P-Value : 0.002448        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9943   0.9695   0.9790   0.9925   0.9878
## Specificity            0.9969   0.9967   0.9946   0.9942   0.9995
## Pos Pred Value         0.9921   0.9861   0.9744   0.9708   0.9978
## Neg Pred Value         0.9977   0.9927   0.9956   0.9985   0.9973
## Prevalence             0.2843   0.1936   0.1744   0.1638   0.1838
## Detection Rate         0.2827   0.1877   0.1708   0.1626   0.1816
## Detection Prevalence   0.2849   0.1903   0.1753   0.1675   0.1820
## Balanced Accuracy      0.9956   0.9831   0.9868   0.9933   0.9937
```


## Prediction of the Test data set
Let`s apply the final model to the Test data set and submit it to the Quiz part. 

```r
	prediction_RF_TestPML <- predict(model_RF_final, newdata = pml_tes)
	prediction_RF_TestPML
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
