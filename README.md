# Breast-Cancer-ML-R
Introduction to Machine Learning Techniques in R Using Breast Cancer Data

Abstract: This document will provide an overview of some popular machine learning techniques, as well as a typical project flow for a machine learning project.  

Author: Nicholas Pylypiw--Cardinal Solutions

Data Source: https://www.kaggle.com/uciml/breast-cancer-wisconsin-data

Title: Machine Learning and Breast Cancer Prediction

This case study will require the following packages.

```{r Load Packages}

#install.packages("readr")
#install.packages("rpart")
#install.packages("rpart.plot")
#install.packages("party")
#install.packages("ggplot2")
#install.packages("corrplot")
#install.packages("plyr")
#install.packages("plotly")
#install.packages("class")
#install.packages("randomForest")
#install.packages("e1071, dependencies=TRUE)
#install.packages("caret")
#install.packages("dplyr")
#install.packages("scales")

library(readr)
library(rpart)
library(rpart.plot)
library(party)
library(ggplot2)
library(corrplot)
library(plyr)
library(plotly)
library(class)
library(randomForest)
library(e1071)
library(caret)
library(dplyr)
library(scales)

```

Load the .csv into the R environment as a data frame.  
-- str() can be used to view the structure of the object, as well as variable types
-- head(data, n) provides the first n rows of the data frame.

```{r Import Data}

# read csv
breast_cancer_raw <- readr::read_csv("~/Projects/Breast Cancer/breast_cancer.csv")

# Look at first 6 rows
head(breast_cancer_raw)

```

Notice in the output from head() that a few variables have spaces in their names.  This naming convention is not 
compatible with many of the R procdures used in this tutorial, so those will have to be changed.

```{r Rename Columns}

# Replaces space in variable name with an underscore
names(breast_cancer_raw) <- gsub(" ", "_", names(breast_cancer_raw))

```

Now the data is ready to be explored a bit.

```{r Explore Data}

# Mean and Five Number Summary for each variable
summary(breast_cancer_raw)

# Quick count of cancer rate in data set
table(breast_cancer_raw$diagnosis)

# Looking at distribution for area_mean variable
hist(breast_cancer_raw$area_mean, main = 'Distribution of area_mean')

```

No race or age variables?!!!!!  Well that's boring... I'll just create them.
Using http://ww5.komen.org/BreastCancer/RaceampEthnicity.html, I created relative probabilities of risk by race.
I just guessed at age probabilities, increasing as patient gets older.

```{r Add Demographics}

# Set seed to your favorite number for replication
set.seed(8675309)

# Divide datasets by diagnosis
breast_cancer_raw_M <- breast_cancer_raw[which(breast_cancer_raw$diagnosis=='M'),]
breast_cancer_raw_B <- breast_cancer_raw[which(breast_cancer_raw$diagnosis=='B'),]

# Assign risk probabilities by race
breast_cancer_raw_M$race <- sample( c('White', 'Black', 'Asian', 'Hispanic', 'Other'), 
                                    nrow(breast_cancer_raw_M), 
                                    replace = TRUE, 
                                    prob = c(.41, .31, .11, 0.14, .03) )
breast_cancer_raw_B$race <- sample( c('White', 'Black', 'Asian', 'Hispanic', 'Other'), 
                                    nrow(breast_cancer_raw_B), 
                                    replace = TRUE, 
                                    prob = c(.28, .28, .18, 0.20, .06) )

# Assign risk probabilities by age
breast_cancer_raw_M$age <- sample( 18:40, 
                                   size = nrow(breast_cancer_raw_M), 
                                   replace = TRUE, 
                                   prob = c(0.005, 0.005, 0.006, 0.006, 0.009, 0.012, 0.016, 0.022, 
                                            0.025, 0.08, 0.17, 0.19, 0.14, 0.044, 0.038, 0.029, 0.027,
                                            0.024, 0.021, 0.028, 0.042, 0.03, 0.031) )
breast_cancer_raw_B$age <- sample( 18:40, 
                                   size = nrow(breast_cancer_raw_B), 
                                   replace = TRUE, 
                                   prob = c(0.01, 0.01, 0.01, 0.012, 0.015, 0.018, 0.022, 0.032,
                                            0.04, 0.1, 0.2, 0.201, 0.12, 0.04, 0.033, 0.027, 0.022,
                                            0.018, 0.016, 0.014, 0.02, 0.01, 0.01) )

# Combine tables back together
breast_cancer <- rbind(breast_cancer_raw_M, breast_cancer_raw_B)

# Delete all unneeded data
rm(breast_cancer_raw_M, breast_cancer_raw_B, breast_cancer_raw)

```

Most techniques in R require character levels to be coded as factors.

```{r Convert Variables}

# Conver variables to factor
breast_cancer$diagnosis <- as.factor(breast_cancer$diagnosis)
breast_cancer$race <- as.factor(breast_cancer$race)

# Some methods will require a numeric binary input
breast_cancer$diagnosis_1M_0B <- ifelse(breast_cancer$diagnosis=="M",1,0)

# Pull numeric variables into list for normalization
variables <- breast_cancer[!(names(breast_cancer) %in% c("id", "diagnosis", "race"))]

# Create function for normalization
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x))) }

# Normalize variables
breast_cancer_n <- as.data.frame(lapply(breast_cancer[3:32], normalize))

# Name new variables with a suffix '_n'
colnames(breast_cancer_n) <- paste0(colnames(breast_cancer_n), "_n")

# Combine new variables with old
breast_cancer <- cbind(breast_cancer, breast_cancer_n)

# Remove unnecessary objects
rm(breast_cancer_n, normalize)

```

Let's look a little closer at some of these variables...

```{r Additional Exploration}

# Display row percentages
prop.table(table(breast_cancer$race, breast_cancer$diagnosis), 1)

# Display column percentages
prop.table(table(breast_cancer$race, breast_cancer$diagnosis), 2)

# Display row percentages
age_diagnosis <- as.data.frame(prop.table(table(breast_cancer$age, breast_cancer$diagnosis), 1))
age_diagnosis <- age_diagnosis[age_diagnosis$Var2 == 'M',]

f <- list(
  family = "Courier New, monospace",
  size = 18,
  color = "#7f7f7f"
)

plot_ly(age_diagnosis, x = ~Var1, y = ~Freq, type = 'scatter', mode = 'lines') %>%
  layout(title = "Breast Cancer by Age",
         xaxis = list(
              title = "Age",
              titlefont = f), 
         yaxis = list(
              title = "% of Patients Malignant",
              titlefont = f))

# Display differences in perimeter_worst between groups
library(ggplot2)
ggplot(breast_cancer, aes(x=breast_cancer$perimeter_worst,
                          group=breast_cancer$diagnosis,
                          fill=breast_cancer$diagnosis)) +
geom_histogram(position="identity",binwidth=10, alpha = .5) + 
theme_bw() + 
xlab("Worst Perimeter") + 
ylab("Number of Patients") + 
ggtitle("Distribution of Cell Perimeter by Diagnosis Group") + 
guides(fill=guide_legend(title="Benign/Malignant"))

# Create Correlation plot
corrplot(cor(variables), method="circle", order = "AOE")

# Remove unnecessary objects
rm(age_diagnosis, variables)
```

Split Data into ~70% train ~30% validation

```{r Train and Validation Sets}

# Assign variable for Train/Test designation
breast_cancer$train_0_test_1 <- sample(0:1, nrow(breast_cancer), replace = TRUE, prob = c(0.7, 0.3))

# Create Train/Test datasets
breast_cancer_train <- breast_cancer[breast_cancer$train_0_test_1==0,]
breast_cancer_test <- breast_cancer[breast_cancer$train_0_test_1==1,]

```

First, let's look at a simple decision tree.  

A decision tree attempts to narrow down the groups at each split by asking binary questions, eventually
sorting all of the data into groups of predictions.

We'll try one with all of the measurements, then add race and age.

```{r I: Decision Tree}

# Basic Decision Tree with all cell measurements
model1 <- diagnosis ~ radius_mean + texture_mean + perimeter_mean + area_mean + smoothness_mean + 
  compactness_mean + concavity_mean + concave_points_mean + symmetry_mean + 
  fractal_dimension_mean + radius_se + texture_se + perimeter_se + area_se + 
  smoothness_se + compactness_se + concavity_se + concave_points_se + symmetry_se + 
  fractal_dimension_se + radius_worst + texture_worst + perimeter_worst + 
  area_worst + smoothness_worst + compactness_worst + concavity_worst + 
  concave_points_worst + symmetry_worst + fractal_dimension_worst

# Train the model
dtree1 <- ctree(model1, data = breast_cancer_train)

#Output confusion matrix on training data
table(predict(dtree1), breast_cancer_train$diagnosis)

# Display Decision Tree Specs
print(dtree1)

# Graph the tree, two different views
plot(dtree1)
plot(dtree1, type = "simple")

# Test model
test_dtree1 <- predict(dtree1, newdata=breast_cancer_test)

# Output test results as confusion matrix
table(test_dtree1, breast_cancer_test$diagnosis)

# Append results to master data set
breast_cancer_train <- cbind(breast_cancer_train, dtree1 = predict(dtree1))
breast_cancer_test <- cbind(breast_cancer_test, dtree1 = test_dtree1)

# Basic Decision Tree with all cell measurements, race and age added
model2 <- diagnosis ~ radius_mean + texture_mean + perimeter_mean + area_mean + smoothness_mean + 
  compactness_mean + concavity_mean + concave_points_mean + symmetry_mean + 
  fractal_dimension_mean + radius_se + texture_se + perimeter_se + area_se + 
  smoothness_se + compactness_se + concavity_se + concave_points_se + symmetry_se + 
  fractal_dimension_se + radius_worst + texture_worst + perimeter_worst + 
  area_worst + smoothness_worst + compactness_worst + concavity_worst + 
  concave_points_worst + symmetry_worst + fractal_dimension_worst + race + age

# Train the model
dtree2 <- ctree(model2, data = breast_cancer_train)

#Output confusion matrix on training data
table(predict(dtree2), breast_cancer_train$diagnosis)

# Display Decision Tree Specs
print(dtree2)

# Graph the tree, two different views
plot(dtree2)
plot(dtree2, type = "simple")

# Test model
test_dtree2<-predict(dtree2, newdata=breast_cancer_test)

# Output test results as confusion matrix
table(test_dtree2, breast_cancer_test$diagnosis)

# Check if two model predictions sets are same
all.equal(test_dtree1, test_dtree2)

# Remove unnecessary objects
rm(dtree1, dtree2, test_dtree1, test_dtree2)

```

Recursive Partitioning can occur in many different ways.  The decision trees above use permutation tests to statistically
determine variable importance.  Meanwhile, the rpart procedure may be more biased toward poly level categorial variables.

```{r II: Recursive Partitioning}

# Train the model using the model above, including race and age
rp <- rpart(formula = model2, method = 'class', data = breast_cancer_train)

# Plots the rp map
rpart.plot(rp)

# Displays Model Specs
printcp(rp)

# Output predictions for training set
pred = predict(rp, type = 'class')

# Output predictions for test set
pred_new = predict(rp, newdata=breast_cancer_test, type = 'class')

# Confusion Matrices for Train/Test
table(pred, breast_cancer_train$diagnosis)
table(pred_new, breast_cancer_test$diagnosis)

# Append results to master data set
breast_cancer_train <- cbind(breast_cancer_train, rp = pred)
breast_cancer_test <- cbind(breast_cancer_test, rp = pred_new)

# Remove unnecessary objects
rm(pred, pred_new, rp)
```

One of the major critiques of Recursive Partitioning and Decision Trees is the inherent bias of the training set itself,
leading to a tendency to overfit.  One way to address this concern is through use of the random forest technique, which 
creates a multitude of decision trees during training time and uses them together for prediction.

```{r III: Random Forest}

# Train random forest on model 1 (no race and age)
rf1 <- randomForest(model1,
                      data=breast_cancer_train, 
                      importance=TRUE, 
                      ntree=2000)

# Display Variable Importance Plot
varImpPlot(rf1)

# Predict outcome for test set
rf1_pred <- predict(rf1, breast_cancer_test)

# Confusion Matrix for Test
table(rf1_pred, breast_cancer_test$diagnosis)

# Train random forest on model 2 (with race and age)
rf2 <- randomForest(model2,
                      data=breast_cancer_train, 
                      importance=TRUE, 
                      ntree=2000)

# Display Variable Importance Plot
varImpPlot(rf2)

# Predict outcome for test set
rf2_pred <- predict(rf2, breast_cancer_test)

# Confusion Matrix for Test
table(rf2_pred, breast_cancer_test$diagnosis)

# Check if two model predictions sets are same
all.equal(rf1_pred, rf2_pred)

# Append results to master data set
breast_cancer_train$rf1_pred <- 'NA'
breast_cancer_test <- cbind(breast_cancer_test, rf1_pred)

# Remove unnecessary objects
rm(rf1, rf2, rf1_pred, rf2_pred)

```

Logistic Regression predicts the probability of an individual falling into a certain class, as determined by its values of the
independent variables. 

Generally best to rebuild this model, to avoid quasi-complete separation.

```{r IV: Logistic Regression}

# Unlike Recursive Partioning methods, Logistic Regression cannot accept missing values.
sapply(breast_cancer_train, function(x) sum(is.na(x)))
sapply(breast_cancer_train, function(x) length(x))
sapply(breast_cancer_train, function(x) length(unique(x)))

# Create Logistic Regression model (Race had no effect so left out)
model_logreg1 <- glm(breast_cancer_train$diagnosis_1M_0B ~ 
                    breast_cancer_train$concave_points_mean +
                    breast_cancer_train$concave_points_worst +
                    breast_cancer_train$perimeter_worst +
                    breast_cancer_train$radius_worst +
                    # breast_cancer_train$race +
                    breast_cancer_train$age
                    ,family = binomial(link = 'logit'), data=breast_cancer_train, control = list(maxit = 50))

# Output Model Specs
summary(model_logreg1)

# Analysis of Variance
anova(model_logreg1, test="Chisq")

# Collect raw probabilities from training set
fitted.results_tr_raw <- predict(model_logreg1, type='response')

# Round probablities to 1 or 0
fitted.results_tr <- ifelse(fitted.results_tr_raw > 0.5,1,0)

# Calculate and print misclassification
misClasificError <- mean(fitted.results_tr != breast_cancer_train$diagnosis_1M_0B)
print(paste('Accuracy',1-misClasificError))

# Collect raw probabilities from test set
fitted.results_tst_raw <- predict(model_logreg1, newdata=list(breast_cancer_train=breast_cancer_test), type='response')

# Round probablities to 1 or 0
fitted.results_tst <- ifelse(fitted.results_tst_raw > 0.5,1,0)

# Calculate and print misclassification
misClasificError <- mean(fitted.results_tst != breast_cancer_test$diagnosis_1M_0B)
print(paste('Accuracy',1-misClasificError))

# Append results to master data set
breast_cancer_train <- cbind(breast_cancer_train, glm_raw = fitted.results_tr_raw, glm = fitted.results_tr)
breast_cancer_test <- cbind(breast_cancer_test, glm_raw = fitted.results_tst_raw, glm = fitted.results_tst)

# Recode numeric results to M / B
breast_cancer_train$glm <- ifelse(breast_cancer_train$glm == 1, 'M', 'B')
breast_cancer_test$glm <- ifelse(breast_cancer_test$glm == 1, 'M', 'B')

# Remove unnecessary objects
rm(fitted.results_tr, fitted.results_tr_raw, fitted.results_tst, fitted.results_tst_raw, misClasificError, model_logreg1)
```

Partitions data into k many clusters.  An observation's classification in a given cluster is determined by the distance from that point to the cluster's center.

```{r V: K-Nearest Neighbor}

# Selecting only the normalized variables
bc_knn_train <- cbind(diagnosis = breast_cancer_train$diagnosis, subset(breast_cancer_train, 
                             select = grep("_n", names(breast_cancer_train))))
bc_knn_test <- subset(breast_cancer_test, 
                             select = grep("_n", names(breast_cancer_test)))

# Iterate through solutions with 1 to 25 clusters
model_knn <- train(diagnosis ~ ., data = bc_knn_train, method = 'knn',
    tuneGrid = expand.grid(.k = 1:25), metric = 'Accuracy',
    trControl = trainControl(method = 'repeatedcv', number = 10, repeats = 15))

# Plot Accuracy by number of clusters (7 is best)
plot(model_knn)

# Remove ID
bc_knn_train <- bc_knn_train[c(-1)]

# Predict a 7 cluster solution
knn.7 <-  knn(bc_knn_train, bc_knn_test, breast_cancer_train$diagnosis, k=7, prob=TRUE)

# Confusion Matrix of Results
table(knn.7 ,breast_cancer_test$diagnosis)

# Append results to master data set
breast_cancer_train$knn.7 <- 'NA'
breast_cancer_test <- cbind(breast_cancer_test, knn.7)

# Remove unnecessary objects
rm(bc_knn_test, bc_knn_train, knn.7, model_knn)

```


```{r Results}

# Join training and test reults back together
breast_cancer <- rbind(breast_cancer_train, breast_cancer_test)

# Create subset of results
breast_cancer_results <- breast_cancer_test[c('id', 'diagnosis', 'race', 'age', 'dtree1', 'rp', 
                                              'glm_raw', 'glm', 'knn.7', 'rf1_pred')]

# Create _num column: 1 if diagnosis accurate, else 0
breast_cancer_results$dtree1_num <- ifelse(breast_cancer_results$dtree1 == breast_cancer_results$diagnosis, 1, 0)
breast_cancer_results$rp_num <- ifelse(breast_cancer_results$rp == breast_cancer_results$diagnosis, 1, 0)
breast_cancer_results$glm_num <- ifelse(breast_cancer_results$glm == breast_cancer_results$diagnosis, 1, 0)
breast_cancer_results$knn.7_num <- ifelse(breast_cancer_results$knn.7 == breast_cancer_results$diagnosis, 1, 0)
breast_cancer_results$rf1_pred_num <- ifelse(breast_cancer_results$rf1_pred == breast_cancer_results$diagnosis, 1, 0)

# Subset test patients with cancer
breast_cancer_results_M <- breast_cancer_results[which(breast_cancer_results$diagnosis=='M'),]

# Create accuracy %
breast_cancer_results$dtree1_acc <- percent(sum(breast_cancer_results$dtree1_num)/nrow(breast_cancer_results))
breast_cancer_results$rp_acc <- percent(sum(breast_cancer_results$rp_num)/nrow(breast_cancer_results))
breast_cancer_results$glm_acc <- percent(sum(breast_cancer_results$glm_num)/nrow(breast_cancer_results))
breast_cancer_results$knn.7_acc <- percent(sum(breast_cancer_results$knn.7_num)/nrow(breast_cancer_results))
breast_cancer_results$rf1_pred_acc <- percent(sum(breast_cancer_results$rf1_pred_num)/nrow(breast_cancer_results))

# Create _num column: 1 if cancer predicted, else 0
breast_cancer_results$dtree1_num <- ifelse(breast_cancer_results$dtree1 == 'M', 1, 0)
breast_cancer_results$rp_num <- ifelse(breast_cancer_results$rp == 'M', 1, 0)
breast_cancer_results$glm_num <- ifelse(breast_cancer_results$glm == 'M', 1, 0)
breast_cancer_results$knn.7_num <- ifelse(breast_cancer_results$knn.7 == 'M', 1, 0)
breast_cancer_results$rf1_pred_num <- ifelse(breast_cancer_results$rf1_pred == 'M', 1, 0)

# Add up _num to get a consensus number.  Use probability for glm.
breast_cancer_results$decision_sum <- (breast_cancer_results$dtree1_num +
                                       # breast_cancer_results$rp_num +
                                       breast_cancer_results$glm_raw * breast_cancer_results$glm_num +
                                       breast_cancer_results$knn.7_num +
                                       breast_cancer_results$rf1_pred_num)

# Remove unnecessary columns
breast_cancer_results$dtree1_num <- NULL
breast_cancer_results$rp_num <- NULL
breast_cancer_results$glm_num <- NULL
breast_cancer_results$knn.7_num <- NULL
breast_cancer_results$rf1_pred_num <- NULL

# Create risk and diagnosis.pred variables based on ensemble
breast_cancer_results <- breast_cancer_results %>% 
  # mutate(risk = cut(decision_sum, c(-1, 2, 3.5, 4.5, 6), 
  mutate(risk = cut(decision_sum, c(-1, 1.6, 2.8, 3.6, 6), 
                    labels = c("LOW", "MEDIUM", "HIGH", "VERY HIGH"))) %>% 
  mutate(diagnosis.pred = ifelse(decision_sum < .4, "B", "M"))

# Calculate ensemble model accuracy
breast_cancer_results$pred_num <- ifelse(breast_cancer_results$diagnosis.pred == breast_cancer_results$diagnosis, 1, 0)
breast_cancer_results$correct <- ifelse(breast_cancer_results$diagnosis.pred == breast_cancer_results$diagnosis, 
                                        'CORRECT', 'INCORRECT')
breast_cancer_results$ensemble_acc <- percent(sum(breast_cancer_results$pred_num)/nrow(breast_cancer_results))

# Remove unnecessary column
breast_cancer_results$pred_num <- NULL

# Remove unnecessary objects
rm(model1, model2, breast_cancer_test, breast_cancer_train)

```


```{r Results Continued}

# Create accuracy %
breast_cancer_results_M$dtree1_acc <- percent(sum(breast_cancer_results_M$dtree1_num)/nrow(breast_cancer_results_M))
breast_cancer_results_M$rp_acc <- percent(sum(breast_cancer_results_M$rp_num)/nrow(breast_cancer_results_M))
breast_cancer_results_M$glm_acc <- percent(sum(breast_cancer_results_M$glm_num)/nrow(breast_cancer_results_M))
breast_cancer_results_M$knn.7_acc <- percent(sum(breast_cancer_results_M$knn.7_num)/nrow(breast_cancer_results_M))
breast_cancer_results_M$rf1_pred_acc <- percent(sum(breast_cancer_results_M$rf1_pred_num)/nrow(breast_cancer_results_M))

#Create _num column: 1 if cancer predicted, else 0
breast_cancer_results_M$dtree1_num <- ifelse(breast_cancer_results_M$dtree1 == 'M', 1, 0)
breast_cancer_results_M$rp_num <- ifelse(breast_cancer_results_M$rp == 'M', 1, 0)
breast_cancer_results_M$glm_num <- ifelse(breast_cancer_results_M$glm == 'M', 1, 0)
breast_cancer_results_M$knn.7_num <- ifelse(breast_cancer_results_M$knn.7 == 'M', 1, 0)
breast_cancer_results_M$rf1_pred_num <- ifelse(breast_cancer_results_M$rf1_pred == 'M', 1, 0)

# Add up _num to get a consensus number.  Use probability for glm.
breast_cancer_results_M$decision_sum <- (breast_cancer_results_M$dtree1_num +
                                       # breast_cancer_results_M$rp_num +
                                       breast_cancer_results_M$glm_raw * breast_cancer_results_M$glm_num +
                                       breast_cancer_results_M$knn.7_num +
                                       breast_cancer_results_M$rf1_pred_num)

# Remove unnecessary columns
breast_cancer_results_M$dtree1_num <- NULL
breast_cancer_results_M$rp_num <- NULL
breast_cancer_results_M$glm_num <- NULL
breast_cancer_results_M$knn.7_num <- NULL
breast_cancer_results_M$rf1_pred_num <- NULL

# Create risk and diagnosis.pred variables based on ensemble
breast_cancer_results_M <- breast_cancer_results_M %>% 
  # mutate(risk = cut(decision_sum, c(-1, 2, 3.5, 4.5, 6), 
  mutate(risk = cut(decision_sum, c(-1, 1.6, 2.8, 3.6, 6), 
                    labels = c("LOW", "MEDIUM", "HIGH", "VERY HIGH"))) %>% 
  mutate(diagnosis.pred = ifelse(decision_sum < .4, "B", "M"))

# Calculate ensemble model accuracy
breast_cancer_results_M$pred_num <- ifelse(breast_cancer_results_M$diagnosis.pred == breast_cancer_results_M$diagnosis, 1, 0)
breast_cancer_results_M$correct <- ifelse(breast_cancer_results_M$diagnosis.pred == breast_cancer_results_M$diagnosis, 
                                        'CORRECT', 'INCORRECT')
breast_cancer_results_M$ensemble_acc <- percent(sum(breast_cancer_results_M$pred_num)/nrow(breast_cancer_results_M))

# Remove unnecessary column
breast_cancer_results_M$pred_num <- NULL

# Subset test patients predicted incorrectly by any model
breast_cancer_results_wrong <- breast_cancer_results_M[which(breast_cancer_results_M$decision_sum < 3.4),]

# Compare model performance
head(breast_cancer_results[c('dtree1_acc', 'rp_acc', 'glm_acc', 'knn.7_acc', 'rf1_pred_acc', 'ensemble_acc')], 1)
head(breast_cancer_results_M[c('dtree1_acc', 'rp_acc', 'glm_acc', 'knn.7_acc', 'rf1_pred_acc', 'ensemble_acc')], 1)

```
