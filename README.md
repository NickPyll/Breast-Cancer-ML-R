# Breast-Cancer-ML-R
Introduction to Machine Learning Techniques in R Using Breast Cancer Data

Abstract: This document will provide an overview of some popular machine learning techniques, as well as a typical project flow for a machine learning project.  

Author: Nicholas Pylypiw--Cardinal Solutions

Data Source: https://www.kaggle.com/uciml/breast-cancer-wisconsin-data

Title: Machine Learning and Breast Cancer Prediction

This case study will require the following packages.

```{r Load Packages}

# install.packages("readr")
# install.packages("rpart")
# install.packages("rpart.plot")
# install.packages("party")
# install.packages("ggplot2")
# install.packages("corrplot")
# install.packages("plyr")
# install.packages("plotly")
# install.packages("class")
# install.packages("randomForest")
# install.packages("e1071", dependencies=TRUE)
# install.packages("caret")
# install.packages("dplyr")
# install.packages("scales")
# install.packages("pROC")
# install.packages("partykit")
# install.packages("tidyr"")

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
library(pROC)
library(partykit)
library(tidyr)

```

Load the .csv into the R environment as a data frame.  
-- str() can be used to view the structure of the object, as well as variable types
-- head(data, n) provides the first n rows of the data frame.

```{r Import Data}

# read csv
breast.cancer.raw <- readr::read_csv("~/Projects/Breast Cancer/breast_cancer.csv")

# Look at first 6 rows
head(breast.cancer.raw)

```

Notice in the output from head() that a few variables have spaces in their names.  This naming convention is not 
compatible with many of the R procdures used in this tutorial, so those will have to be changed.

```{r Rename Columns}

# Replaces spaces and underscores in variable names with an underscore
names(breast.cancer.raw) <- gsub(" ", ".", names(breast.cancer.raw))
names(breast.cancer.raw) <- gsub("_", ".", names(breast.cancer.raw))

# Order Columns by Alpha
breast.cancer.raw <- breast.cancer.raw[, order(names(breast.cancer.raw))]

# Moving ID  and Diagnosis variables
ID <- c("diagnosis", "id")
for (i in 1:2){
  breast.cancer.raw <-  breast.cancer.raw[,c(ID[i],setdiff(names(breast.cancer.raw),ID[i]))]
}

# Remove unnecessary objects
rm(ID, i)
```

Now the data is ready to be explored a bit.

```{r Explore Data}

# Mean and Five Number Summary for each variable
variable.summary <- data.frame(do.call(cbind, lapply(breast.cancer.raw[c(-1,-2)], summary)))

# Quick count of cancer rate in data set
table(breast.cancer.raw$diagnosis)

# Looking at distribution for area.mean variable
hist(breast.cancer.raw$area.mean, 
     main = 'Distribution of Cell Area Means',
     xlab = 'Mean Area',
     col = 'green')

```

No race or age variables?!!!!!  Well that's boring... I'll just create them.
Using http://ww5.komen.org/BreastCancer/RaceampEthnicity.html, I created relative probabilities of risk by race.
I just guessed at age probabilities, increasing as patient gets older.

```{r Add Demographics}

# Set seed to your favorite number for replication
set.seed(8675309)

# Divide datasets by diagnosis
breast.cancer.raw.M <- breast.cancer.raw[which(breast.cancer.raw$diagnosis=='M'),]
breast.cancer.raw.B <- breast.cancer.raw[which(breast.cancer.raw$diagnosis=='B'),]

# Assign risk probabilities by race
breast.cancer.raw.M$race <- sample( c('White', 'Black', 'Asian', 'Hispanic', 'Other'), 
                                    nrow(breast.cancer.raw.M), 
                                    replace = TRUE, 
                                    prob = c(.41, .31, .11, 0.14, .03) )
breast.cancer.raw.B$race <- sample( c('White', 'Black', 'Asian', 'Hispanic', 'Other'), 
                                    nrow(breast.cancer.raw.B), 
                                    replace = TRUE, 
                                    prob = c(.28, .28, .18, 0.20, .06) )

# Assign risk probabilities by age
breast.cancer.raw.M$age <- sample( 18:40, 
                                   size = nrow(breast.cancer.raw.M), 
                                   replace = TRUE, 
                                   prob = c(0.005, 0.005, 0.006, 0.006, 0.009, 0.012, 0.016, 0.022, 
                                            0.025, 0.08, 0.17, 0.19, 0.14, 0.044, 0.038, 0.029, 0.027,
                                            0.024, 0.021, 0.028, 0.042, 0.03, 0.031) )
breast.cancer.raw.B$age <- sample( 18:40, 
                                   size = nrow(breast.cancer.raw.B), 
                                   replace = TRUE, 
                                   prob = c(0.01, 0.01, 0.01, 0.012, 0.015, 0.018, 0.022, 0.032,
                                            0.04, 0.1, 0.2, 0.201, 0.12, 0.04, 0.033, 0.027, 0.022,
                                            0.018, 0.016, 0.014, 0.02, 0.01, 0.01) )

# Combine tables back together
breast.cancer <- rbind(breast.cancer.raw.M, breast.cancer.raw.B)

# Moving variables
ID <- c("diagnosis", "age", "race", "id")
for (i in 1:4){
  breast.cancer <-  breast.cancer[,c(ID[i],setdiff(names(breast.cancer),ID[i]))]
}

# Delete all unneeded data
rm(breast.cancer.raw.M, breast.cancer.raw.B, breast.cancer.raw, ID, i)

```

Most techniques in R require character levels to be coded as factors.

```{r Convert Variables}

# Convert variables to factor
breast.cancer$diagnosis <- as.factor(breast.cancer$diagnosis)
breast.cancer$race <- as.factor(breast.cancer$race)

# Some methods will require a numeric binary input
breast.cancer$diagnosis.1M.0B <- ifelse(breast.cancer$diagnosis == "M", 1, 0)

# Create function for normalization
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x))) 
}

# Normalize variables
breast.cancer.n <- as.data.frame(lapply(breast.cancer[5:34], normalize))

# Name new variables with a suffix '.n'
colnames(breast.cancer.n) <- paste0(colnames(breast.cancer.n), ".n")

# Order Columns by Alpha
breast.cancer.n <- breast.cancer.n[, order(names(breast.cancer.n))]

# Pull numeric variables into list for correlation calculations
variables <- breast.cancer[!(names(breast.cancer) %in% c("id", "diagnosis", "race"))]

# Order Columns by Alpha
breast.cancer.n <- breast.cancer.n[, order(names(breast.cancer.n))]

# Combine new variables with old
breast.cancer <- cbind(breast.cancer, breast.cancer.n)

# Moving variables
ID <- c("diagnosis.1M.0B", "diagnosis", "age", "race", "id")
for (i in 1:5){
  breast.cancer <-  breast.cancer[,c(ID[i],setdiff(names(breast.cancer),ID[i]))]
}

# Remove unnecessary objects
rm(breast.cancer.n, normalize, ID, i)

```

Let's look a little closer at some of these variables...

```{r Additional Exploration}

# Display row percentages
prop.table(table(breast.cancer$race, breast.cancer$diagnosis), 1)

# Display column percentages
prop.table(table(breast.cancer$race, breast.cancer$diagnosis), 2)

# Display % Malignant by Age
age.diagnosis <- as.data.frame(prop.table(table(breast.cancer$age, breast.cancer$diagnosis), 1))
age.diagnosis <- age.diagnosis[age.diagnosis$Var2 == 'M',]

windowsFonts(Times=windowsFont("Times New Roman"))
ggplot2::ggplot() +
  geom_line(aes(y = Freq, x = Var1, group = 1), size=1, data = age.diagnosis) +
  theme(legend.position="bottom", legend.direction="horizontal",
        legend.title = element_blank()) +
  labs(x="Age in Years", y="Percentage of Malignance") +
  ggtitle("Percentage of Malignant Patients by Age") +
  scale_color_manual(values=fill) +
  theme(axis.line = element_line(size=1, colour = "black"), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), panel.border = element_blank(),
        panel.background = element_blank()) +
  theme(plot.title=element_text(family="Times"), text=element_text(family="Times"),
        axis.text.x=element_text(colour="black", size = 9),
        axis.text.y=element_text(colour="black", size = 9),
        legend.key=element_rect(fill="white", colour="white"),
        plot.margin = unit(c(1, 1, 1, 1), "in"))

# Display differences in perimeter.worst between groups
library(ggplot2)
ggplot(breast.cancer, aes(x=breast.cancer$perimeter.worst,
                          group=breast.cancer$diagnosis,
                          fill=breast.cancer$diagnosis)) +
geom_histogram(position="identity",binwidth=10, alpha = .5) + 
theme_bw() + 
xlab("Worst Perimeter") + 
ylab("Number of Patients") + 
ggtitle("Distribution of Cell Perimeter by Diagnosis Group") + 
guides(fill=guide_legend(title="Benign/Malignant"))

# Create Correlation plot
corrplot(cor(variables), method="circle", order = "AOE")

# Remove unnecessary objects
rm(age.diagnosis, variables, variable.summary)
```

Split Data into ~70% train ~30% validation

```{r Train and Validation Sets}

# Assign variable for Train/Test designation
breast.cancer$train.0.test.1 <- sample(0:1, nrow(breast.cancer), replace = TRUE, prob = c(0.7, 0.3))

# Create Train/Test datasets
bc.train <- breast.cancer[breast.cancer$train.0.test.1==0,]
bc.test <- breast.cancer[breast.cancer$train.0.test.1==1,]

# Create results datasets
results.train <- bc.train[c('id', 'race', 'age', 'diagnosis.1M.0B', 'diagnosis')]
results.test <- bc.test[c('id', 'race', 'age', 'diagnosis.1M.0B', 'diagnosis')]

```

Recursive partitioning is a quick and easy way to predict categorical variables.  However, the rpart procedure
in R may be biased toward poly level categorical variables

```{r I: Recursive Partitioning}

# Basic Model with all cell measurements
model1 <- diagnosis ~ radius.mean + texture.mean + perimeter.mean + area.mean + smoothness.mean + 
  compactness.mean + concavity.mean + concave.points.mean + symmetry.mean + 
  fractal.dimension.mean + radius.se + texture.se + perimeter.se + area.se + 
  smoothness.se + compactness.se + concavity.se + concave.points.se + symmetry.se + 
  fractal.dimension.se + radius.worst + texture.worst + perimeter.worst + 
  area.worst + smoothness.worst + compactness.worst + concavity.worst + 
  concave.points.worst + symmetry.worst + fractal.dimension.worst

# Train the model using the model above, not including race and age
rec.part1 <- rpart(formula = model1, method = 'class', data = bc.train)

# Plots the rp map
rpart.plot(rec.part1)

# Displays Model Specs
printcp(rec.part1)

# Output predictions for training set
pred1 = predict(rec.part1, type = 'class')

# Output predictions for test set
pred.new1 = predict(rec.part1, newdata=bc.test, type = 'class')

# Confusion Matrices for Train/Test
table(pred1, bc.train$diagnosis)
table(pred.new1, bc.test$diagnosis)

# Basic Model with all cell measurements, race and age added
model2 <- diagnosis ~ radius.mean + texture.mean + perimeter.mean + area.mean + smoothness.mean + 
  compactness.mean + concavity.mean + concave.points.mean + symmetry.mean + 
  fractal.dimension.mean + radius.se + texture.se + perimeter.se + area.se + 
  smoothness.se + compactness.se + concavity.se + concave.points.se + symmetry.se + 
  fractal.dimension.se + radius.worst + texture.worst + perimeter.worst + 
  area.worst + smoothness.worst + compactness.worst + concavity.worst + 
  concave.points.worst + symmetry.worst + fractal.dimension.worst + race + age

# Train the model using the model above, including race and age
rec.part2 <- rpart(formula = model2, method = 'class', data = bc.train)

# Plots the rp map
rpart.plot(rec.part2)

# Displays Model Specs
printcp(rec.part2)

# Output predictions for training set
pred2 = predict(rec.part2, type = 'class')

# Output predictions for test set
pred.new2 = predict(rec.part2, newdata=bc.test, type = 'class')

# Confusion Matrices for Train/Test
table(pred2, bc.train$diagnosis)
table(pred.new2, bc.test$diagnosis)

# Check if two model predictions sets are same
all.equal(pred.new1, pred.new2)

# Append results to results data sets
results.train <- cbind(results.train, rec.part1 = pred1)
results.test <- cbind(results.test, rec.part1 = pred.new1)

# Remove unnecessary objects
rm(pred1, pred2, pred.new1, pred.new2, rec.part1, rec.part2)

```

Next, let's look at a simple decision tree.  

A decision tree attempts to narrow down the groups at each split by asking binary questions, eventually
sorting all of the data into groups of predictions.

Unlike the general recursive partitioning method above, decision trees above use permutation tests to statistically
determine variable importance.

We'll try one with all of the measurements, then add race and age.

```{r II: Decision Tree}

# Train the model
dec.tree1 <- partykit::ctree(model1, data = bc.train)

# Output confusion matrix on training data
table(predict(dec.tree1), bc.train$diagnosis)

# Display Decision Tree Specs
print(dec.tree1)

# Graph the tree, two different views
plot(dec.tree1, gp = gpar(fontsize = 8))
plot(dec.tree1, type = "simple", gp = gpar(fontsize = 8))

# Test model
test.dec.tree1 <- predict(dec.tree1, newdata=bc.test)

# Output test results as confusion matrix
table(test.dec.tree1, bc.test$diagnosis)

# Train the model
dec.tree2 <- partykit::ctree(model2, data = bc.train)

# Output confusion matrix on training data
table(predict(dec.tree2), bc.train$diagnosis)

# Display Decision Tree Specs
print(dec.tree2)

# Graph the tree, two different views
plot(dec.tree2, gp = gpar(fontsize = 8))
plot(dec.tree2, type = "simple", gp = gpar(fontsize = 8))

# Test model
test.dec.tree2<-predict(dec.tree2, newdata=bc.test)

# Output test results as confusion matrix
table(test.dec.tree2, bc.test$diagnosis)

# Check if two model predictions sets are same
all.equal(test.dec.tree1, test.dec.tree2)

# Append results to results data set
results.train <- cbind(results.train, dec.tree1 = predict(dec.tree1))
results.test <- cbind(results.test, dec.tree1 = test.dec.tree1)

# Remove unnecessary objects
rm(dec.tree1, dec.tree2, test.dec.tree1, test.dec.tree2)

```

One of the major critiques of Recursive Partitioning and Decision Trees is the inherent bias of the training set itself,
leading to a tendency to overfit.  One way to address this concern is through use of the random forest technique, which 
creates a multitude of decision trees during training time and uses them together for prediction.

```{r III: Random Forest}

# Train random forest on model 1 (no race and age)
rand.for1 <- randomForest(model1,
                      data=bc.train, 
                      importance=TRUE, 
                      ntree=2000)

# Display Variable Importance Plot
varImpPlot(rand.for1)

# Predict outcome for test set
rand.for1.pred <- predict(rand.for1, bc.test)

# Confusion Matrix for Test
table(rand.for1.pred, bc.test$diagnosis)

# Train random forest on model 2 (with race and age)
rand.for2 <- randomForest(model2,
                      data=bc.train, 
                      importance=TRUE, 
                      ntree=2000)

# Display Variable Importance Plot
varImpPlot(rand.for2)

# Predict outcome for test set
rand.for2.pred <- predict(rand.for2, bc.test)

# Confusion Matrix for Test
table(rand.for2.pred, bc.test$diagnosis)

# Check if two model predictions sets are same
all.equal(rand.for1.pred, rand.for2.pred)

# Append results to results data set
results.train$rand.for1 <- 'NA'
results.test <- cbind(results.test, rand.for1 = rand.for1.pred)

# Remove unnecessary objects
rm(rand.for1, rand.for2, rand.for1.pred, rand.for2.pred)

```

Logistic Regression predicts the probability of an individual falling into a certain class, as determined by 
its values of the independent variables. 

Generally best to rebuild this model, to avoid quasi-complete separation.

```{r IV: Logistic Regression}

# Unlike Recursive Partioning methods, Logistic Regression cannot accept missing values.
sapply(bc.train, function(x) sum(is.na(x)))
sapply(bc.train, function(x) length(x))
sapply(bc.train, function(x) length(unique(x)))

# Create Logistic Regression model (Race had no effect so left out)
model.logreg1 <- glm(bc.train$diagnosis.1M.0B ~ 
                    bc.train$concave.points.mean +
                    bc.train$concave.points.worst +
                    bc.train$perimeter.worst +
                    bc.train$radius.worst +
                    # bc.train$race +
                    bc.train$age
                    ,family = binomial(link = 'logit'), data=bc.train, control = list(maxit = 50))



# Output Model Specs
summary(model.logreg1)

# Analysis of Variance
anova(model.logreg1, test = "Chisq")

# Collect raw probabilities from training set
fitted.results.tr.raw <- predict(model.logreg1, type = 'response')

# Round probablities to 1 or 0
fitted.results.tr <- ifelse(fitted.results.tr.raw > 0.5, 1, 0)

# Calculate and print misclassification
misClasificError <- mean(fitted.results.tr != bc.train$diagnosis.1M.0B)
print(paste('Accuracy', 1 - misClasificError))

# Collect raw probabilities from test set
fitted.results.tst.raw <- predict(model.logreg1, 
                                  newdata = list(bc.train = bc.test), 
                                  type = 'response')

# Round probablities to 1 or 0
fitted.results.tst <- ifelse(fitted.results.tst.raw > 0.5, 1, 0)

# Calculate and print misclassification
misClasificError <- mean(fitted.results.tst != bc.test$diagnosis.1M.0B)
print(paste('Accuracy',1 - misClasificError))

# Append results to results data set
results.train <- cbind(results.train, log.reg.p = fitted.results.tr.raw, log.reg = fitted.results.tr)
results.test <- cbind(results.test, log.reg.p = fitted.results.tst.raw, log.reg = fitted.results.tst)

# ROC Curve
plot(roc(diagnosis.1M.0B ~ log.reg.p, data = results.train))  
plot(roc(diagnosis.1M.0B ~ log.reg.p, data = results.test))  

# Recode numeric results to M / B
results.train$log.reg <- ifelse(results.train$log.reg == 1, 'M', 'B')
results.test$log.reg <- ifelse(results.test$log.reg == 1, 'M', 'B')

# Remove unnecessary objects
rm(fitted.results.tr, fitted.results.tr.raw, fitted.results.tst, fitted.results.tst.raw, misClasificError, model.logreg1)

```

Partitions data into k many clusters.  An observation's classification in a given cluster is determined by the distance from that point to the cluster's center.

```{r V: K-Nearest Neighbor}

# Selecting only the normalized variables
bc.knn.train <- cbind(diagnosis = bc.train$diagnosis, subset(bc.train, 
                             select = grep("\\.n", names(bc.train))))
bc.knn.test <- subset(bc.test, 
                             select = grep("\\.n", names(bc.test)))

# Iterate through solutions with 1 to 25 clusters
model.knn <- train(diagnosis ~ ., data = bc.knn.train, method = 'knn',
    tuneGrid = expand.grid(.k = 1:25), metric = 'Accuracy',
    trControl = trainControl(method = 'repeatedcv', number = 10, repeats = 15))

# Plot Accuracy by number of clusters (7 is best)
plot(model.knn)

# Remove ID
bc.knn.train <- bc.knn.train[c(-1)]

# Predict a 7 cluster solution
knn.7 <-  knn(bc.knn.train, bc.knn.test, bc.train$diagnosis, k=7, prob=TRUE)

# Confusion Matrix of Results
table(knn.7 ,bc.test$diagnosis)

# Append results to master data set
results.train$knn.7 <- 'NA'
results.test <- cbind(results.test, knn.7)

# Remove unnecessary objects
rm(bc.knn.test, bc.knn.train, knn.7, model.knn)

```
Let's join train and test back together and do some post-modeling data manipulation.


```{r Results}

# Create .num column: 1 if diagnosis accurate, else 0
results.test$rec.part1.num <- ifelse(results.test$rec.part1 == results.test$diagnosis, 1, 0)
results.test$dec.tree1.num <- ifelse(results.test$dec.tree1 == results.test$diagnosis, 1, 0)
results.test$rand.for1.num <- ifelse(results.test$rand.for1 == results.test$diagnosis, 1, 0)
results.test$log.reg.num <- ifelse(results.test$log.reg == results.test$diagnosis, 1, 0)
results.test$knn.7.num <- ifelse(results.test$knn.7 == results.test$diagnosis, 1, 0)

# Subset test patients by diagnosis
results.test.M <- results.test[which(results.test$diagnosis=='M'),]
results.test.B <- results.test[which(results.test$diagnosis=='B'),]

# Create accuracy %
rec.part1.acc <- percent(sum(results.test$rec.part1.num)/nrow(results.test))
dec.tree1.acc <- percent(sum(results.test$dec.tree1.num)/nrow(results.test))
rand.for1.acc <- percent(sum(results.test$rand.for1.num)/nrow(results.test))
log.reg.acc <- percent(sum(results.test$log.reg.num)/nrow(results.test))
knn.7.acc <- percent(sum(results.test$knn.7.num)/nrow(results.test))
group <- 'Overall'

accuracy.comparison <- data.frame(group,rec.part1.acc,dec.tree1.acc,rand.for1.acc,log.reg.acc,knn.7.acc)

# Create .num column: 1 if cancer predicted, else 0
results.test$rec.part1.num <- ifelse(results.test$rec.part1 == 'M', 1, 0)
results.test$dec.tree1.num <- ifelse(results.test$dec.tree1 == 'M', 1, 0)
results.test$rand.for1.num <- ifelse(results.test$rand.for1 == 'M', 1, 0)
results.test$log.reg.num <- ifelse(results.test$log.reg == 'M', 1, 0)
results.test$knn.7.num <- ifelse(results.test$knn.7 == 'M', 1, 0)

# Add up .num to get a consensus number.  Use probability for log.reg.
results.test$decision.sum <- (# results.test$rec.part1.num +
                                       results.test$dec.tree1.num +
                                       results.test$rand.for1.num +
                                       results.test$log.reg.p +
                                       results.test$knn.7.num)

# Remove unnecessary columns
results.test$rec.part1.num <- NULL
results.test$dec.tree1.num <- NULL
results.test$rand.for1.num <- NULL
results.test$log.reg.num <- NULL
results.test$knn.7.num <- NULL

# Create risk and diagnosis.pred variables based on ensemble
results.test <- results.test %>% 
  # mutate(risk = cut(decision.sum, c(-1, 2, 3.5, 4.5, 6), 
  mutate(risk = cut(decision.sum, c(-1, 1.6, 2.8, 3.6, 6), 
                    labels = c("LOW", "MEDIUM", "HIGH", "VERY HIGH"))) %>% 
  mutate(diagnosis.pred = ifelse(decision.sum < .4, "B", "M"))

# Calculate ensemble model accuracy
results.test$pred.num <- ifelse(results.test$diagnosis.pred == results.test$diagnosis, 1, 0)
results.test$correct <- ifelse(results.test$diagnosis.pred == results.test$diagnosis, 
                                        'CORRECT', 'INCORRECT')
accuracy.comparison$ensemble.acc <- percent(sum(results.test$pred.num)/nrow(results.test))

accuracy.comparison <- head(accuracy.comparison)

# Remove unnecessary column
results.test$pred.num <- NULL

# Remove unnecessary objects
rm(model1, model2, bc.test, bc.train)
rm(group,rec.part1.acc,dec.tree1.acc,rand.for1.acc,log.reg.acc,knn.7.acc)

```
Doing the same as above, but by diagnosis.

```{r Results Continued}

# Create accuracy % for M
rec.part1.acc <- percent(sum(results.test.M$rec.part1.num)/nrow(results.test.M))
dec.tree1.acc <- percent(sum(results.test.M$dec.tree1.num)/nrow(results.test.M))
rand.for1.acc <- percent(sum(results.test.M$rand.for1.num)/nrow(results.test.M))
log.reg.acc <- percent(sum(results.test.M$log.reg.num)/nrow(results.test.M))
knn.7.acc <- percent(sum(results.test.M$knn.7.num)/nrow(results.test.M))
group <- 'M'
accuracy.comparison.M <- data.frame(group,rec.part1.acc,dec.tree1.acc,rand.for1.acc,log.reg.acc,knn.7.acc)

# Create accuracy % for B
rec.part1.acc <- percent(sum(results.test.B$rec.part1.num)/nrow(results.test.B))
dec.tree1.acc <- percent(sum(results.test.B$dec.tree1.num)/nrow(results.test.B))
rand.for1.acc <- percent(sum(results.test.B$rand.for1.num)/nrow(results.test.B))
log.reg.acc <- percent(sum(results.test.B$log.reg.num)/nrow(results.test.B))
knn.7.acc <- percent(sum(results.test.B$knn.7.num)/nrow(results.test.B))
group <- 'B'
accuracy.comparison.B <- data.frame(group,rec.part1.acc,dec.tree1.acc,rand.for1.acc,log.reg.acc,knn.7.acc)

#Create .num column: 1 if cancer predicted, else 0
results.test.M$rec.part1.num <- ifelse(results.test.M$rec.part1 == 'M', 1, 0)
results.test.M$dec.tree1.num <- ifelse(results.test.M$dec.tree1 == 'M', 1, 0)
results.test.M$rand.for1.num <- ifelse(results.test.M$rand.for1 == 'M', 1, 0)
results.test.M$log.reg.num <- ifelse(results.test.M$log.reg == 'M', 1, 0)
results.test.M$knn.7.num <- ifelse(results.test.M$knn.7 == 'M', 1, 0)

results.test.B$rec.part1.num <- ifelse(results.test.B$rec.part1 == 'M', 1, 0)
results.test.B$dec.tree1.num <- ifelse(results.test.B$dec.tree1 == 'M', 1, 0)
results.test.B$rand.for1.num <- ifelse(results.test.B$rand.for1 == 'M', 1, 0)
results.test.B$log.reg.num <- ifelse(results.test.B$log.reg == 'M', 1, 0)
results.test.B$knn.7.num <- ifelse(results.test.B$knn.7 == 'M', 1, 0)

# Add up .num to get a consensus number.  Use probability for log.reg.
results.test.M$decision.sum <- (# results.test.M$rec.part1.num +
                                results.test.M$dec.tree1.num +
                                results.test.M$rand.for1.num +
                                results.test.M$log.reg.p +
                                results.test.M$knn.7.num)

results.test.B$decision.sum <- (# results.test.B$rec.part1.num +
                                results.test.B$dec.tree1.num +
                                results.test.B$rand.for1.num +
                                results.test.B$log.reg.p +
                                results.test.B$knn.7.num)

# Remove unnecessary columns
results.test.M$rec.part1.num <- NULL
results.test.M$dec.tree1.num <- NULL
results.test.M$rand.for1.num <- NULL
results.test.M$log.reg.num <- NULL
results.test.M$knn.7.num <- NULL

results.test.B$rec.part1.num <- NULL
results.test.B$dec.tree1.num <- NULL
results.test.B$rand.for1.num <- NULL
results.test.B$log.reg.num <- NULL
results.test.B$knn.7.num <- NULL

# Create risk and diagnosis.pred variables based on ensemble
results.test.M <- results.test.M %>% 
  # mutate(risk = cut(decision.sum, c(-1, 2, 3.5, 4.5, 6), 
  mutate(risk = cut(decision.sum, c(-1, 1.6, 2.8, 3.6, 6), 
                    labels = c("LOW", "MEDIUM", "HIGH", "VERY HIGH"))) %>% 
  mutate(diagnosis.pred = ifelse(decision.sum < .4, "B", "M"))

results.test.B <- results.test.B %>% 
  # mutate(risk = cut(decision.sum, c(-1, 2, 3.5, 4.5, 6), 
  mutate(risk = cut(decision.sum, c(-1, 1.6, 2.8, 3.6, 6), 
                    labels = c("LOW", "MEDIUM", "HIGH", "VERY HIGH"))) %>% 
  mutate(diagnosis.pred = ifelse(decision.sum < .4, "B", "M"))

# Calculate ensemble model accuracy
results.test.M$pred.num <- ifelse(results.test.M$diagnosis.pred == results.test.M$diagnosis, 1, 0)
results.test.M$correct <- ifelse(results.test.M$diagnosis.pred == results.test.M$diagnosis, 
                                        'CORRECT', 'INCORRECT')
accuracy.comparison.M$ensemble.acc <- percent(sum(results.test.M$pred.num)/nrow(results.test.M))

results.test.B$pred.num <- ifelse(results.test.B$diagnosis.pred == results.test.B$diagnosis, 1, 0)
results.test.B$correct <- ifelse(results.test.B$diagnosis.pred == results.test.B$diagnosis, 
                                        'CORRECT', 'INCORRECT')
accuracy.comparison.B$ensemble.acc <- percent(sum(results.test.B$pred.num)/nrow(results.test.B))

# Combine results
accuracy.comparison <- rbind(accuracy.comparison, accuracy.comparison.B, accuracy.comparison.M)

# Remove unnecessary column
results.test.M$pred.num <- NULL
results.test.B$pred.num <- NULL

# Subset test patients predicted incorrectly by any model
results.test.wrong <- results.test.M[which(results.test.M$decision.sum < 3.4),]

# Format variable to numeric
accuracy.comparison$rec.part1.acc <- as.numeric(sub("%", "", as.character(accuracy.comparison$rec.part1.acc)))
accuracy.comparison$dec.tree1.acc <- as.numeric(sub("%", "", as.character(accuracy.comparison$dec.tree1.acc)))
accuracy.comparison$rand.for1.acc <- as.numeric(sub("%", "", as.character(accuracy.comparison$rand.for1.acc)))
accuracy.comparison$log.reg.acc <- as.numeric(sub("%", "", as.character(accuracy.comparison$log.reg.acc)))
accuracy.comparison$knn.7.acc <- as.numeric(sub("%", "", as.character(accuracy.comparison$knn.7.acc)))
accuracy.comparison$ensemble.acc <- as.numeric(sub("%", "", as.character(accuracy.comparison$ensemble.acc)))

# Transpose data for graph
accuracy.comparison.long <- gather(accuracy.comparison, model, accuracy, rec.part1.acc:ensemble.acc, factor_key=TRUE)

# Clustered bar plot of accuracy
ggplot(accuracy.comparison.long, aes(group, accuracy, fill = model)) + 
  geom_bar(stat = "identity", color = 'black', position = "dodge") + 
  geom_text(aes(label = accuracy), 
            vjust = .5, hjust = 1.7,
            color = "white",
            position = position_dodge(0.9), size = 3.5, angle = 90) +
  coord_cartesian(ylim=c(85,100)) +
  scale_fill_brewer( palette = "Dark2") +
  labs(fill = "Model") +
  labs(x = "Patient Subset") +
  labs(y = "Accuracy %") +
  ggtitle("Model Accuracy % by Patient Subset")

# Remove unnecessary objects
rm(accuracy.comparison.M, accuracy.comparison.B, accuracy.comparison.long)
rm(group, rec.part1.acc, dec.tree1.acc, rand.for1.acc, log.reg.acc, knn.7.acc)
rm(results.test.M, results.test.B)

```

