# Course:        CS 513-B Final Project
# Project Name:  Austin Animal Center Shelter Outcome Classification
# By :            Keyur Patel CWID - 10427232

###################################################################################################################

# Data Cleanining Process

rm(list=ls())

# Loading the original dataset
aacData <- read.csv("C:/Users/harsh/Desktop/Fall 2018/Knowledge Discovery & Data Mining/Project/AAC Shelter Intakes and Outcomes/aac_intakes_outcomes.csv");

# Exploratory Data Analysis
summary(aacData)
str(aacData)
head(aacData) 

# Removing null values from traget variable "outcome_type"
aacData <- subset(aacData, outcome_type != "")

# Creating subset of the dataset by removing columns such as age_upon_outcome, date_of_birth, etc. which already has prased values into columns age_upon_outcome_.days., age_upon_outcome_.years., dob_year, dob_month, etc.
aacData_1 <- subset(aacData, select = -c(1:4, 10, 13, 19:21, 23:25, 29, 33, 36, 40))  

# Columns sex_upon_intake and sex_upon_outcome has 1 null value for the same observation, so we changed it's value to "Unknown" 
aacData_1$sex_upon_intake[aacData_1$sex_upon_intake == ""] <- "Unknown"
aacData_1$sex_upon_outcome[aacData_1$sex_upon_outcome == "NULL"] <- "Unknown"

# Writing the cleaned dataset to csv file
write.csv(aacData_1, "C:/Users/harsh/Desktop/Fall 2018/Knowledge Discovery & Data Mining/Project/aacData_1.csv")

# Reading the processed dataset
aacData_1 <- read.csv("C:/Users/harsh/Desktop/Fall 2018/Knowledge Discovery & Data Mining/Project/aacData_1.csv")

# Exploratory Data Analysis
View(aacData_1)
summary(aacData_1)
str(aacData_1)

###################################################################################################################

# APPLYING DATA MINING TECHNIQUES 

########################################## Random Forest Classifier 1 #############################################

rm(list=ls())

# Loading the processed dataset 
aacData_1 <- read.csv("C:/Users/harsh/Desktop/Fall 2018/Knowledge Discovery & Data Mining/Project/aacData_1.csv")

# Applying Random Forest Algorithm to the dataset ####
library(randomForest)

# Creating a test dataset with 25% of the data. 
# The remaining 75% of data is used as training dataset.
index <- sample(nrow(aacData_1),as.integer(.25*nrow(aacData_1)))
test <- aacData_1[index,]        # Test dataset
training <- aacData_1[-index,]   # Training dataset

# Training the model
fitRF1 <- randomForest( factor(outcome_type)~., data = training, importance = TRUE, ntree = 1000)
importance(fitRF1)
varImpPlot(fitRF1)

# Plotting the trained model
plot(fitRF1)

# Applying the model to the test dataset and predicting the outcome_type of the test dataset
Prediction_RF1 <- predict(fitRF1, test)

# Table depicting Actual and Predicted Values 
table_RF1 <- table(Actual = test$outcome_type, Prediction = Prediction_RF1)

# Calculating error rate
wrong_RF1 <- (test$outcome_type != Prediction_RF1)
error_RF1 <- sum(wrong_RF1)/length(wrong_RF1)
error_RF1

#install.packages("caret")
library(caret)

# Confusion Matrix
CM_RF1 <- confusionMatrix(table_RF1)
CM_RF1

# Accuracy with Dataset 1 is 83.29% (83.12%) ####

########################################## Random Forest Classifier 2 #############################################

#### From the implemention of first the random forest classifier we found 24 important features valuable to our classification model. ####
# We decided to go with the first 20 important features and eliminated the rest 4 features. ####
# We applied the random forest classifier again to this new subset with 20 predictor variables and 1 target variable (outcome_type). ####

# Creating the subset of dataset based on the first 20 important features given by the random forest classifier
aacData_2 <- subset(aacData_1, select = -c(5, 10, 19, 24))

# Writing the final dataset into csv file.
# Using this dataset as the input to the rest of the classifiers
write.csv(aacData_2, "C:/Users/harsh/Desktop/Fall 2018/Knowledge Discovery & Data Mining/Project/aacData_2.csv")

# Loading the dataset
aacData_2 <- read.csv("C:/Users/harsh/Desktop/Fall 2018/Knowledge Discovery & Data Mining/Project/aacData_2.csv")

# Applying Random Forest Classiefier to the dataset
library(randomForest)

# Creating a test dataset with 25% of the data. 
# The remaining 75% of data is used as training dataset.
index <- sample(nrow(aacData_2),as.integer(.25*nrow(aacData_2)))
test <- aacData_2[index,]           # Test dataset
training <- aacData_2[-index,]      # Training dataset

# Training the model
fit_RF2 <- randomForest( factor(outcome_type)~., data = training, importance = TRUE, ntree = 1000)
importance(fit_RF2)
varImpPlot(fit_RF2)

plot(fit_RF2)

# Applying the model to the test dataset and predicting the outcome_type of the test dataset
Prediction_RF2 <- predict(fit_RF2, test)

# Table depicting Actual and Predicted Values 
table_RF2 <- table(Actual = test$outcome_type, Prediction = Prediction_RF2)

# Calculating error rate
wrong_RF2 <- (test$outcome_type != Prediction_RF2)
error_RF2 <- sum(wrong_RF2)/length(wrong_RF2)
error_RF2

library(caret)

# Confusion Matrix
CM_RF2 <- confusionMatrix(table_RF2)
CM_RF2

# Accuracy with Dataset 2 is 83.24% (82.98) ####

# After running the  random forest classifier on both the first dataset (aacData_1.csv) and the subset with 20 important features (aacData_2),   ####
#we can see that the accuracy is not affected much with the elimination of those 4 features. ####
# So, we decided to use those 20 important features to use as the predictor variables for the rest of the models. ####

############################################ Decision Tree - CART #################################################
#Libraries for Decision trees
library(rpart)
library(rpart.plot)
library(rattle)

# Generating the tree
mytree <- rpart(outcome_type~., data = training) #method="class")
mytree

#Plotting a clear tree
prp(mytree, extra = 101)

fancyRpartPlot(mytree)

# PLOTTING THE DECISION TREE
plot <- rpart.plot(mytree, extra = 101)
#fancyRpartPlot(mytree, palettes=c("Greys", "Oranges"))


#Predicting the outcome
Prediction_D <- predict(mytree, test, type="class")

#Confusion Matrix
table_D <- table(Actual = test$outcome_type, Prediction_D)


#Error rate using Decision Trees
error_D <- 1-sum(diag(table_D))/sum(table_D)
error_D

#Accuracy
Accuracy_D <- (1-error_D)*100
Accuracy_D

#Printing the result
cat("Accuracy for Decision Tree:",Accuracy_D,"%")

# load Caret package for computing Confusion matrix
library(caret) 
CM_D <- confusionMatrix(table_D)
CM_D

############################################ Decision Tree - C5.0 #################################################

library(C50)

# Generating a subset of the dataset based on the features selected by the Random Forest

aacData_3.train <- subset(training, select = c(1, 2, 8, 13:15))
aacData_3.test <- subset(test, select = c(1, 2, 8, 13:15))

c50_tree <- C5.0(factor(outcome_type)~.,data = aacData_3.train)
c50_tree
plot(c50_tree)

summary(c50_tree)
#plot(C50_class)

# Predicting the test data
Predict_c50 <- predict(c50_tree, aacData_3.test, type="class")
length(test)

table_c50 <- table(Actual = test$outcome_type, Prediction = Predict_c50)
table_c50

# Calculating error rate and accuracy
wrong_c50 <- (test$outcome_type != Predict_c50)
error_c50 <- sum(wrong_c50)/length(wrong_c50)
error_c50

Accuracy_c50 <- (1-error_C50)*100
Accuracy_c50

# Computing Confusion matrix
CM_c50 <- confusionMatrix(table_c50)
CM_c50

############################################### k - Nearest Neighbors #############################################

# install.packages("kknn")
# install.packages("caret")

library(kknn)

# Converting factors to numeric
aacData_2$sex_upon_outcome <- as.numeric(factor(aacData_2$sex_upon_outcome, levels = c("Intact Female", "Intact Male", "Neutered Male", "Spayed Female", "Unknown"), labels = c("1", "2", "3", "4", "5")))

aacData_2$outcome_weekday <- as.numeric(factor(aacData_2$outcome_weekday, levels = c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"), labels = c("1", "2", "3", "4", "5", "6", "7")))

aacData_2$animal_type <- as.numeric(factor(aacData_2$animal_type, levels = c("Dog", "Cat", "Bird", "Other"), labels = c("1", "2", "3", "4")))

aacData_2$intake_condition <- as.numeric(factor(aacData_2$intake_condition, levels = c("Aged", "Feral", "Injured", "Normal", "Nursing", "Pregnant", "Sick", "Other"), labels = c("1", "2", "3", "4", "5", "6", "7", "8")))

aacData_2$intake_type <- as.numeric(factor(aacData_2$intake_type, levels = c("Euthanasia Request", "Owner Surrender", "Public Assist", "Stray", "Wildlife"), labels = c("1", "2", "3", "4", "5")))

aacData_2$sex_upon_intake <- as.numeric(factor(aacData_2$sex_upon_intake, levels = c("Intact Female", "Intact Male", "Neutered Male", "Spayed Female", "Unknown"), labels = c("1", "2", "3", "4", "5")))

aacData_2$intake_weekday <- as.numeric(factor(aacData_2$intake_weekday, levels = c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"), labels = c("1", "2", "3", "4", "5", "6", "7")))

# Created normalization function to normalize the data
normalize <- function(x){
  return ((x - min(x))/(max(x) - min(x)))
}

# Applying the normalization function to Predictor variables.
aacData2_new <- as.data.frame(lapply(aacData_2[, c(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)], normalize))

# Normalized dataset
aacData2_norm <- as.data.frame(cbind(outcome_type = aacData_2$outcome_type, aacData2_new))
# View(aacData2_norm)

# str(aacData2_norm)
# summary(aacData2_norm)

# Applying K-Nearest Neighborhood Algorithm  ####

# Creating a test dataset with 25% of the data. 
# The remaining 75% of data is used as training dataset.
index <- sample(nrow(aacData2_norm),as.integer(.25*nrow(aacData2_norm)))
test_norm <- aacData2_norm[index,]              # Test dataset
training_norm <- aacData2_norm[-index,]         # Training dataset

# Using K = 1 to K = 15 to classify the test dataset     ####
for(i in c(1:15)){
  Predict_knn <- kknn(formula = outcome_type~., training_norm, test_norm, kernel="rectangular", k=i)
  fit <- fitted(Predict_knn)
  wrong_knn <- (test_norm[, 1] != fit)
  rate_knn <- sum(wrong_knn)/length(wrong_knn)
  print(paste("Error Rate -", rate_knn))
  CM_knn <- table(Test = test_norm[, 1], Prediction = fit)
  accuracy_knn <- sum(diag(CM_knn))/sum(CM_knn)
  print(paste("Accuracy -", accuracy_knn))
}

# From the observation of error rate and accuracy (Measure of the performance of knn) for K = 1 to K = 15, 
# the optimal accuracy (74.3%) and minimum error rate (25.6%) is found when k = 12.  ####

# Optimal results are at k=12
Predict_knn <- kknn(formula = outcome_type~., training_norm, test_norm, k = 12, kernel="rectangular")
fit_knn <- fitted(Predict_knn)    
fit_knn

# Measure of the performance of knn
wrong_knn <- (test_norm[, 1]!=fit_knn)
rate_knn <- sum(wrong_knn)/length(wrong_knn)
print(paste("Error Rate -", rate_knn))

table_knn <- table(Test = test_norm[, 1], Prediction = fit_knn)
table_knn


#Accuracy
accuracy_knn <- sum(diag(table_knn))/sum(table_knn)
print(paste("Acurracy -", accuracy_knn))

# Confusion Matrix
cM_knn <- confusionMatrix(table_knn)
cM_knn

############################################# SUPPORT VECTOR MACHINE ##############################################

library(e1071)
svm.model <- svm(factor(outcome_type)~., data = training)
svm.model

Predict_svm <- predict(svm.model, test)

table_svm <- table(actual=test$outcome_type, Predict_svm)


#Error rate
wrong_SVM <- (test$outcome_type!= Predict_svm)
error_SVM <- sum(wrong_SVM)/length(wrong_SVM)
error_SVM


#Accuracy
Accuracy_SVM <- (1-error_SVM)*100
Accuracy_SVM
cat("Accuracy for SVM:",Accuracy_SVM,"%")


#Confusion Matrix for SVM
CM_SVM <- confusionMatrix(table_svm)
CM_SVM

x.svm.prob <- predict(svm.model, newdata=test)

x.svm.prob.rocr <- prediction(attr(x.svm.prob, "probabilities")[,2], BreastCancer[ind == 2,'Class'])
x.svm.perf <- performance(x.svm.prob.rocr, "tpr","fpr")
plot(x.svm.perf, col=6, add=TRUE)

############################################## NAIVE BAYESIAN #####################################################
# Importing libraries for Naive Bayesian classifier
library(e1071)
library(class)

#Naive-Bayesian classification
nBayes_OT <- naiveBayes(outcome_type~., data = aacData_2)
nBayes_OT

Predict_NB <- predict(nBayes_OT, aacData_2)


#Generating Confusion Matrix
table_NB <- table(Actual = aacData_2$outcome_type, Prediction = Predict_NB)

# Measure of the performance of NB
wrong_NB <- (test_norm[, 1]!=Predict_NB)
rate_NB <- sum(wrong_NB)/length(wrong_NB)
print(paste("Error Rate -", rate_NB))

accuracy_NB <- sum(diag(table_NB))/sum(table_NB)
print(paste("Acurracy -", accuracy_NB))

CM_NB <- confusionMatrix(table_NB)
CM_NB

###################################################################################################################

# Prediction Model Selection and Final results

result <- data.frame(Random_Forest = CM_RF2$overall,
                     Decision_Tree = CM_D$overall,
                     C5.0 = CM_c50$overall,
                     Knn = cM_knn$overall,
                     SVM = CM_SVM$overall,
                     NB = CM_NB$overall)
View(result)

result1 <- data.frame(Random_Forest = CM_RF2$overall[1],
                     Decision_Tree = CM_D$overall[1],
                     C5.0 = CM_c50$overall[1],
                     Knn = cM_knn$overall[1],
                     SVM = CM_SVM$overall[1],
                     NB = CM_NB$overall[1])

write.csv(result, "C:/Users/harsh/Desktop/Fall 2018/Knowledge Discovery & Data Mining/Project/finalresults.csv")
View(result1)


