**Austin Animal Shelter Classification Problem**

This is a classification problem on the Austin Animal Shelter Dataset for final outcomes types of the animals.
The dataset can be found [here](https://www.kaggle.com/aaronschlegel/austin-animal-center-shelter-outcomes-and#aac_shelter_outcomes.csv).

The prediction variable Outcome_Type consists following outcomes:
- **Adoption**
- **Transfer**
- **Return to Owner**
- **Euthanasia**
- **Died**
- **Disposal**

The following Data Mining techniques have been used in this project:
- Random Forest Algorithm
- Decision Trees
- C5.0
- k-Nearest Neighbours
- Support Vector Machine
- Naive Bayes

The R program can be found [here](https://github.com/KeyurPatel0124/Austin-Animal-Shelter-Classification-Problem/blob/master/ProjectCode.R).

**Random Forest Algorithm**
```
# Training the model
fitRF1 <- randomForest( factor(outcome_type)~., data = training, importance = TRUE, ntree = 1000)
importance(fitRF1)
varImpPlot(fitRF1)
```

This is the plot for [Decision Tree](https://github.com/KeyurPatel0124/Austin-Animal-Shelter-Classification-Problem/blob/master/Decision%20Tree.pdf) looks like this.

**C5.0**
```
#Generating the tree
c50_tree <- C5.0(factor(outcome_type)~.,data = aacData_3.train)

# Predicting the test data
Predict_c50 <- predict(c50_tree, aacData_3.test, type="class")

#Confusion Matrix
table_c50 <- table(Actual = test$outcome_type, Prediction = Predict_c50)
table_c50

#Error in classification
wrong_c50 <- (test$outcome_type != Predict_c50)
error_c50 <- sum(wrong_c50)/length(wrong_c50)

#Accuracy
Accuracy_c50 <- (1-error_c50)*100
Accuracy_c50
```
**Error:** 0.3975395
**Accuracy:** 60.24605

The [Confusion Matrix](https://github.com/KeyurPatel0124/Austin-Animal-Shelter-Classification-Problem/blob/master/C5.0%2520Confusion%2520Matrix.PNG) for C5.0.
