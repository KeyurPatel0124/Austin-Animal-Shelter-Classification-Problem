This project includes a classification problem on the Austin Animal Shelter Dataset for final outcomes of the animals.
The dataset can be found [here](https://www.kaggle.com/aaronschlegel/austin-animal-center-shelter-outcomes-and#aac_shelter_outcomes.csv).

The Outcome_Type consists following outcomes:
**Adoption
Transfer
Return to Owner
Euthanasia
Died
Disposal** on the basis of which the animals are classified.

**Random Forest Classifier**
```
#Training the model
fitRF1 <- randomForest( factor(outcome_type)~., data = training, importance = TRUE, ntree = 1000)
importance(fitRF1)
varImpPlot(fitRF1)
```

 ```
 #Plotting the trained model
 plot(fitRF1)
```

**Decision Trees**
```
#Generating the tree
mytree <- rpart(outcome_type~., data = training) #method="class")
#Plotting a clear tree
prp(mytree, extra = 101)
```
This is the plot for [Decision Tree](https://github.com/KeyurPatel0124/Austin-Animal-Shelter-Classification-Problem/blob/master/Decision%20Tree.pdf) looks like this.
