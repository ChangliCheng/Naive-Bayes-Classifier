
#Install R package for Naive Bayes
if (!require("e1071")) {
  install.packages("e1071")
  library("e1071")
}

#Install other necessary libraries
if (!require("caret")) {
  install.packages("caret")
  library("caret")
}

#Read in data
myData <- read.csv("flightDelays.csv",stringsAsFactors = TRUE)

#Convert all necessary data to factors (do not leave even binary variables as int)
myData$DayOfWeek <- as.factor(myData$DayOfWeek)
myData$Weather <- as.factor(myData$Weather)
myData$FlightStatus <- as.factor(myData$FlightStatus)
#Split data into 75% training set and 25% test set. Use seed(50).
trainSetSize <- floor(0.75 * nrow(myData))   
RNGkind(sample.kind = "Rejection")
set.seed(50)                       
trainInd <- sample(seq_len(nrow(myData)), size = trainSetSize) 
myDataTrain <- myData[trainInd, ]               
myDataTest <- myData[-trainInd, ] 
dim(myDataTrain)
dim(myDataTest)

#Build Naive Bayes model on the training data
nbModel <- naiveBayes(FlightStatus ~ ., myDataTrain)

#Diplay model
nbModel

#Predict the probabilities and the outcomes for new observation(s) stored in test data set
predTestProb <- predict(nbModel, myDataTest, type = "raw") 
predTestClass <- predict(nbModel, myDataTest) #default cutoff is 0.5

#Print results
predTestProb
predTestClass

#Export the predicted probabilities and outcomes to file predictedDelaysNB.csv
dfToExport <- data.frame(myDataTest,predTestProb,predTestClass)
write.csv(dfToExport, file = "../ROutput/predictedDelays.csv")

#Calculate the confusion matrix and the accuracy
ActualTestClass <- myDataTest$FlightStatus
confusionMatrix(predTestClass,ActualTestClass,positive="1")
