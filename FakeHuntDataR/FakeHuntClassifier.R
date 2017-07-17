#Set working directory

#Load in data
data=read.csv("FakeHuntDat.csv", header=TRUE)

#Stores classification in y_data
y_data=data$Class

#Stores feature data in x_data
x_data=data[-c(9)]

##75% of the sample size
smp_size<-floor(0.75*nrow(data))
#Set seed to make partition reproducible
set.seed(120)
train_ind<-sample(seq_len(nrow(data)), size=smp_size)
train<-data[train_ind,]
test<-data[-train_ind,]

#Load Random Forest libraries
library(splines)
library(parallel)
library(lattice)
library(survival)
library(randomForest)
library(randomForestSRC)


#Build the random forest classifier
rForestFull2<-rfsrc(Class~., data=train, mtry=3, ntree= 500, importance  = TRUE, na.action="na.impute")

#Summary of Performance
rForestFull2

#Predict class
yhat<-predict(rForestFull2, newdata=test, type="class", na.action="na.impute")
yhat$class

#Predicted probabilities
yhat$predicted

#Merges outputs to a dataframe
output<-cbind(as.character(yhat$class), yhat$predicted)

#Outputs dataframe to a csv
write.csv(output, file="OutputFakeHuntClassifier.csv")

#Output misclassification rate
y<-test$Class
sum(y !=yhat$class)/length(y)

#Output Prediction rate
1-sum(y !=yhat$class)/length(y)

