#Set working directory

data=read.csv("FakeHuntDta.csv", header=TRUE)
#Everything else is the x data


#Stores classification in y_data
y_data=data$Class

#Stores feature data in x_data
x_data=data[-c(9)]
#View(x_data)
#View(y_data)

##75% of the sample size
smp_size<-floor(0.75*nrow(data))
#Set seed to make partition reproducible
set.seed(120)
train_ind<-sample(seq_len(nrow(data)), size=smp_size)
train<-data[train_ind,]
test<-data[-train_ind,]

#Random Forest
library(splines)
library(parallel)
library(lattice)
library(survival)
library(randomForest)

######################################################################################################
#
#         RF on No Enrichment
#
######################################################################################################
rForest1<-randomForest(train$Class~CurrentThreat+TitleContainThreat+ThreatLevel, data=train, mtry=3, ntree= 400, importance  = TRUE)
plot(rForest1)
yhatprob<-predict(rForest1, newdata=test, type="prob")
yhatprob

#Predict class
yhat<-predict(rForest1, newdata=test, type="class")
yhat

#Prediction probabilities
yhatprob<-predict(rForest1, newdata=test, type="prob")
yhatprob

#Output misclassification rate
y<-test$Class
sum(y !=yhat)/length(y)

#Output Prediction rate
1-sum(y !=yhat)/length(y)



######################################################################################################
#
#         RF on Full Enrichment
#
######################################################################################################
rForestFull<-randomForest(train$Class~., data=train, mtry=3, ntree= 200, importance  = TRUE)
#plot(rForestFull)  #plots OOB msevs # trees
#Here we can see that we only need about 200 trees
rForestFull #check the OOB mseand  r^2
importance(rForestFull); 
varImpPlot(rForestFull)

#Predict class
yhat<-predict(rForestFull, newdata=test, type="class")
yhat

#Prediction probabilities
yhatprob<-predict(rForestFull, newdata=test, type="prob")
yhatprob

#Output misclassification rate
y<-test$Class
sum(y !=yhat)/length(y)

#Output Prediction rate
1-sum(y !=yhat)/length(y)

######################################################################################################
#
#         RF on Top 4 Predictors
#
######################################################################################################
rForest4<-randomForest(train$Class~VT+CurrentThreat+DNS+Hash, data=train, mtry=3, ntree= 200, importance  = TRUE)
rForest4 #check the OOB mseand  r^2
importance(rForest4); 
varImpPlot(rForest4)

#Predict class
yhat<-predict(rForest4, newdata=test, type="class")
yhat

#Prediction probabilities
yhatprob<-predict(rForest4, newdata=test, type="prob")
yhatprob

#Output misclassification rate
y<-test$Class
sum(y !=yhat)/length(y)

#Output Prediction rate
1-sum(y !=yhat)/length(y)

######################################################################################################
#
#         RF on Top 3 Predictors
#
######################################################################################################
rForest3<-randomForest(train$Class~VT+CurrentThreat+DNS, data=train, mtry=3, ntree= 200, importance  = TRUE)
rForest3 #check the OOB mseand  r^2
importance(rForest3); 
varImpPlot(rForest3)

#Predict class
yhat<-predict(rForest3, newdata=test, type="class")
yhat

#Prediction probabilities
yhatprob<-predict(rForest3, newdata=test, type="prob")
yhatprob

#Output misclassification rate
y<-test$Class
sum(y !=yhat)/length(y)

#Output Prediction rate
1-sum(y !=yhat)/length(y)

######################################################################################################
#
#         RF on Top 2 Predictors
#
######################################################################################################
rForest2<-randomForest(train$Class~VT+CurrentThreat, data=train, mtry=3, ntree= 200, importance  = TRUE)
rForest2 #check the OOB mseand  r^2
importance(rForest2); 
varImpPlot(rForest2)

#Predict class
yhat<-predict(rForest2, newdata=test, type="class")
yhat

#Prediction probabilities
yhatprob<-predict(rForest2, newdata=test, type="prob")
yhatprob

#Output misclassification rate
y<-test$Class
sum(y !=yhat)/length(y)

#Output Prediction rate
1-sum(y !=yhat)/length(y)


######################################################################################################
#
#         Fit a RF SRC
#
######################################################################################################
library(randomForestSRC)
rForestFull2<-rfsrc(Class~., data=train, mtry=3, ntree= 200, importance  = TRUE, na.action="na.impute")
#plot(rForestFull)  #plots OOB msevs # trees
#Here we can see that we only need about 200 trees
rForestFull2


#Predict class
yhat<-predict(rForestFull2, newdata=test, type="class", na.action="na.impute")
yhat$class
#Predicted probabilities
yhat$predicted


#Output misclassification rate
y<-test$Class
sum(y !=yhat$class)/length(y)

#Output Prediction rate
1-sum(y !=yhat$class)/length(y)


######################################################################################################
#
#         Fit a GBM
#
######################################################################################################
library(gbm)

#What is var.monotone?
gbm1<-gbm(Class~., data=data, var.monotone=rep(0,8),  distribution="multinomial", n.trees=5000, shrinkage=0.1, interaction.depth=3, bag.fraction= .5, train.fraction= 1, n.minobsinnode= 3, cv.folds= 3, keep.data=TRUE,  verbose=FALSE)
best.iter<-gbm.perf(gbm1, method="cv")
best.iter
gbm1$cv.error[best.iter]
summary(gbm1, ntrees=best.iter)

yhat<-predict(gbm1, newdata=test, type="response")
yhat

#Output misclassification rate
y<-test$Class
sum(y !=yhat)/length(y)

#Output Prediction rate
1-sum(y !=yhat)/length(y)


gbm2<-gbm(Class~., data=data,distribution="multinomial", n.trees=5000, shrinkage=0.1, interaction.depth=3, bag.fraction= .5, train.fraction= 1, n.minobsinnode= 3, cv.folds= 3, keep.data=TRUE,  verbose=FALSE)
best.iter<-gbm.perf(gbm2, method="cv")
best.iter
gbm2$cv.error[best.iter]
summary(gbm2, ntrees=best.iter)


######################################################################################################
#
#         Fit NN on Full Enrichment
#
######################################################################################################
# library(nnet)
# library(lattice)
# library(ggplot2)
# library(caret)
# nn1<-nnet(train$Class~., data=train, linout=F, skip=F, size=10, decay=0.01, maxit=1000, trace=F)
# 
# #Predict class
# yhat<-predict(nn1, newdata=test, type="prob")
# yhat
# 
# #Output misclassification rate
# y<-test$Class
# sum(y !=yhat)/length(y)
# 
# #Output Prediction rate
# 1-sum(y !=yhat)/length(y)
