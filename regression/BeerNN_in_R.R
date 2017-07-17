#Beer data NN in R
#BOOZED allen hamilton lol

#Set working directory to GitHub
setwd("~/GitHub/unsupervised-learners/regression")

#Import data
beer <- read.csv("beerdata.csv",header=FALSE)
#We want to fit a model that esitmates the first column

#Scale data
beer1<-beer
k<-ncol(beer)  #number of predictors
beer1[2:k]<-sapply(beer1[2:k], function(x) (x-mean(x))/sd(x)) #standardize predictors

##75% of the sample size
smp_size<-floor(0.75*nrow(beer1))
#Set seed to make partition reproducible
set.seed(120)
train_ind<-sample(seq_len(nrow(beer1)), size=smp_size)
train<-beer1[train_ind,]
test<-beer1[-train_ind,]


#Fit Neural Network
library(nnet)
library(lattice)
library(ggplot2)
library(caret)
nn1<-nnet(train$FT.TP.RH~., data=train, linout=F, skip=F, size=10, decay=0.01, maxit=1000, trace=F)

#Predict class
yhat<-predict(nn1, newdata=test, type="class")
yhat

#Output misclassification rate
y<-test$V1
sum(y !=yhat)/length(y)

#Output Prediction rate
1-sum(y !=yhat)/length(y)

rf<-rfsrc(train$V1~., data=train)
