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

#Fit Neural Network
library(nnet)
library(lattice)
library(ggplot2)
library(caret)
nn1<-nnet(beer1$V1~., data=beer1, linout=F, skip=F, size=10, decay=0.01, maxit=1000, trace=F)

#output the class probabilities
phat<-predict(nn1,type="raw")
apply(phat,1,sum)

#Predict class
yhat<-predict(nn1, type="class")

#Output misclassification rate
y<-
sum(y !=yhat)/length(y)
