setwd("~/GitHub/unsupervised-learners/regression")
CRT <- read.csv("CRT.csv",header=TRUE)
k<-ncol(CRT)-1  #number of predictors
CRT1 <- CRT  #will be standardized and scaled version of data
CRT1[1:k]<-sapply(CRT1[1:k], function(x) (x-mean(x))/sd(x)) #standardize predictors
CRT1[k]<-(CRT1[k]-min(CRT1[k]))/(max(CRT1[k])-min(CRT1[k]))
CRT[1:10,]
pairs(CRT, cex=.5, pch=16)

#predictive numerical response
library(nnet)
nn1<-nnet(Strength~.,CRT1, linout=T, skip=F, size=10, decay=0.01, maxit=1000, trace=F)
yhat<-as.numeric(predict(nn1))
y<-CRT1[[9]]; e<-y-yhat
plot(yhat,y, main="predicted values vs data", col="lightgoldenrod")
c(sd(y),sd(e))
summary(nn1)


#repeat but using logistic output function, for which the response MUST BE SCALED TO [0,1] RANGE
nn2<-nnet(Strength.bin~.,CRT1[, -9], linout=F, skip=F, size=10, decay=0.01, maxit=1000, trace=F)
yhat<-as.numeric(predict(nn2)) 
y<-CRT1[[9]]; e<-y-yhat
plot(yhat,y)
c(sd(y),sd(e))
##
summary(nn1)
