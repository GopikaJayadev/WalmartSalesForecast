# Walmart Sales Forecast Project
# Author: Gopika Jayadev

library(fpp)
library(glmnet)
library(plyr)
library(dplyr)
library(zoo)
library(xts)

Features <- read.csv("features.csv", header = TRUE)
Stores <- read.csv("stores.csv", header = TRUE)
Test <- read.csv("test.csv", header = TRUE)
Train <- read.csv("train.csv")


###############################################################################
# Statistical Summary of the data
###############################################################################
summary(Features)
summary(Train)

## Histogram of the continuous variables ##
par(mfrow = c(2,2))
hist(Features$Temperature, col = "grey")
hist(Features$Fuel_Price , col = "grey")
hist(Features$CPI , col = "grey")
hist(Features$Unemployment , col = "grey")
par(mfrow = c(2,3))
hist(Features$MarkDown1 , col = "blue")
hist(Features$MarkDown2, col = "pink")
hist(Features$MarkDown3, col = "red")
hist(Features$MarkDown4, col = "cyan")
hist(Features$MarkDown5, col = "grey")
par(mfrow = c(1,1))


###############################################################################
## Merging data frames
###############################################################################
# Adding store details to the training and testing set
Train <- merge(Train, Stores, all.x = TRUE)
Test <- merge(Test, Stores, all.x = TRUE)

# Adding the features to the training and testing set
Train <- merge(Train, Features, all.x = TRUE)
Test <- merge(Test, Features, all.x = TRUE)

# Checking the coulumn names of the data set to set character columns
#colnames(Train)
#colnames(Test)

# Converting dates into charater
Train$Date <- as.character(Train$Date)
Test$Date <- as.character(Test$Date)
Features$Date <- as.character(Features$Date)
startdate <- as.Date('2010-02-05')
Train$Day_no <- as.numeric(as.Date(Train$Date) - startdate) 
Test$Day_no <- as.numeric(as.Date(Test$Date) - startdate)

# Unique date identification
unique_dates <- unique(Features$Date)
dates <- as.Date(Train$Date)
year <- as.numeric(format(dates, "%Y"))
month <- as.numeric(format(dates, "%m"))
day <- as.numeric(format(dates, "%d"))
Train <- cbind(Train, year, month, day)
dates <- as.Date(Test$Date)
year <- as.numeric(format(dates, "%Y"))
month <- as.numeric(format(dates, "%m"))
day <- as.numeric(format(dates, "%d"))
Test <- cbind(Test, year, month, day)


###############################################################################
## LASSO Modeling
###############################################################################
Sales <- Train$Weekly_Sales
# Replacing all the NA with 0
Train1 <- Train
Train1[is.na(Train1)] <- 0


###############################################################################
## Function for fitting LASSO Model
###############################################################################
Lasso_Regression <- function(Training_set) # Training set without any NA values
{
  library(fpp)
  library(glmnet)
  # Defining the y variable
  Sales <- Training_set$Weekly_Sales
  X <- model.matrix(Training_set$Weekly_Sales~., data = Training_set[,c(-1,-2,-4,-5,-6,-7,-17,-18,-19,-20)])[,-1]
  
  # Values for lambda
  l.val <- exp(seq( 1, 8, length = 100))
  
  # Fitting the model
  f.L <- glmnet( X, Sales, alpha = 1, lambda = l.val)
  
  # Plot the L1 Norm
  plot(f.L)  
  
  # Plotting the log Lambda
  plot(f.L, xvar = "lambda", label = TRUE)
  
  # Cross Validating the model
  CV.L <- cv.glmnet( X, Sales, alpha = 1)
  plot(CV.L)
  
  # Selecting Lambda's for min and 1 se
  L.min <- CV.L$lambda.min
  L.1se <- CV.L$lambda.1se
  View(cbind(L.min, L.1se))
  
  # Getting coefficients of significant variables
  coef.L.min <- as.data.frame(as.matrix(coef(CV.L, s = c(L.min))))
  coef.L.4 <- as.data.frame(as.matrix(coef(CV.L, s = c(exp(4)))))
  coef.L.5.5 <- as.data.frame(as.matrix(coef(CV.L, s = c(exp(5.5)))))
  coef.L.6 <- as.data.frame(as.matrix(coef(CV.L, s = c(exp(6)))))
  coef.L.6.5 <- as.data.frame(as.matrix(coef(CV.L, s = c(exp(6.5)))))
  coef.L.7 <- as.data.frame(as.matrix(coef(CV.L, s = c(exp(7)))))
  coef.L.1se <- as.data.frame(as.matrix(coef(CV.L, s = c(L.1se))))
  C <- data.frame(cbind(coef.L.min, coef.L.4, coef.L.5.5, coef.L.6, coef.L.6.5, coef.L.7, coef.L.1se))
  colnames(C) <- c('L min', 'L 4','L 5.5', 'L 6', 'L 6.5', 'L 7', 'L 1se')
  View(C)
}

## Function Call on the training feature set ##
Lasso_Regression(Train1)


###############################################################################
# Initial Data Fitting
###############################################################################
tssales <- ts(Train$Weekly_Sales, start = c(2010,5), frequency = 52)
plot(tssales)
plot(decompose(tssales))
tssales.TR <- window(tssales, start = c(2010,5), end = c(2011,52), frequency = 52)
tssales.TS <- window(tssales, start = 2012, frequency = 52)
ndiffs(tssales.TR)
tsdisplay(tssales.TR)
tsdisplay(diff(tssales.TR))


###############################################################################
# ARIMA Model using sparse data
###############################################################################
Model1 <- Arima(tssales.TR, order = c(0,1,1))
summary(Model1)
tsdiag(Model1, gof.lag = 36)
Model1.Aicc <- Model1$aicc
Model2 <- auto.arima(tssales.TR)
summary(Model2)
tsdiag(Model2, gof.lag = 36)
Model2.aicc <- Model2$aicc
cbind(Model1.Aicc, Model2.aicc)
fore_cast <- forecast(Model2, h = 52)
plot(fore_cast, main = "Prediction from Arima(0,0,1)", xlab = "Year", ylab = "Weekly Sales")
lines(tssales.TS, col = "green", lwd = 1.5)

### Out of sample RMSE
f <- forecast(Model2, h= 43)
pred <- f$mean
error <- mean((fore_cast$mean-tssales.TS)^2)
sqrt(error)

###############################################################################
# Sorting the data frames and aggregating for better and meaningful results
###############################################################################
Train <- arrange(Train, Store, Dept, Date)
Train_aggr <- aggregate( Weekly_Sales~Date, data = Train, FUN = "sum")
# Creating a time series
y2 <- ts(Train_aggr$Weekly_Sales, start = c(2010,5), frequency = 52)


###############################################################################
## ARIMA Modeling
###############################################################################
adf.test(y2, alternative = "stationary")
# Splitting into training and testing set
y2.TR <- window(y2, start = c(2010,5), end = c(2011,52), frequency = 52)
y2.TS <- window(y2, start = 2012, frequency = 52)
# Check how many times we need to difference the data for stationarity
ndiffs(y2.TR)
# Checking the AF and PACF
plot(y2.TR, main = "Aggregated Weekly Sales", xlab = "Year", ylab = "Weekly Sales")
tsdisplay(y2.TR)

## Fitting an Auto Arima Model ##
m1 <- auto.arima(y2.TR)
summary(m1)
tsdiag(m1, gof.lag = 36)
plot(forecast(m1, h=52), main = "ARIMA(2,0,1) Forecast", xlab = "Year", ylab = "Weekly Sales")
lines(y2.TS, col = "green", led = 1.5)
m1.aicc <- m1$aicc
#out of sample error
f <- forecast(m1, h= 52)
pred <- f$mean
error <- mean((fore_cast$mean-y2.TS)^2)
sqrt(error)

## Trying Model with BoxCox transformation ##
L <- BoxCox.lambda(y2.TR)
m2 <- auto.arima(y2.TR, lambda = L)
summary(m2)
m2.aicc <- m2$aicc
tsdiag(m2, gof.lag = 36)
plot(forecast(m2, h=12))
# Model doesn't improve!!


###############################################################################
## ETS Modeling
###############################################################################
fit.tr <- ets(y2.TR, model = "ZZZ", damped = FALSE, restrict = FALSE)
summary(fit.tr)
ETS.weekly.aicc <- fit.tr$aicc
# ETS throws an error saying it might not be able to handle frequencies above 24
# We try to aggregate the weekly data to monthly data
Train_aggr$Date <- as.Date(Train_aggr$Date)
zoo <- zoo(Train_aggr$Weekly_Sales, Train_aggr$Date)
monthly.y1 <- apply.monthly(zoo, sum)
y1.mon <- ts(monthly.y1, start = c(2010,2), frequency = 12)
y1.mon.TR <- window(y1.mon, start = c(2010,2), end = c(2011,12), frequency = 12)
y1.mon.TS <- window(y1.mon, start = 2012, frequency = 12)
fit.tr <- ets(y1.mon.TR, model = "ZZZ", damped = FALSE, restrict = FALSE)
summary(fit.tr)
ETS.mon.aicc1 <- fit.tr$aicc
fore_cast <- forecast(fit.tr, h=12)
plot(fore_cast, main = "ETS Forecast", xlab = "Year", ylab = "Sales")
lines(y1.mon.TS, col = "green", lwd = 1.5)
sqrt(mean((fore_cast$mean-y1.mon.TS)^2))
residualsets <- fit.tr$residuals
tsdisplay(residualsets)
# Trying Model with BoxCox transformation 
L <- BoxCox.lambda(y1.mon.TR)
fit.tr1 <- ets(y1.mon.TR, lambda = L, model = "ZZZ", restrict = FALSE)
summary(fit.tr1)
ETS.mon.aicc2 <- fit.tr1$aicc
# Model doesn't improve


###############################################################################
# Trying Dynamic Regression Model
###############################################################################
Train1$Date <- as.Date(Train1$Date, format = "%Y-%m-%d")
# Model matrix generation is for the training set range 
# Filtering out the range for the training set
Train1_req <- subset(Train1, Train1$Date > as.Date("2010-02-04", format = "%Y-%m-%d") & Train1$Date < as.Date("2012-01-01", format = "%Y-%m-%d"))
Train1_req <- aggregate(cbind(IsHoliday, Temperature, MarkDown1, MarkDown2, MarkDown3, MarkDown4, MarkDown5, Fuel_Price, Unemployment, CPI)~Date, data = Train1_req, FUN = "mean")
# Regression Variable Matrix for Regression with ARIMA Errors
xR <- data.frame(as.factor(Train1_req$IsHoliday), Train1_req$Temperature, as.numeric(Train1_req$MarkDown1), as.numeric(Train1_req$MarkDown2), 
                 as.numeric(Train1_req$MarkDown3), as.numeric(Train1_req$MarkDown4),as.numeric(Train1_req$MarkDown5), as.numeric(Train1_req$Fuel_Price),
                 as.numeric(Train1_req$Unemployment), as.numeric(Train1_req$CPI))
Train_xR <- NULL
Train_xR <- data.matrix(xR)

tssales <- ts(y2, start = c(2010,5), frequency = 52)
plot(tssales)
plot(decompose(tssales))
tssales.TR <- window(tssales, start = c(2010,5), end = c(2011,52), frequency = 52)
tssales.TS <- window(tssales, start = 2012, frequency = 52)
tsdisplay(tssales.TR)
tsdisplay(diff(tssales))


###############################################################################
# ARIMA Model using dynamic regression model
###############################################################################
### Markdown5 Included
ModelxR <- auto.arima(tssales.TR, xreg = Train_xR[,7])
summary(ModelxR)
#### Checking the residuals of the predicted model ####
residualxR <- ModelxR$residuals
tsdisplay(arima.errors(ModelxR))
plot(residualxR)
qqnorm(residualxR)
qqline(residualxR)
Box.test(residualxR, lag = 52, type = "Ljung-Box", fitdf = 1)
tsdisplay(residualxR)
tsdisplay(diff(residualxR))
ModelxR <- Arima(tssales.TR, order = c(5,0,0), xreg = Train_xR[,7])
residualxR <- ModelxR$residuals
plot(residualxR)
qqnorm(residualxR)
qqline(residualxR)
Box.test(residualxR, lag = 52, type = "Ljung-Box", fitdf = 1)
tsdisplay(residualxR)
fore_cast <- forecast(ModelxR, h = 52, xreg = Train_xR[,7])
plot(fore_cast, main = "Prediction from Arima")
lines(tssales.TS, col = "red", lwd = 1.5)
legend("topleft",
       legend=c(expression("Forecasted Sales", "Actual Sales")),
       lty=c(1,1), col=c("blue", "red"))
sqrt(mean((fore_cast$mean-tssales.TS)^2))

# Including Markdwon 1,3 and 5
ModelxR <- auto.arima(tssales.TR, xreg = Train_xR[,c(3,5,7)])
summary(ModelxR)
#### Checking the residuals of the predicted model ####
residualxR <- ModelxR$residuals
plot(residualxR)
qqnorm(residualxR)
qqline(residualxR)
Box.test(residualxR, lag = 52, type = "Ljung-Box", fitdf = 1)
fore_cast <- forecast(ModelxR, h = 52, xreg = Train_xR[,c(3,5,7)])
plot(fore_cast, main = "Prediction from Arima")
lines(tssales.TS, col = "red", lwd = 1.5)
legend("topleft",
       legend=c(expression("Forecasted Sales", "Actual Sales")),
       lty=c(1,1), col=c("blue", "red"))
sqrt(mean((fore_cast$mean-tssales.TS)^2))

