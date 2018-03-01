# Walmart Sales Forecasting Project (with KNN)
# Author: Gopika Jayadev
# EID: ggj236

####
#NOTE: Run this file after the Project.R file
####
##########################################################################
# Filling missing values
##########################################################################

library(DMwR)

Features <- read.csv("features.csv", header = TRUE)
Train <- read.csv("train.csv")
# Performing knn imputation
knnOutput <- knnImputation(Features)  

Train_1 <- merge(Train, knnOutput, all.x = TRUE)
Lasso_Regression(Train_1)

###############################################################################
# Sorting the data frames and aggregating for better and meaningful results
###############################################################################
Train <- arrange(Train_1, Store, Dept, Date)
Train_aggr <- aggregate( Weekly_Sales~Date, data = Train, FUN = "sum")
# Creating a time series
y2 <- ts(Train_aggr$Weekly_Sales, start = c(2010,5), frequency = 52)

###############################################################################
# Trying Dynamic Regression Model
###############################################################################
Train_1$Date <- as.Date(Train1$Date, format = "%Y-%m-%d")
# Model matrix generation is for the training set range 
# Filtering out the range for the training set
Train1_req <- subset(Train_1, Train_1$Date > as.Date("2010-02-04", format = "%Y-%m-%d") & Train_1$Date < as.Date("2012-01-01", format = "%Y-%m-%d"))
Train1_req <- aggregate(cbind(IsHoliday, Temperature, MarkDown1, MarkDown2, MarkDown3, MarkDown4, MarkDown5, Fuel_Price, Unemployment, CPI)~Date, data = Train1_req, FUN = "mean")
# Regression Variable Matrix for Regression with ARIMA Errors
xR <- data.frame(as.factor(Train1_req$IsHoliday), Train1_req$Temperature, as.numeric(Train1_req$MarkDown1), as.numeric(Train1_req$MarkDown2), 
                 as.numeric(Train1_req$MarkDown3), as.numeric(Train1_req$MarkDown4),as.numeric(Train1_req$MarkDown5), as.numeric(Train1_req$Fuel_Price),
                 as.numeric(Train1_req$Unemployment), as.numeric(Train1_req$CPI))
Train_xR <- NULL
Train_xR <- data.matrix(xR)
tssales <- ts(y2, start = c(2010,5), frequency = 52)
tssales.TR <- window(tssales, start = c(2010,5), end = c(2011,52), frequency = 52)
tssales.TS <- window(tssales, start = 2012, frequency = 52)

# Including Markdwon 1,2 and 3
ModelxR <- auto.arima(tssales.TR, xreg = Train_xR[,c(3,5)])
summary(ModelxR)
#### Checking the residuals of the predicted model ####
#ModelxR <- Arima(tssales.TR,order = c(5,0,0), xreg = Train_xR[,c(3,4,5)])
residualxR <- ModelxR$residuals
tsdisplay(arima.errors(ModelxR))
Box.test(residualxR, lag = 52, type = "Ljung-Box", fitdf = 1)
fore_cast <- forecast(ModelxR, h = 1, xreg = Train_xR[,c(3,5)])
plot(fore_cast, main = "Prediction from Arima")
lines(tssales.TS, col = "red", lwd = 1.5)
legend("topleft",
       legend=c(expression("Forecasted Sales", "Actual Sales")),
       lty=c(1,1), col=c("blue", "red"))
sqrt(mean((fore_cast$mean-tssales.TS)^2))
