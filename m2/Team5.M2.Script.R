setwd("C:/Users/bhoga/Desktop/Desktop/UMD FALL'23/INST737/Milestone-2/M2.Data.Files/Code_generated_Files")

# install.packages("e1071")
# install.packages("nnet")
# install.packages("glmnet")
# install.packages("ISLR")
# install.packages("gmodels")
# install.packages("C50")
# install.packages("foreign")
# install.packages("reshape2")
# install.packages("caret")

library(dplyr)
library("ggplot2")
library(tidyr)
library(gmodels)
library(C50)
library(partykit)
library(rpart)
library(e1071)
library(ISLR)
library(MASS)
library(randomForest)
library(nnet)
library(foreign)
library(reshape2)
library(caret)

# Explore dataset created from Milestone-1
#####PHASE-1 CLEANING###
data <- read.csv("FinalVariables_FM_AP_R_2022Q4.csv")
summary(data)
str(data)
head(data)
glimpse(data)

# Verify if data has Null values
summary(data)
nullValues <- colSums(is.na(data))
if (any(nullValues >0)) {
  print(nullValues)
} else {
  print("No Null values")
}

#Remove Outliers,Checking 8 Numerical Variables
#Checking Original Interest Rate statistics
summary(data$Original.Interest.Rate)

#Boxplot of Outcome variable or Dependent Variable 'Original.Interest.Rate' to visualize outliers
intRatePlot <- ggplot(data) +
  aes(x = "", y = Original.Interest.Rate) +
  geom_boxplot(fill = "#FFF5EE") +
  labs(title = "Interest Rate plot before elimination of outliers")
  theme_minimal()
intRatePlot
#List of outliers and their row nos. in Interest Rate variable
intRateOutliers <- boxplot.stats(data$Original.Interest.Rate)$out
intRateOutliers
#To extract Row nos of outliers in Interest Rates
rowNos_intRateOutliers <- which(data$Original.Interest.Rate %in% c(intRateOutliers))
rowNos_intRateOutliers
data[rowNos_intRateOutliers, ] #Complete list not output, terminated with message
# USING IQR method to detect and eliminate outliers
iqr <- IQR(data$Original.Interest.Rate)
upperBound <- quantile(data$Original.Interest.Rate, 0.75) + 1.5*iqr
upperBound
lowerBound <- quantile(data$Original.Interest.Rate, 0.25) - 1.5*iqr
lowerBound
data <- subset(data, data$Original.Interest.Rate > lowerBound & data$Original.Interest.Rate < upperBound)
#Writing a file without Interest Rate outliers
intRateFile <- "IntRate_FM2022Q4.csv"
write.csv(data, file = intRateFile, row.names = FALSE)
data <- read.csv("IntRate_FM2022Q4.csv")
dim(data)
# Creating Boxplot of Interest Rate variable after removing outliers
ggplot(data) +
  aes(x = "", y = Original.Interest.Rate) +
  geom_boxplot(fill = "#FFF5EE") +
  labs(title = "Interest Rate plot after elimination of outliers")
  theme_minimal()
#Checking Min and Max values and summary statistics of interest rate variable to verify removal of outliers
min(data$Original.Interest.Rate)
max(data$Original.Interest.Rate)
summary(data$Original.Interest.Rate)


##Plotting Predictor variables and removing Outliers
# Boxplot of Original.UPB variable
data <- read.csv("IntRate_FM2022Q4.csv")
upbPlot <- ggplot(data) +
  aes(x = "", y = Original.UPB) +
  geom_boxplot(fill = "#F5F5DC") +
  labs(title = " Origination Loan Balance plot before elimination of outliers")
  theme_minimal()
upbPlot
# USING IQR method to detect and eliminate rows with outliers in Original.UPB column
iqr <- IQR(data$Original.UPB)
upperBound <- quantile(data$Original.UPB, 0.75) + 1.5*iqr
upperBound
lowerBound <- quantile(data$Original.UPB, 0.25) - 1.5*iqr
lowerBound
data <- subset(data, data$Original.UPB > lowerBound & data$Original.UPB < upperBound)
#Writing a file without UPB outliers
upbFile <- "UPB_FM2022Q4.csv"
write.csv(data, file = upbFile, row.names = FALSE)
data <- read.csv("UPB_FM2022Q4.csv")
dim(data)
glimpse(data)

# Boxplot of Original.UPB variable after removing OL
E_upbPlot <- ggplot(data) +
  aes(x = "", y = Original.UPB) +
  geom_boxplot(fill = "#F5F5DC") +
  scale_y_continuous(labels = scales::number_format(scale = 1)) +
  labs(title = "Origination Loan Balance plot after elimination of outliers")
  theme_minimal()
E_upbPlot



#Boxplot of Original.LTV variable
data <- read.csv("UPB_FM2022Q4.csv")
ltvPlot <- ggplot(data) +
  aes(x = "", y =Original.Loan.to.Value.Ratio..LTV.) +
  geom_boxplot(fill = "#FAEBD7") +
  labs(title = "LTV plot before elimination of outliers")
  theme_minimal()
ltvPlot
# USING IQR method to detect and eliminate rows with outliers in Original.LTV column
iqr <- IQR(data$Original.Loan.to.Value.Ratio..LTV.)
upperBound <- quantile(data$Original.Loan.to.Value.Ratio..LTV., 0.75) + 1.5*iqr
lowerBound <- quantile(data$Original.Loan.to.Value.Ratio..LTV., 0.25) - 1.5*iqr
data <- subset(data, data$Original.Loan.to.Value.Ratio..LTV. > lowerBound & data$Original.Loan.to.Value.Ratio..LTV. < upperBound)
#Writing a file without LTV outliers
ltvFile <- "LTV_FM2022Q4.csv"
write.csv(data, file = ltvFile, row.names = FALSE)
data <- read.csv("LTV_FM2022Q4.csv")
dim(data)
# Boxplot of Original.LTV variable after removing OL
E_ltvPlot <- ggplot(data) +
  aes(x = "", y =Original.Loan.to.Value.Ratio..LTV.) +
  geom_boxplot(fill = "#FAEBD7") +
  labs(title = "LTV plot after elimination of outliers")
  theme_minimal()
E_ltvPlot
hist(data$Original.Loan.to.Value.Ratio..LTV.)



#Boxplot of CLTV Variable before removing outliers
data <- read.csv("LTV_FM2022Q4.csv")
cltvPlot <- ggplot(data) +
  aes(x = "", y =Original.Combined.Loan.to.Value.Ratio..CLTV.) +
  geom_boxplot(fill = "#E0FFFF") +
  labs(title = "CLTV plot before elimination of outliers")
  theme_minimal()
cltvPlot
# USING IQR method to detect and eliminate rows with outliers in Original.CLTV column
iqr <- IQR(data$Original.Combined.Loan.to.Value.Ratio..CLTV.)
upperBound <- quantile(data$Original.Combined.Loan.to.Value.Ratio..CLTV., 0.75) + 1.5*iqr
lowerBound <- quantile(data$Original.Combined.Loan.to.Value.Ratio..CLTV., 0.25) - 1.5*iqr
data <- subset(data, data$Original.Combined.Loan.to.Value.Ratio..CLTV. > lowerBound & data$Original.Combined.Loan.to.Value.Ratio..CLTV. < upperBound)
#Writing a file without LTV outliers
cltvFile <- "CLTV_FM2022Q4.csv"
write.csv(data, file = cltvFile, row.names = FALSE)
data <- read.csv("CLTV_FM2022Q4.csv")
dim(data)
#Boxplot of CLTV Variable after removal of Outliers
E_cltvPlot <- ggplot(data) +
  aes(x = "", y =Original.Combined.Loan.to.Value.Ratio..CLTV.) +
  geom_boxplot(fill = "#E0FFFF") +
  labs(title = "CLTV plot after elimination of outliers")
  theme_minimal()
E_cltvPlot



#BoxPlot of DTI variable before removing outliers
data <- read.csv("CLTV_FM2022Q4.csv")
dtiPlot <- ggplot(data) +
  aes(x = "", y = Debt.To.Income..DTI.) +
  geom_boxplot(fill = "#7FFF00") +
  labs(title = "DTI plot before elimination of outliers")
  theme_minimal()
dtiPlot
# USING IQR method to detect and eliminate rows with outliers in Original.DTI column
iqr <- IQR(data$Debt.To.Income..DTI.)
upperBound <- quantile(data$Debt.To.Income..DTI., 0.75) + 1.5*iqr
lowerBound <- quantile(data$Debt.To.Income..DTI., 0.25) - 1.5*iqr
data <- subset(data, data$Debt.To.Income..DTI. > lowerBound & data$Debt.To.Income..DTI. < upperBound)
#Writing a file without DTI outliers
dtiFile <- "DTI_FM2022Q4.csv"
write.csv(data, file = dtiFile, row.names = FALSE)
data <- read.csv("DTI_FM2022Q4.csv")
dim(data)
#BoxPlot of DTI variable after removing outliers
E_dtiPlot <- ggplot(data) +
  aes(x = "", y = Debt.To.Income..DTI.) +
  geom_boxplot(fill = "#7FFF00") +
  labs(title = "DTI plot after elimination of outliers")
  theme_minimal()
E_dtiPlot



#BOXPLOT of Borrower Credit Score variable before removing outliers
data <- read.csv("DTI_FM2022Q4.csv")
creditScorePlot <- ggplot(data) +
  aes(x = "", y = Borrower.Credit.Score.at.Origination) +
  geom_boxplot(fill = "#6495ED") +
  labs(title = "Borrower Credit Score plot before elimination of outliers")
  theme_minimal()
creditScorePlot
# USING IQR method to detect and eliminate rows with outliers in Borrowers credit score column
iqr <- IQR(data$Borrower.Credit.Score.at.Origination)
upperBound <- quantile(data$Borrower.Credit.Score.at.Origination, 0.75) + 1.5*iqr
lowerBound <- quantile(data$Borrower.Credit.Score.at.Origination, 0.25) - 1.5*iqr
data <- subset(data, data$Borrower.Credit.Score.at.Origination > lowerBound & data$Borrower.Credit.Score.at.Origination < upperBound)
#Writing a file without DTI outliers
creditscoreFile <- "CS_FM2022Q4.csv"
write.csv(data, file = creditscoreFile, row.names = FALSE)
data <- read.csv("CS_FM2022Q4.csv")
dim(data)
#BOXPLOT of Borrower Credit Score variable after removing outliers
E_creditScorePlot <- ggplot(data) +
  aes(x = "", y = Borrower.Credit.Score.at.Origination) +
  geom_boxplot(fill = "#6495ED") +
  labs(title = "Borrower Credit Score plot after elimination of outliers")
E_creditScorePlot
theme_minimal()


#BOXPLOT of No.of Borrowers variable
data <- read.csv("CS_FM2022Q4.csv")
borrowersPlot <- ggplot(data) +
  aes(x = "", y =Number.of.Borrowers) +
  geom_boxplot(fill = "#9370DB") +
  labs(title = "No. of Borrowers plot before elimination of outliers")
  theme_minimal()
borrowersPlot
# USING IQR method to detect and eliminate rows with outliers in No of Borrowers column
iqr <- IQR(data$Number.of.Borrowers)
upperBound <- quantile(data$Number.of.Borrowers, 0.75) + 1.5*iqr
upperBound
lowerBound <- quantile(data$Number.of.Borrowers, 0.25) - 1.5*iqr
lowerBound
data <- subset(data, data$Number.of.Borrowers > lowerBound & data$Number.of.Borrowers < upperBound)
dim(data)
#Writing a file without DTI outliers
creditscoreFile <- "Borrowers_FM2022Q4.csv"
write.csv(data, file = creditscoreFile, row.names = FALSE)
data <- read.csv("Borrowers_FM2022Q4.csv")
dim(data)
#BOXPLOT of No.of Borrowers variable after elimination of outliers
E_borrowersPlot <- ggplot(data) +
  aes(x = "", y =Number.of.Borrowers) +
  geom_boxplot(fill = "#9370DB") +
  labs(title = "No. of Borrowers plot after elimination of outliers")
theme_minimal()
E_borrowersPlot


# BOXPLOT OF NO OF UNITS BEFORE REMOVING OUTLIERS
data <- read.csv("Borrowers_FM2022Q4.csv")
unitsPlot <- ggplot(data) +
  aes(x = "", y =Number.of.Units) +
  geom_boxplot(fill = "#7FFFD4") +
  labs(title = "No. of Units plot before elimination of outliers")
  theme_minimal()
unitsPlot
summary(data$Number.of.Units)
# USING IQR method to detect and eliminate rows with outliers in No of uNITS column
glimpse(data)
summary(data$Number.of.Units)
iqr <- IQR(data$Number.of.Units)
upperBound <- quantile(data$Number.of.Units, 0.75) + 1.5*iqr
upperBound
lowerBound <- quantile(data$Number.of.Units, 0.25) - 1.5*iqr
lowerBound
quantile(data$Number.of.Units, c(0.25, 0.75))
### The lowerBound and upperBound values are the same. Hence subsetting will return nil rows.
#The requirement therefore is to elimininate all rows with No.of.Units !=1
data <- subset(data,data$Number.of.Units ==1)
summary(data$Number.of.Units)
dim(data)
#Writing a file without No of Units outliers
unitsFile <- "Cleaned_FM2022Q4.csv"
write.csv(data, file = unitsFile, row.names = FALSE)
data <- read.csv("Cleaned_FM2022Q4.csv")
dim(data)
#BOXPLOT of No.of Units variable after elimination of outliers
ggplot(data) +
  aes(x = "", y =Number.of.Units) +
  geom_boxplot(fill = "#9370DB") +
  labs(title = "No. of Units plot after elimination of outliers")
  theme_minimal()

#SPLIT DATA INTO TRAINING AND TEST DATASETS
#FINAL DATASET "Cleaned_FM2022Q4.csv" used for splitting
data <- read.csv("Cleaned_FM2022Q4.csv")
dim(data)
View(data)
## SPLIT Training and TEst datasets viDEO5 FROM begin  1.17
#Generate every 4th row number for the test dataset

testIndexNos <- which(1:nrow(data)%%4==0)
testIndexNos
#Write test data using the randomly selected row nos
testData <-data[testIndexNos, ]
dim(testData)
write.csv(testData, file = "Test_FM2022Q4.csv", row.names = FALSE)
data <- read.csv("Test_FM2022Q4.csv")
dim(data)
data <- read.csv("Cleaned_FM2022Q4.csv")
trainData <- data[-testIndexNos, ]
trainData
dim(trainData)
write.csv(trainData, file = "Training_FM2022Q4.csv", row.names = FALSE)
data <- read.csv("Training_FM2022Q4.csv")
dim(data)
##LINEAR REGRESSION MODELS WITH INDEPENDENT VARIABLES
train <- read.csv("Training_FM2022Q4.csv")
dim(train)
test <- read.csv("Test_FM2022Q4.csv")
dim(test)
attach(train)

#LINEAR MODEL OF 1st NUMERIC VARIABLE UPB
lmUPB <- lm(Original.Interest.Rate ~ Original.UPB)
summary(lmUPB)
plot(Original.UPB, Original.Interest.Rate,
     xlab= "UPB",
     ylab = "Interest Rate", pch =6)
abline(lmUPB, col = "red", lwd =3)

#LINEAR MODEL OF 2ND NUMERIC VARIABLE LTV
lmLTV <- lm(Original.Interest.Rate ~ Original.Loan.to.Value.Ratio..LTV.)
summary(lmLTV)
plot(Original.Loan.to.Value.Ratio..LTV., Original.Interest.Rate,
     xlab= "LTV",
     ylab = "Interest Rate", pch =6)
abline(lmLTV, col = "red", lwd =3)


#LINEAR MODEL OF 3RD NUMERIC VARIABLE CLTV
lmCLTV <- lm(Original.Interest.Rate ~ Original.Combined.Loan.to.Value.Ratio..CLTV.)
summary(lmCLTV)
plot(Original.Combined.Loan.to.Value.Ratio..CLTV., Original.Interest.Rate,
     xlab= "CLTV",
     ylab = "Interest Rate", pch =6)
abline(lmCLTV, col = "red", lwd =3)

#LINEAR MODEL OF 4th NUMERIC VARIABLE NO.of.Borrowers
lmBorrowers <- lm(Original.Interest.Rate ~ Number.of.Borrowers)
summary(lmBorrowers)
plot(Number.of.Borrowers, Original.Interest.Rate,
     xlab= "Borrowers",
     ylab = "Interest Rate", pch =6)
abline(lmBorrowers, col = "red", lwd =3)

#LINEAR MODEL OF 5th NUMERIC VARIABLE DTI
lmDTI <- lm(Original.Interest.Rate ~ Debt.To.Income..DTI.)
summary(lmDTI)
plot(Debt.To.Income..DTI., Original.Interest.Rate,
     xlab= "DTI",
     ylab = "Interest Rate", pch =6)
abline(lmDTI, col = "red", lwd =3)

#LINEAR MODEL OF 6th NUMERIC VARIABLE Borrower Credit Score
lmCS <- lm(Original.Interest.Rate ~ Borrower.Credit.Score.at.Origination)
summary(lmCS)
plot(Borrower.Credit.Score.at.Origination, Original.Interest.Rate,
     xlab= "Borrower Credit Score",
     ylab = "Interest Rate", pch =6)
abline(lmCS, col = "red", lwd =3)


#LINEAR MODEL OF 7th NUMERIC VARIABLE No of Units
lmUnits <- lm(Original.Interest.Rate ~ Number.of.Units)
summary(lmUnits)
plot(Number.of.Units, Original.Interest.Rate,
     xlab= "Number of Units",
     ylab = "Interest Rate", pch =6)
#abline(lmUnits, col = "red", lwd =3)

#cONVERTING CATEGORICAL AND NOMINAL VARIABLES INTO FACTORS FOR BUILDING LINEAR MODELS
trainFactors <- train %>%
  mutate(Property.State = as.factor(Property.State),
         Seller.Name = as.factor(Seller.Name),
         Loan.Purpose = as.factor(Loan.Purpose),
         Occupancy.Status = as.factor(Occupancy.Status),
         Property.Type = as.factor(Property.Type))

# Build a linear regression model with categorical variables
#LINEAR Regression model for Property State Variable Nominal
lmPS <- lm(Original.Interest.Rate ~ Property.State, data = trainFactors)
summary(lmPS)
plot(trainFactors$Property.State, trainFactors$Original.Interest.Rate,
     xlab= "Property State",
     ylab = "Interest Rate")
par(cex.axis = 0.3)

#LINEAR Regression model for Seller Name Variable Nominal
lmSN <- lm(Original.Interest.Rate ~ Seller.Name, data = trainFactors)
summary(lmSN)
plot(trainFactors$Seller.Name, trainFactors$Original.Interest.Rate,
     xlab= "Seller Name",
     ylab = "Interest Rate", pch=6)
par(cex.axis = 0.7)

#LINEAR Regression model for Loan Purpose Variable Categorical
lmLP <- lm(Original.Interest.Rate ~ Loan.Purpose, data = trainFactors)
summary(lmLP)
plot(trainFactors$Loan.Purpose, trainFactors$Original.Interest.Rate,
     xlab= "Loan Purpose",
     ylab = "Interest Rate", pch=6)
par(cex.axis = 0.7)


#LINEAR Regression model for Occupancy Status Variable Categorical
lmOS <- lm(Original.Interest.Rate ~ Occupancy.Status, data = trainFactors)
summary(lmOS)
plot(trainFactors$Occupancy.Status, trainFactors$Original.Interest.Rate,
     xlab= "Occupancy Status",
     ylab = "Interest Rate", pch=6)
par(cex.axis = 0.7)


#LINEAR Regression model for Property Type Variable Categorical
lmPT <- lm(Original.Interest.Rate ~ Property.Type, data = trainFactors)
summary(lmPT)
plot(trainFactors$Property.Type, trainFactors$Original.Interest.Rate,
     xlab= "Property Type",
     ylab = "Interest Rate", pch=6)
par(cex.axis = 0.7)


# PREDICTIVE FUNCTION ON TEST DATA FOR VARIABLES THAT ARE PREDICTIVE FEATURE
# Our Training dataset with categorical variables as factors
View(trainFactors)
#Converting categorical variables in Test data as factors
test <- read.csv("Test_FM2022Q4.csv")
testFactors <- test %>%
  mutate(Property.State = as.factor(Property.State),
         Seller.Name = as.factor(Seller.Name),
         Loan.Purpose = as.factor(Loan.Purpose),
         Occupancy.Status = as.factor(Occupancy.Status),
         Property.Type = as.factor(Property.Type))

dim(testFactors)
View(testFactors)


#Creating a Predictive Function for Original UPB variable with Test data
#This function can be used to predict test data later??
UPB.fit <- function(x) lmUPB$coefficient[1]=lmUPB$coefficient[2]*x

# # RESIDUALS AND FITTED VALUES OF A LINEAR MODEL
fitted(lmUPB)
resid(lmUPB)

#wHETHER THE RESIDUALS FOLLOW A NORMAL DISTRIBUTION
# THE RESIDUALS OF UPB Linear model
qqnorm(resid(lmUPB), main = "")
title(main = "Q-Q Plot: Residuals of UPB Linear Model")
qqline(resid(lmUPB))

# THE RESIDUALS OF LTV Linear model
qqnorm(resid(lmLTV), main = "")
title(main = "Q-Q Plot: Residuals of LTV Linear Model")
qqline(resid(lmLTV))


# THE RESIDUALS OF CLTV Linear model
qqnorm(resid(lmCLTV), main = "")
title(main = "Q-Q Plot: Residuals of CLTV Linear Model")
qqline(resid(lmCLTV))


# THE RESIDUALS OF No of Borrowers Linear model
qqnorm(resid(lmBorrowers), main = "")
title(main = "Q-Q Plot: Residuals of Borrowers Linear Model")
qqline(resid(lmBorrowers))

# THE RESIDUALS OF DTI Linear model
qqnorm(resid(lmDTI), main = "")
title(main = "Q-Q Plot: Residuals of DTI Linear Model")
qqline(resid(lmDTI))

# THE RESIDUALS OF Borrower Credit Score Linear model
qqnorm(resid(lmCS), main = "")
title(main = "Q-Q Plot: Residuals of Borrower Credit Score Linear Model")
qqline(resid(lmCS))

# THE RESIDUALS OF No of Units Linear model
qqnorm(resid(lmUnits), main = "")
title(main = "Q-Q Plot: Residuals of No of Units Linear Model")
qqline(resid(lmUnits))


# THE RESIDUALS OF Property State Linear model
qqnorm(resid(lmPS), main = "")
title(main = "Q-Q Plot: Residuals of Property State Linear Model")
qqline(resid(lmPS))


# THE RESIDUALS OF Seller Name Linear model
qqnorm(resid(lmSN), main = "")
title(main = "Q-Q Plot: Residuals of Seller Name Linear Model")
qqline(resid(lmSN))


# THE RESIDUALS OF Loan Purpose Linear model
qqnorm(resid(lmLP), main = "")
title(main = "Q-Q Plot: Residuals of Loan Purpose Linear Model")
qqline(resid(lmLP))


# THE RESIDUALS OF Occupancy Status Linear model
qqnorm(resid(lmOS), main = "")
title(main = "Q-Q Plot: Residuals of Occupancy Status Linear Model")
qqline(resid(lmOS))


# THE RESIDUALS OF Property Type Linear model
qqnorm(resid(lmPT), main = "")
title(main = "Q-Q Plot: Residuals of Property Type Linear Model")
qqline(resid(lmPT))


#using the trained model on test data and Confidence and Prediction Bands for Borrower Credit Score LInear MOdel
lmCS <- lm(Original.Interest.Rate ~ Borrower.Credit.Score.at.Origination, data = train)
summary(lmCS)
pc <- predict(lmCS, int = "c", newdata = test)
pp <- predict(lmCS, int = "p", newdata = test)

plot(Borrower.Credit.Score.at.Origination,Original.Interest.Rate, col = 'yellow', data= test)
matlines(test$Borrower.Credit.Score.at.Origination, pp)
matlines(test$Borrower.Credit.Score.at.Origination, pc)
title(main = "Model Predictions with Confidence and prediction Bands")


predictionCS <-predict(lmCS, newdata = test)
View(predictionCS)
cor(predictionCS, test$Original.Interest.Rate)
mse <- mean((predictionCS-test$Original.Interest.Rate)^2)
mse
rs <- residuals(lmCS)
qqnorm(rs, main ="")
title(main = "Q-Q Plot: Residuals of Borrower Credit Score Linear Model")
qqline(rs)
summary(lmCS)

#MULTIVARIATE REGRESSION ANALYSIS

attach(trainFactors)
lmUPB_CS <- lm(Original.Interest.Rate~ Original.UPB+Borrower.Credit.Score.at.Origination)
summary(lmUPB_CS)

lm6Ivars <- lm(Original.Interest.Rate~ Original.UPB+Borrower.Credit.Score.at.Origination+Property.State+Property.Type+Seller.Name+Occupancy.Status)
summary(lm6Ivars)


lm4Ivars <- lm(Original.Interest.Rate~ Original.UPB+Borrower.Credit.Score.at.Origination+Seller.Name+Occupancy.Status)
summary(lm4Ivars)


lmSN <- lm(Original.Interest.Rate~Seller.Name)
summary(lmSN)

lmSNOS <- lm(Original.Interest.Rate~Seller.Name+Occupancy.Status)
summary(lmSNOS)


lmSNOSUPB <- lm(Original.Interest.Rate~Seller.Name+Occupancy.Status+Original.UPB)
summary(lmSNOSUPB)


#Realising that we have to find independent variables that may have better predictive powe
#OUr original Final file with shortlisted variables that contains Loan Origination date for performing vlookup
data <- read.csv("Final_FM_AP_R_2022Q4.csv")
dim(data)


#NEW Variable file created: New_variables_FM_AP_R_2022Q4.csv was created using vlookup on Loan Origination Date to save time
# This file contains all the 13 variables used previously and the 3 new variables infused from 3 datasets
data <- read.csv("New_Variables_FM_AP_R_2022Q4.csv")
dim(data) #NULLS NOT VERIFIED
# VERIFYING AND Removing Nulls from New_variables file and splitting into Training and Test data
nullValues <- colSums(is.na(data))
if (any(nullValues >0)) {
  print(nullValues)
} else {
  print("No Null values")
}
# Removing 364 Null from Credit score AND 11 FROM DTI
data <- data[complete.cases(data$Borrower.Credit.Score.at.Origination),]
data <- data[complete.cases(data$Debt.To.Income..DTI.),]
dim(data) # ARRIVING AT THE NUMBER OF OBSERVATIONS USED FOR LINEAR REGRESSION
# Running is.na() function on dataset to verify if dataset has ANY null VALUES
nullValues <- colSums(is.na(data))
if (any(nullValues >0)) {
  print(nullValues)
} else {
  print("No Null values exist in the dataframe")
}
glimpse(data)
View(data)
#REMOVING NA VALUES FROM THE DATASET ( our dataset has #NA values)
data[data == "#N/A"] <- NA
data <- na.omit(data)
dim(data)
View(data)
glimpse(data)
#Bond Rate and FED Rate variables have char datatype. Changing them
data$bondRate <- as.double(data$bondRate)
glimpse(data)
data$fedFundsRate <- as.double(data$fedFundsRate)
glimpse(data)

#converting categorical variables as factors for plotting later and creating an intermetidiary csv file

data <- data %>%
  mutate(Property.State = as.factor(Property.State),
         Seller.Name = as.factor(Seller.Name),
         Loan.Purpose = as.factor(Loan.Purpose),
         Occupancy.Status = as.factor(Occupancy.Status),
         Property.Type = as.factor(Property.Type))
write.csv(data, file = "NoNullsDataAsFactors.csv", row.names = FALSE)

# Removing outliers from NUMERICAL variables in the new dataset
data <- read.csv("NoNullsDataAsFactors.csv")
dim(data)
#Original Interest Rate
iqr <- IQR(data$Original.Interest.Rate)
upperBound <- quantile(data$Original.Interest.Rate, 0.75) + 1.5*iqr
lowerBound <- quantile(data$Original.Interest.Rate, 0.25) - 1.5*iqr
data <- subset(data, data$Original.Interest.Rate > lowerBound & data$Original.Interest.Rate < upperBound)
dim(data)

#Original UPB
iqr <- IQR(data$Original.UPB)
upperBound <- quantile(data$Original.UPB, 0.75) + 1.5*iqr
lowerBound <- quantile(data$Original.UPB, 0.25) - 1.5*iqr
data <- subset(data, data$Original.UPB > lowerBound & data$Original.UPB < upperBound)
dim(data)

#Original LTV
iqr <- IQR(data$Original.Loan.to.Value.Ratio..LTV.)
upperBound <- quantile(data$Original.Loan.to.Value.Ratio..LTV., 0.75) + 1.5*iqr
lowerBound <- quantile(data$Original.Loan.to.Value.Ratio..LTV., 0.25) - 1.5*iqr
data <- subset(data, data$Original.Loan.to.Value.Ratio..LTV. > lowerBound & data$Original.Loan.to.Value.Ratio..LTV. < upperBound)
dim(data)

#Original CLTV
iqr <- IQR(data$Original.Combined.Loan.to.Value.Ratio..CLTV.)
upperBound <- quantile( data$Original.Combined.Loan.to.Value.Ratio..CLTV., 0.75) + 1.5*iqr
lowerBound <- quantile(data$ Original.Combined.Loan.to.Value.Ratio..CLTV., 0.25) - 1.5*iqr
data <- subset(data, data$ Original.Combined.Loan.to.Value.Ratio..CLTV. > lowerBound & data$ Original.Combined.Loan.to.Value.Ratio..CLTV. < upperBound)
dim(data)

# Number of Borrowers
iqr <- IQR(data$Number.of.Borrowers)
upperBound <- quantile( data$Number.of.Borrowers, 0.75) + 1.5*iqr
lowerBound <- quantile(data$Number.of.Borrowers, 0.25) - 1.5*iqr
data <- subset(data, data$Number.of.Borrowers > lowerBound & data$Number.of.Borrowers < upperBound)
dim(data)

# Number of Units, THE LOWER BOUND AND UPPERBOUND ARE =1, THEREFORE 
glimpse(data)
View(data)
iqr <- IQR(data$Number.of.Units)
upperBound <- quantile( data$Number.of.Units, 0.75) + 1.5*iqr
upperBound
lowerBound <- quantile(data$Number.of.Units, 0.25) - 1.5*iqr
lowerBound
data <- subset(data,data$Number.of.Units ==1)
dim(data)

# DTI
iqr <- IQR(data$Debt.To.Income..DTI.)
upperBound <- quantile( data$Debt.To.Income..DTI., 0.75) + 1.5*iqr
upperBound
lowerBound <- quantile(data$Debt.To.Income..DTI., 0.25) - 1.5*iqr
lowerBound
data <- subset(data, data$Debt.To.Income..DTI. > lowerBound & data$Debt.To.Income..DTI. < upperBound)
dim(data)

# Credit Score
iqr <- IQR(data$Borrower.Credit.Score.at.Origination)
upperBound <- quantile( data$Borrower.Credit.Score.at.Origination, 0.75) + 1.5*iqr
upperBound
lowerBound <- quantile(data$Borrower.Credit.Score.at.Origination, 0.25) - 1.5*iqr
lowerBound
data <- subset(data, data$Borrower.Credit.Score.at.Origination > lowerBound & data$Borrower.Credit.Score.at.Origination < upperBound)
dim(data)
#Before and after BOX PLots and USING THE INTERQUARTILE RANGE METHOD TO DETECT AND REMOVE OUTLIERS FROM NEW VARIABLES
bondRatePlot <- ggplot(data) +
  aes(x = "", y = bondRate) +
  geom_boxplot(fill = "#FFF5EE") +
  labs(title = "10 Year Treasury Bond Rate plot before elimination of outliers")
theme_minimal()
bondRatePlot

#10 YEar Treasury BOND Date variable
iqr <- IQR(data$bondRate)
upperBound <- quantile(data$bondRate, 0.75) + 1.5*iqr
lowerBound <- quantile(data$bondRate, 0.25) - 1.5*iqr
data <- subset(data, data$bondRate > lowerBound & data$bondRate < upperBound)
dim(data)
#Boxplot of bondRate Variable after removal of Outliers
E_bondRatePlot <- ggplot(data) +
  aes(x = "", y = bondRate) +
  geom_boxplot(fill = "#E0FFFF") +
  labs(title = "10 Year Treasury Bond Rate plot after elimination of outliers")
theme_minimal()
E_bondRatePlot

# Federal Funds Rate variable

fedRatePlot <- ggplot(data) +
  aes(x = "", y = fedFundsRate) +
  geom_boxplot(fill = "#E0FFFF") +
  labs(title = "Federal Funds Rate plot before elimination of outliers")
theme_minimal()
fedRatePlot
#Removing outliers from FEdfunds rate variable
iqr <- IQR(data$fedFundsRate)
upperBound <- quantile(data$fedFundsRate, 0.75) + 1.5*iqr
lowerBound <- quantile(data$fedFundsRate, 0.25) - 1.5*iqr
data <- subset(data, data$fedFundsRate > lowerBound & data$fedFundsRate < upperBound)
dim(data)
#Boxplot of bondRate Variable after removal of Outliers
E_fedRatePlot <- ggplot(data) +
  aes(x = "", y = fedFundsRate) +
  geom_boxplot(fill = "#E0FFFF") +
  labs(title = "Federal Funds Rate plot after elimination of outliers")
theme_minimal()
E_fedRatePlot



#Bench Mark Mortgage Rate variable
mortgageRatePlot <- ggplot(data) +
  aes(x = "", y = benchMarkMortgageRate) +
  geom_boxplot(fill = "#E0FFFF") +
  labs(title = "30 Year Mortgage Rate plot before elimination of outliers")
theme_minimal()
mortgageRatePlot
#Removing outliers from FEdfunds rate variable
iqr <- IQR(data$benchMarkMortgageRate)
upperBound <- quantile(data$benchMarkMortgageRate, 0.75) + 1.5*iqr
lowerBound <- quantile(data$benchMarkMortgageRate, 0.25) - 1.5*iqr
data <- subset(data, data$benchMarkMortgageRate > lowerBound & data$benchMarkMortgageRate < upperBound)
dim(data)
#Boxplot of bondRate Variable after removal of Outliers
E_mortgageRatePlot <- ggplot(data) +
  aes(x = "", y = benchMarkMortgageRate) +
  geom_boxplot(fill = "#E0FFFF") +
  labs(title = "30 Year Mortgage Rate plot after elimination of outliers")
theme_minimal()
E_mortgageRatePlot
write.csv(data, file ="New_FM2022Q4.csv", row.names = FALSE)
data <- read.csv("New_FM2022Q4.csv")
dim(data)


# SPLITTING DATA INTO TRAINING AND TEST DATASETS AND WRITING TO FILES
data <- read.csv("New_FM2022Q4.csv")
testIndexNos <- which(1:nrow(data)%%4==0)
testIndexNos
#Write test data using the randomly selected row nos
testData <-data[testIndexNos, ]
dim(testData)
write.csv(testData, file = "Test_FM2022Q4.csv", row.names = FALSE)
data <- read.csv("Test_FM2022Q4.csv")
dim(data)
data <- read.csv("New_FM2022Q4.csv")
trainData <- data[-testIndexNos, ]
dim(trainData)
write.csv(trainData, file = "Training_FM2022Q4.csv", row.names = FALSE)
data <- read.csv("Training_FM2022Q4.csv")
dim(data)


#CHECKING LINEAR MODELS WITH NEW TRAINING DATASET WITH 16 VARIABLES
train <- read.csv("Training_FM2022Q4.csv")
dim(train)
names(train)
attach(train)
lmUPB <- lm(Original.Interest.Rate~Original.UPB, data =train)
summary(lmUPB)

lmLTV <- lm(Original.Interest.Rate~Original.Loan.to.Value.Ratio..LTV.,data =train)
summary(lmLTV)

lmCLTV <- lm(Original.Interest.Rate~Original.Combined.Loan.to.Value.Ratio..CLTV.,data =train)
summary(lmCLTV)

lmBorrowers <- lm(Original.Interest.Rate~Number.of.Borrowers, data =train)
summary(lmBorrowers)

lmDTI <- lm(Original.Interest.Rate~Debt.To.Income..DTI., data =train)
summary(lmDTI)

lmCS <- lm(Original.Interest.Rate~Borrower.Credit.Score.at.Origination, data =train)
summary(lmCS)

lmUnits <- lm(Original.Interest.Rate~Number.of.Units, data =train)
summary(lmUnits)

lmPS <- lm(Original.Interest.Rate~Property.State, data =train)
summary(lmPS)

lmSN <- lm(Original.Interest.Rate~Seller.Name, data =train)
summary(lmSN)


lmLP <- lm(Original.Interest.Rate~Loan.Purpose, data =train)
summary(lmLP)


lmOS <- lm(Original.Interest.Rate~Occupancy.Status, data =train)
summary(lmOS)


lmPT <- lm(Original.Interest.Rate~ Property.Type, data =train)
summary(lmPT)

lmBR <- lm(Original.Interest.Rate~ bondRate, data =train)
summary(lmBR)


lmFFR <- lm(Original.Interest.Rate~ fedFundsRate, data =train)
summary(lmFFR)

lmBMMR <- lm(Original.Interest.Rate~ benchMarkMortgageRate, data =train)
summary(lmBMMR)

model <-lm(Original.Interest.Rate~bondRate+fedFundsRate, data =train)
summary(model)

model <-lm(Original.Interest.Rate~fedFundsRate+benchMarkMortgageRate, data =train)
summary(model)

model <-lm(Original.Interest.Rate~bondRate+benchMarkMortgageRate, data =train)
summary(model)


model <-lm(Original.Interest.Rate~bondRate+fedFundsRate+benchMarkMortgageRate+Borrower.Credit.Score.at.Origination+Original.UPB, data =train)
summary(model)

model <-lm(Original.Interest.Rate~bondRate+fedFundsRate+benchMarkMortgageRate+Borrower.Credit.Score.at.Origination, data =train)
summary(model)

model <-lm(Original.Interest.Rate~bondRate+fedFundsRate+Borrower.Credit.Score.at.Origination+Original.UPB, data =train)
summary(model)

model <-lm(Original.Interest.Rate~bondRate+fedFundsRate+Borrower.Credit.Score.at.Origination+Original.UPB+Seller.Name, data =train)
summary(model)


model <-lm(Original.Interest.Rate~bondRate+fedFundsRate+Seller.Name, data =train)
summary(model)

model <-lm(Original.Interest.Rate~bondRate+Seller.Name, data =train)
summary(model)

model <-lm(Original.Interest.Rate~fedFundsRate+Seller.Name, data =train)
summary(model)


model <-lm(Original.Interest.Rate~benchMarkMortgageRate+Seller.Name, data =train)
summary(model)

#DISTRIBUTION OF RESIDUALS OF THE NEW VARIABLES WITH PREDICTIVE POWER
#Checking if residuals of the bondRate model follow a Guassian Distribution
rs <- residuals(lmBR)
qqnorm(rs, main ="")
title(main = "Q-Q Plot: Residuals of 10 Year Treasury Bond Rate and Interest Rate Linear Model")
qqline(rs)

#Checking if residuals of the fedFundsRate model follow a Guassian Distribution
rs <- residuals(lmFFR)
qqnorm(rs, main ="")
title(main = "Q-Q Plot: Residuals of Federal Funds Rate and Interest Rate Linear Model")
qqline(rs)

#Checking if residuals of the benchMarkMortgageRate model follow a Guassian Distribution
rs <- residuals(lmBMMR)
qqnorm(rs, main ="")
title(main = "Q-Q Plot: Residuals of Bench Mark Mortgage Rate and Interest Rate Linear Model")
qqline(rs)


#Using the bondRate linear model to predict Interest rate  on Test data
train <- read.csv("Training_FM2022Q4.csv")
test <- read.csv("Test_FM2022Q4.csv")

prediction <- predict(lmBR, newdata = test)
View(prediction)
#Checking for correlation between predicted and actual values of dependent variable
cor(prediction, test$Original.Interest.Rate)
mse <- mean((prediction-test$Original.Interest.Rate)^2)
mse

## Confidence and Prediction Bands for bondRate Linear MOdel

pc <- predict(lmBR, int = "c", newdata = test)
pp <- predict(lmBR, int = "p", newdata = test)
attach(test)
plot(bondRate,Original.Interest.Rate)
matlines(test$bondRate, pp)
matlines(test$bondRate, pc)
title(main = "Model Predictions with Confidence and prediction Bands")




#Using the fedFundsRate linear model to predict Interest rate  on Test data
prediction <- predict(lmFFR, newdata = test)
View(prediction)
#Checking for correlation between predicted and actual values of dependent variable
cor(prediction, test$Original.Interest.Rate)
mse <- mean((prediction-test$Original.Interest.Rate)^2)
mse

## Confidence and Prediction Bands for fedFundsRate Linear MOdel
pc <- predict(lmFFR, int = "c", newdata = test)
pp <- predict(lmFFR, int = "p", newdata = test)
attach(test)
plot(fedFundsRate,Original.Interest.Rate)
matlines(test$fedFundsRate, pp)
matlines(test$fedFundsRate, pc)
title(main = "Model Predictions with Confidence and prediction Bands")



#Using the benchMarkMortgageRate linear model to predict Interest rate  on Test data
prediction <- predict(lmBMMR, newdata = test)
View(prediction)
#Checking for correlation between predicted and actual values of dependent variable
cor(prediction, test$Original.Interest.Rate)
mse <- mean((prediction-test$Original.Interest.Rate)^2)
mse

## Confidence and Prediction Bands for fedFundsRate Linear MOdel
pc <- predict(lmBMMR, int = "c", newdata = test)
pp <- predict(lmBMMR, int = "p", newdata = test)
attach(test)
plot(benchMarkMortgageRate,Original.Interest.Rate)
matlines(test$benchMarkMortgageRate, pp)
matlines(test$benchMarkMortgageRate, pc)
title(main = "Model Predictions with Confidence and prediction Bands")

#MULTIVARIATE REGRESSION ANALYSIS WITH VARIABLES THAT HAVE BETTER PREDICTIVE POWER
attach(train)
#In this model we observe that coeff of benchMarkMortgagerate is not displayed due to singularities
model <-lm(Original.Interest.Rate~bondRate+fedFundsRate+benchMarkMortgageRate)
summary(model)

#TESTING FOR MULTICOLLINEARITY (PERFECT CORRELATION BETWEEN IVS AND FIND THAT FEDFUNDSRATE AND BENCHMARKMORTGAGE RATE HAVE PERFECT COLLINEARITIES)
cor(bondRate, benchMarkMortgageRate)
cor(bondRate, fedFundsRate)
cor(fedFundsRate, benchMarkMortgageRate)



##CORRELATION AND MSE OF MODELS WITH HIGH PREDICTIVE POWER


#uSING bondRate + fedFundsRate+ Borrower Credit Score+ UPB + Seller Name  & Original Interest Rate MODEL TO PREDICT
model <-lm(Original.Interest.Rate~bondRate+fedFundsRate+Borrower.Credit.Score.at.Origination+Original.UPB+Seller.Name, data =train)
prediction <- predict(model, newdata = test)
#Checking for correlation between predicted and actual values of dependent variable
cor(prediction, test$Original.Interest.Rate)
mse <- mean((prediction-test$Original.Interest.Rate)^2)
mse

#uSING bondRate + fedFundsRate+  Seller Name  & Original Interest Rate MODEL TO PREDICT
model <-lm(Original.Interest.Rate~bondRate+fedFundsRate+Seller.Name, data =train)
prediction <- predict(model, newdata = test)
#Checking for correlation between predicted and actual values of dependent variable
cor(prediction, test$Original.Interest.Rate)
mse <- mean((prediction-test$Original.Interest.Rate)^2)
mse


#uSING bondRate + fedFundsRate+Borrower Credit Score+UPB  & Original Interest Rate MODEL TO PREDICT
model <-lm(Original.Interest.Rate~bondRate+fedFundsRate+Borrower.Credit.Score.at.Origination+Original.UPB, data =train)
prediction <- predict(model, newdata = test)
#Checking for correlation between predicted and actual values of dependent variable
cor(prediction, test$Original.Interest.Rate)
mse <- mean((prediction-test$Original.Interest.Rate)^2)
mse

#uSING bondRate + fedFundsRateB  & Original Interest Rate MODEL TO PREDICT
model <-lm(Original.Interest.Rate~bondRate+fedFundsRate, data =train)
prediction <- predict(model, newdata = test)
#Checking for correlation between predicted and actual values of dependent variable
cor(prediction, test$Original.Interest.Rate)
mse <- mean((prediction-test$Original.Interest.Rate)^2)
mse

#uSING bondRate + Seller Name  & Original Interest Rate MODEL TO PREDICT
model <-lm(Original.Interest.Rate~bondRate+ Seller.Name, data =train)
prediction <- predict(model, newdata = test)
#Checking for correlation between predicted and actual values of dependent variable
cor(prediction, test$Original.Interest.Rate)
mse <- mean((prediction-test$Original.Interest.Rate)^2)
mse

#uSING fedFundsRate + BenchMarkMortgage Rate  & Original Interest Rate MODEL TO PREDICT
model <-lm(Original.Interest.Rate~ fedFundsRate+ benchMarkMortgageRate, data =train)
prediction <- predict(model, newdata = test)
#Checking for correlation between predicted and actual values of dependent variable
cor(prediction, test$Original.Interest.Rate)
mse <- mean((prediction-test$Original.Interest.Rate)^2)
mse

#uSING bondRate + BenchMarkMortgage Rate  & Original Interest Rate MODEL TO PREDICT
model <-lm(Original.Interest.Rate~ bondRate+ benchMarkMortgageRate, data =train)
prediction <- predict(model, newdata = test)
#Checking for correlation between predicted and actual values of dependent variable
cor(prediction, test$Original.Interest.Rate)
mse <- mean((prediction-test$Original.Interest.Rate)^2)
mse

#uSING fedFundsRate + Seller Name  & Original Interest Rate MODEL TO PREDICT
model <-lm(Original.Interest.Rate~ fedFundsRate+ Seller.Name, data =train)
prediction <- predict(model, newdata = test)
#Checking for correlation between predicted and actual values of dependent variable
cor(prediction, test$Original.Interest.Rate)
mse <- mean((prediction-test$Original.Interest.Rate)^2)
mse

#uSING Bench Mark Mortgage Rate + Seller Name  & Original Interest Rate MODEL TO PREDICT
model <-lm(Original.Interest.Rate~ benchMarkMortgageRate+ Seller.Name, data =train)
prediction <- predict(model, newdata = test)
#Checking for correlation between predicted and actual values of dependent variable
cor(prediction, test$Original.Interest.Rate)
mse <- mean((prediction-test$Original.Interest.Rate)^2)
mse

############################################################
# Reduced Dataset removing collinear variables Number of Units, Loan Purpose, Bench Mark Mortgage Rate
library(dplyr)
data <- read.csv("New_FM2022Q4.csv")
dim(data)
names(data)
data <- subset(data, select = -c(Number.of.Units, Loan.Purpose, benchMarkMortgageRate))
dim(data)
names(data)
write.csv(data, file = "NC_FM2022Q4.csv", row.names = FALSE)
data <-read.csv("NC_FM2022Q4.csv")
dim(data)
#CHECKING IF COLLINEAR VARIABLES EXIST in the entire dataset before splitting AND FINDING NONE
model <-lm(Original.Interest.Rate~.,data = data)
summary(model)

#SPLITTING DATA INTO TEST AND TRAINING
#ENSURE RANDOMNESS IN THE  DATA FIRST
#To ensure we get the same set of random nos every time so that our results will be same  We use the set seed function
set.seed(123)
#CREATE RANDOM NOS USING nrow() to create nrows of random nos
randomNos <- runif(nrow(data)) 
#USE randomNos to SHUFFLE DATA, 
datar <- data[order(randomNos), ]
str(datar)
train <- datar[1:108097, ] #75% of the dataset
write.csv(train, file = "NC_Train_FM2022Q4.csv", row.names = FALSE)
data <- read.csv("NC_Train_FM2022Q4.csv")
dim(data)
test <- datar[108098:144130, ] #25% of the dataset
write.csv(test, file= "NC_Test_FM2022Q4.csv", row.names = FALSE)
data <- read.csv("NC_Test_FM2022Q4.csv")
dim(data)

########REGULARIZATION
# Train Model on the complete Training dataset
train <- read.csv("NC_Train_FM2022Q4.csv")
test <- read.csv("NC_Test_FM2022Q4.csv")
model <-lm(Original.Interest.Rate~.,data = train)
summary(model)
#CHeck Residuals' distribution follow a Guassian model
rs <- residuals(model)
qqnorm(rs, main ="")
title(main = "Q-Q Plot: Residuals of Training Dataset Linear Model")
qqline(rs)

#Predicting output using the linear model
prediction <- predict(model, newdata = test)
cor(prediction, test$Original.Interest.Rate)
mse <- mean((prediction-test$Original.Interest.Rate)^2)
mse

#ReGULARIZATION ONLY FOR NUMERIC VARIABLES
train <- read.csv("NC_Train_FM2022Q4.csv")
names(train)
test <- read.csv("NC_Test_FM2022Q4.csv")
names(test)

library(glmnet)
cv.fit <- cv.glmnet(as.matrix(train[,c(-1,-2,-11,-12,-13)]),
                    as.vector(train[,2]),
                    alpha = 1)

cv.fit
cv.fit$lambda.min
plot(cv.fit)
coef(cv.fit)


#TESTING PREDICTIVE POWER OF CV.FIT
predictions <- predict(cv.fit, s= cv.fit$lambda.min,
                       newx=as.matrix(test[,c(-1,-2,-11,-12,-13)]))

View(predictions)
cor(predictions,as.vector(test[,2]))
mse <- mean((predictions-test$Original.Interest.Rate)^2)
mse


########################   1D  #########
# TESTING LINEAR REGRESSION, MULTIVARIATE REGRESSION AND REGULARIZATION WITH DIFFERENT TRAINING AND TESTING DATASETS
data <- read.csv("NC_FM2022Q4.csv")
numofsets <- 3
# Creating empty lists to store the datasets
train_datasets <- list()
test_datasets <- list()

# Creating and storing multiple training and testing datasets
for (i in 1:numofsets) {
  
  sample_size <- floor(0.75 * nrow(data))  # 75% for training
  train_indices <- sample(1:nrow(data), size = sample_size, replace = FALSE)
  
  train_data <- data[train_indices, ]
  test_data <- data[-train_indices, ]
  
  # Storing the datasets in the lists
  train_datasets[[i]] <- train_data
  test_datasets[[i]] <- test_data
  
  # Writing the datasets to CSV files
  train_file <- paste0("train_data_", i, ".csv")
  test_file <- paste0("test_data_", i, ".csv")
  write.csv(train_data, file = train_file, row.names = FALSE)
  write.csv(test_data, file = test_file, row.names = FALSE)
}
train_data_1 <- read.csv("train_data_1.csv")
dim(train_data_1)

train_data_2 <- read.csv("train_data_2.csv")
dim(train_data_2)

train_data_3 <- read.csv("train_data_3.csv")
dim(train_data_3)

test_data_1 <- read.csv("test_data_1.csv")
dim(test_data_1)
test_data_2 <- read.csv("test_data_2.csv")
dim(test_data_2)

test_data_3 <- read.csv("test_data_3.csv")
dim(test_data_3)

#Training Linear and Multivariate Models on most predictive variables

#Using the bondRate linear model to train on Training data and predict Interest rate  on Test data
#RUn-1
lmBR1 <- lm(Original.Interest.Rate~bondRate, data = train_data_1)
summary(lmBR1)
prediction1 <- predict(lmBR1, newdata = test_data_1)
#Checking for correlation between predicted and actual values of dependent variable
cor(prediction1, test_data_1$Original.Interest.Rate)
mse <- mean((prediction1-test_data_1$Original.Interest.Rate)^2)
mse

#RUn-2
lmBR2 <- lm(Original.Interest.Rate~bondRate, data = train_data_2)
summary(lmBR2)
prediction2 <- predict(lmBR2, newdata = test_data_2)
#Checking for correlation between predicted and actual values of dependent variable
cor(prediction2, test_data_2$Original.Interest.Rate)
mse <- mean((prediction2-test_data_2$Original.Interest.Rate)^2)
mse

#RUn-3
lmBR3 <- lm(Original.Interest.Rate~bondRate, data = train_data_3)
summary(lmBR3)
prediction3 <- predict(lmBR3, newdata = test_data_3)
#Checking for correlation between predicted and actual values of dependent variable
cor(prediction3, test_data_3$Original.Interest.Rate)
mse <- mean((prediction3-test_data_3$Original.Interest.Rate)^2)
mse


#Using the fedFundsRate linear model to predict Interest rate  on Test data
#RUn-1
lmFFR1 <- lm(Original.Interest.Rate~fedFundsRate, data = train_data_1)
summary(lmFFR1)
#Checking for correlation between predicted and actual values of dependent variable
prediction1 <- predict(lmFFR1, newdata = test_data_1)
cor(prediction1, test_data_1$Original.Interest.Rate)
mse <- mean((prediction1-test_data_1$Original.Interest.Rate)^2)
mse

#RUn-2
lmFFR2 <- lm(Original.Interest.Rate~fedFundsRate, data = train_data_2)
summary(lmFFR2)
#Checking for correlation between predicted and actual values of dependent variable
prediction2 <- predict(lmFFR2, newdata = test_data_2)
cor(prediction2, test_data_2$Original.Interest.Rate)
mse <- mean((prediction2-test_data_2$Original.Interest.Rate)^2)
mse

#RUn-3
lmFFR3 <- lm(Original.Interest.Rate~fedFundsRate, data = train_data_3)
summary(lmFFR3)
#Checking for correlation between predicted and actual values of dependent variable
prediction3 <- predict(lmFFR3, newdata = test_data_3)
cor(prediction3, test_data_3$Original.Interest.Rate)
mse <- mean((prediction3-test_data_3$Original.Interest.Rate)^2)
mse


#uSING bondRate + fedFundsRate+ Borrower Credit Score+ UPB + Seller Name  & Original Interest Rate MODEL TO PREDICT
#Run-1
model1 <-lm(Original.Interest.Rate~bondRate+fedFundsRate+Borrower.Credit.Score.at.Origination+Original.UPB+Seller.Name,
            data =train_data_1)
summary(model1)
#Checking for correlation between predicted and actual values of dependent variable
prediction1 <- predict(model1, newdata = test_data_1)
cor(prediction1, test_data_1$Original.Interest.Rate)
mse <- mean((prediction1-test_data_1$Original.Interest.Rate)^2)
mse


#Run-2
model2 <-lm(Original.Interest.Rate~bondRate+fedFundsRate+Borrower.Credit.Score.at.Origination+Original.UPB+Seller.Name,
            data =train_data_2)
summary(model2)
#Checking for correlation between predicted and actual values of dependent variable
prediction2 <- predict(model2, newdata = test_data_2)
cor(prediction2, test_data_2$Original.Interest.Rate)
mse <- mean((prediction2-test_data_2$Original.Interest.Rate)^2)
mse

#Run-3
model3 <-lm(Original.Interest.Rate~bondRate+fedFundsRate+Borrower.Credit.Score.at.Origination+Original.UPB+Seller.Name,
            data =train_data_3)
summary(model3)
#Checking for correlation between predicted and actual values of dependent variable
prediction3 <- predict(model3, newdata = test_data_3)
cor(prediction3, test_data_3$Original.Interest.Rate)
mse <- mean((prediction3-test_data_3$Original.Interest.Rate)^2)
mse

##REGULARIZATION WITH MULTIPLE DATASETS

#REGULARIZATION ONLY FOR NUMERIC VARIABLES

library(glmnet)
names(train_data_1)
#Run-1
cv.fit1 <- cv.glmnet(as.matrix(train_data_1[,c(-1,-2,-11,-12,-13)]),
                     as.vector(train_data_1[,2]),
                     alpha = 1)
cv.fit1
cv.fit1$lambda.min
coef(cv.fit1)


#TESTING PREDICTIVE POWER OF CV.FIT
predictions1 <- predict(cv.fit1, s= cv.fit1$lambda.min,
                        newx=as.matrix(test_data_1[,c(-1,-2,-11,-12,-13)]))

cor(predictions1,as.vector(test_data_1[,2]))
mse <- mean((predictions1-test_data_1$Original.Interest.Rate)^2)
mse

#Run-2
cv.fit2 <- cv.glmnet(as.matrix(train_data_2[,c(-1,-2,-11,-12,-13)]),
                     as.vector(train_data_2[,2]),
                     alpha = 1)
cv.fit2
cv.fit2$lambda.min
coef(cv.fit2)


#TESTING PREDICTIVE POWER OF CV.FIT
predictions2 <- predict(cv.fit2, s= cv.fit2$lambda.min,
                        newx=as.matrix(test_data_2[,c(-1,-2,-11,-12,-13)]))

cor(predictions2,as.vector(test_data_2[,2]))
mse <- mean((predictions2-test_data_2$Original.Interest.Rate)^2)
mse

#Run-3
cv.fit3 <- cv.glmnet(as.matrix(train_data_3[,c(-1,-2,-11,-12,-13)]),
                     as.vector(train_data_3[,2]),
                     alpha = 1)
cv.fit3
cv.fit3$lambda.min
coef(cv.fit3)


#TESTING PREDICTIVE POWER OF CV.FIT
predictions3 <- predict(cv.fit3, s= cv.fit3$lambda.min,
                        newx=as.matrix(test_data_3[,c(-1,-2,-11,-12,-13)]))

cor(predictions3,as.vector(test_data_3[,2]))
mse <- mean((predictions3-test_data_3$Original.Interest.Rate)^2)
mse


#################################################
##LOGISTIC REGRESSION WITH NON MULTI COLLINEAR DATASET##################
train <- read.csv("NC_Train_FM2022Q4.csv")
dim(train)
test <- read.csv("NC_Test_FM2022Q4.csv")
dim(test)
names(train)

###checking levels of categorical variables 
train$Seller.Name <- as.factor(train$Seller.Name)
train$Occupancy.Status <- as.factor(train$Occupancy.Status)
levels(train$Occupancy.Status)

test$Seller.Name <- as.factor(test$Seller.Name)
test$Occupancy.Status <- as.factor(test$Occupancy.Status)
levels(test$Occupancy.Status)

####MULTINOMIAL LOGISTIC REGRESSION AS PER UCLA STATISTICAL METHODS AND DA ARTICLE
library(nnet)
library(foreign)
library(reshape2)
#SELECTING VARIABLES FROM TRAIN DATASET AND STORING IN trainLogistic dataset
library(dplyr)
dim(train)
trainLogistic <- subset(train,
  select = c(Original.Interest.Rate,bondRate,fedFundsRate,Borrower.Credit.Score.at.Origination,Original.UPB,Occupancy.Status))
trainLogistic$Occupancy.Status <-as.factor(trainLogistic$Occupancy.Status)
dim(trainLogistic)



testLogistic <- subset(test,
  select= c(Original.Interest.Rate,bondRate,fedFundsRate,Borrower.Credit.Score.at.Origination,Original.UPB,Occupancy.Status))
testLogistic$Occupancy.Status <-as.factor(testLogistic$Occupancy.Status)
dim(testLogistic)

#Using table command to generate a frequency table for the outcome variable.
with(trainLogistic, table(Occupancy.Status))

#DISPLAYING MEAN AND SD OF THE DATASET GROUPING BY OCCUPANCY STATUS

result <- trainLogistic %>%
  group_by(Occupancy.Status) %>%
  summarise(
    M_oir = mean(Original.Interest.Rate),
    SD_oir = sd(Original.Interest.Rate),
    M_br = mean(bondRate),
    SD_br = sd(bondRate),
    M_ffr = mean(fedFundsRate),
    SD_ffr = sd(fedFundsRate),
    M_cs = mean(Borrower.Credit.Score.at.Origination),
    SD_cs = sd(Borrower.Credit.Score.at.Origination),
    M_upb = mean(Original.UPB),
    SD_upb = sd(Original.UPB)
  )
result

##TRAINING LOGISTIC MODEL ON TRAINLOGISTIC DATA

logisticModel <- multinom(Occupancy.Status~ Original.Interest.Rate+bondRate+
                            fedFundsRate+Borrower.Credit.Score.at.Origination+Original.UPB,
                          data = trainLogistic)

summary(logisticModel)
#Since summary does not provide p-values, calculating z-statistic
z <- summary(logisticModel)$coefficients/summary(logisticModel)$standard.errors
z
#Verifying if P value is < than Alpha (0.05)
p <- (1 - pnorm(abs(z), 0, 1)) * 2
p


#EXPONENTIATING COEFFS AND INTERPRET AS ODD RATOS
exp(coef(logisticModel))

# Make predictions on test_data
classPredictions <- predict(logisticModel, newdata = testLogistic, type = "class")


library(caret)
true_labels <- testLogistic$Occupancy.Status
cMatrix <- confusionMatrix(classPredictions, true_labels) 
cMatrix

probsPredictions <- predict(logisticModel, newdata = testLogistic, type = "probs")
head(probsPredictions)

####### LOGISTIC REGRESSION INFUSING MORE VARIABLES FROM THE DATASET########################

##########SECOND ATTEMPT#######################
##LOGISTIC REGRESSION
data <-  read.csv("New_FM2022Q4.csv")
dim(data)
set.seed(1234)
trainProportion <- 0.75
indexSet <- 1:nrow(data)
trainIndices <- sample(indexSet, size = round(trainProportion*length(indexSet)))
testIndices <- setdiff(indexSet, trainIndices)
trainLogistic <- data[trainIndices, ]
dim(trainLogistic)
testLogistic <- data[testIndices, ]
dim(testLogistic)
write.csv(trainLogistic, file = "Log_Train_FM2022Q4.csv", row.names = FALSE)
write.csv(testLogistic, file = "Log_Test_FM2022Q4.csv", row.names = FALSE)

train <- read.csv("Log_Train_FM2022Q4.csv")
dim(train)
test <- read.csv("Log_Test_FM2022Q4.csv")
dim(test)
names(train)

#SELECTING VARIABLES FROM TRAIN DATASET AND STORING IN trainLogistic dataset
trainLogistic <- subset(train,
                        select= c(Original.Interest.Rate,bondRate,fedFundsRate,Borrower.Credit.Score.at.Origination,Original.UPB,Loan.Purpose,Property.Type,Occupancy.Status))
trainLogistic$Occupancy.Status <-as.factor(trainLogistic$Occupancy.Status)
trainLogistic$Property.Type <- as.factor(trainLogistic$Property.Type)
trainLogistic$Loan.Purpose <- as.factor(trainLogistic$Loan.Purpose)
dim(trainLogistic)



testLogistic <- subset(test,
                       select=c(Original.Interest.Rate,bondRate,fedFundsRate,Borrower.Credit.Score.at.Origination,Original.UPB,Loan.Purpose,Property.Type,Occupancy.Status))
testLogistic$Occupancy.Status <-as.factor(testLogistic$Occupancy.Status)
testLogistic$Property.Type <-as.factor(testLogistic$Property.Type)
testLogistic$Loan.Purpose <- as.factor(testLogistic$Loan.Purpose)
dim(testLogistic)

#Using table command to generate a frequency table for the outcome variable.
with(trainLogistic, table(Occupancy.Status))

#DISPLAYING MEAN AND SD OF THE DATASET GROUPING BY OCCUPANCY STATUS

result <- trainLogistic %>%
  group_by(Occupancy.Status) %>%
  summarise(
    M_oir = mean(Original.Interest.Rate),
    SD_oir = sd(Original.Interest.Rate),
    M_br = mean(bondRate),
    SD_br = sd(bondRate),
    M_ffr = mean(fedFundsRate),
    SD_ffr = sd(fedFundsRate),
    M_cs = mean(Borrower.Credit.Score.at.Origination),
    SD_cs = sd(Borrower.Credit.Score.at.Origination),
    M_upb = mean(Original.UPB),
    SD_upb = sd(Original.UPB)
  )
result

##TRAINING LOGISTIC MODEL ON TRAINLOGISTIC DATA

logisticModel <- multinom(Occupancy.Status~ Original.Interest.Rate+bondRate+
                            fedFundsRate+Borrower.Credit.Score.at.Origination+Original.UPB+Property.Type+Loan.Purpose,
                          data = trainLogistic)

summary(logisticModel)


#Since summary does not provide p-values, calculating z-statistic
z <- summary(logisticModel)$coefficients/summary(logisticModel)$standard.errors
z
#Verifying if P value is < than Alpha (0.05)
p <- (1 - pnorm(abs(z), 0, 1)) * 2
p


#EXPONENTIATING COEFFS AND INTERPRET AS ODD RATOS
exp(coef(logisticModel))

# Make predictions on test_data
classPredictions <- predict(logisticModel, newdata = testLogistic, type = "class")
CrossTable(testLogistic$Occupancy.Status, classPredictions,
           prop.chisq = FALSE, prop.c = FALSE, 
           prop.r = FALSE, 
           dnn = c('actual Class', 'predicted Class'))


###########using entire dataset to train the model

logisticModel <- multinom(Occupancy.Status~ .,
                          data = train)

summary(logisticModel)

classPredictions <- predict(logisticModel, newdata = test, type = "class")
CrossTable(test$Occupancy.Status, classPredictions,
           prop.chisq = FALSE, prop.c = FALSE, 
           prop.r = FALSE, 
           dnn = c('actual Class', 'predicted Class'))



######################### NAIVE BAYES CLASSIFIER################ first attempt###############

library(e1071)

#SPLITTING DATA INTO TEST AND TRAINING DATASETS

#Alternative method to create Training and test sets using sample function
data <-  read.csv("New_FM2022Q4.csv")
dim(data)
set.seed(1234)
trainProportion <- 0.75
indexSet <- 1:nrow(data)
trainIndices <- sample(indexSet, size = round(trainProportion*length(indexSet)))
testIndices <- setdiff(indexSet, trainIndices)
trainNB <- data[trainIndices, ]
dim(trainNB)
testNB <- data[testIndices, ]
dim(testNB)
write.csv(trainNB, file = "NB_Train_FM2022Q4.csv", row.names = FALSE)
write.csv(testNB, file = "NB_Test_FM2022Q4.csv", row.names = FALSE)

trainNB <- read.csv("NB_Train_FM2022Q4.csv")
dim(trainNB)

testNB <- read.csv("NB_Test_FM2022Q4.csv")
dim(testNB)


#filtering out unpredictive variables based on lessons learned and converting outcome variable as factor
trainNB <- subset(trainNB,
  select= c(Original.Interest.Rate,bondRate,fedFundsRate,Borrower.Credit.Score.at.Origination,Original.UPB,Occupancy.Status))
trainNB$Occupancy.Status <-as.factor(trainNB$Occupancy.Status)
dim(trainNB)


testNB <- subset(testNB,
  select=c(Original.Interest.Rate,bondRate,fedFundsRate,Borrower.Credit.Score.at.Origination,Original.UPB,Occupancy.Status))
testNB$Occupancy.Status <-as.factor(testNB$Occupancy.Status)
dim(testNB)

levels(trainNB$Occupancy.Status)
levels(testNB$Occupancy.Status)

# TRAINING MODEL USING NAIVEBAYES FUNCTION, laplace =0
modelNB <- naiveBayes(Occupancy.Status~ Original.Interest.Rate+bondRate+fedFundsRate+
                        Borrower.Credit.Score.at.Origination+Original.UPB, data= trainNB)


modelNB

predictions <- predict(modelNB, newdata = testNB)

CrossTable(testNB$Occupancy.Status, predictions,
           prop.chisq = FALSE, prop.c = FALSE, 
           prop.r = FALSE, 
           dnn = c('actual Class', 'predicted Class'))

cMatrixNB <- table(Actual = testNB$Occupancy.Status, Predicted = predictions)
cMatrixNB


modelNBL <- naiveBayes(Occupancy.Status~ Original.Interest.Rate+bondRate+fedFundsRate+
                         Borrower.Credit.Score.at.Origination+Original.UPB, data= trainNB,laplace=1)

modelNBL
predictionsL <- predict(modelNBL, newdata = testNB)

CrossTable(testNB$Occupancy.Status, predictionsL,
           prop.chisq = FALSE, prop.c = FALSE, 
           prop.r = FALSE, 
           dnn = c('actual Class', 'predicted Class'))



cMatrixNBL <- table(Actual = testNB$Occupancy.Status, Predicted = predictionsL)
cMatrixNBL



library(gmodels)
CrossTable(testNB$Occupancy.Status, predictionsL,
           prop.chisq = FALSE, prop.c = FALSE, 
           prop.r = FALSE, 
           dnn = c('actual Class', 'predicted Class'))


attach(trainNB)
cor(Original.Interest.Rate,fedFundsRate)
cor(Original.Interest.Rate,bondRate)
cor(bondRate,fedFundsRate)
cor(Original.Interest.Rate,Borrower.Credit.Score.at.Origination)
cor(Original.Interest.Rate,Original.UPB)
#################naive bayes second attempt##########################

##################NAIVE BAYES SECOND ATTEMPT#####################

library(e1071)
trainNB <- read.csv("NB_Train_FM2022Q4.csv")
dim(trainNB)

testNB <- read.csv("NB_Test_FM2022Q4.csv")
dim(testNB)
#filtering out variables based on lessons learned and converting outcome variable as factor
trainNB <- subset(trainNB,
                  select=c(Original.Interest.Rate,bondRate,fedFundsRate,Borrower.Credit.Score.at.Origination,Original.UPB,Property.Type, Loan.Purpose,Occupancy.Status))
trainNB$Occupancy.Status <-as.factor(trainNB$Occupancy.Status)
trainNB$Property.Type <- trainNB$Property.Type
trainNB$Loan.Purpose <- as.factor(trainNB$Loan.Purpose)
dim(trainNB)


testNB <- subset(testNB,
                 select=c(Original.Interest.Rate,bondRate,fedFundsRate,Borrower.Credit.Score.at.Origination,Original.UPB,Property.Type, Loan.Purpose,Occupancy.Status))
testNB$Occupancy.Status <-as.factor(testNB$Occupancy.Status)
testNB$Loan.Purpose <- as.factor(testNB$Loan.Purpose)
testNB$Property.Type <- as.factor(testNB$Property.Type)
dim(testNB)

levels(trainNB$Occupancy.Status)
levels(testNB$Occupancy.Status)

# TRAINING MODEL USING NAIVEBAYES FUNCTION, without laplace

modelNB <- naiveBayes(Occupancy.Status~ Original.Interest.Rate+bondRate+fedFundsRate+
                        Borrower.Credit.Score.at.Origination+Loan.Purpose+Property.Type+Original.UPB, data= trainNB)


modelNB

predictions <- predict(modelNB, newdata = testNB)

CrossTable(testNB$Occupancy.Status, predictions,
           prop.chisq = FALSE, prop.c = FALSE, 
           prop.r = FALSE, 
           dnn = c('actual Class', 'predicted Class'))


modelNBL <- naiveBayes(Occupancy.Status~ Original.Interest.Rate+bondRate+fedFundsRate+
                         Borrower.Credit.Score.at.Origination+Loan.Purpose+Property.Type+Original.UPB, data= trainNB,laplace=1)

modelNBL
predictionsL <- predict(modelNBL, newdata = testNB)

CrossTable(testNB$Occupancy.Status, predictionsL,
           prop.chisq = FALSE, prop.c = FALSE, 
           prop.r = FALSE, 
           dnn = c('actual Class', 'predicted Class'))


##################NB model with entire dataset####################

modelNB <- naiveBayes(Occupancy.Status~ ., data= train)

predictions <- predict(modelNB, newdata = test)
CrossTable(test$Occupancy.Status, predictions,
           prop.chisq = FALSE, prop.c = FALSE, 
           prop.r = FALSE, 
           dnn = c('actual Class', 'predicted Class'))

modelNBL <- naiveBayes(Occupancy.Status~ ., data= train, laplace = 1)

predictions <- predict(modelNBL, newdata = test)
CrossTable(test$Occupancy.Status, predictions,
           prop.chisq = FALSE, prop.c = FALSE, 
           prop.r = FALSE, 
           dnn = c('actual Class', 'predicted Class'))




########### DECISION TREES AND RANDOM FORESTS###############FIRST ATTEMPT###############

## SPLITTING THE DATASET INTO TRAININ AND TESTING SETS USING RANDOM NUMBERS
set.seed(12345)
data <- read.csv("New_FM2022Q4.csv")
#CREATE RANDOM NOS USING nrow() to create nrows of random nos
randomNos <- runif(nrow(data)) 
#USE randomNos to SHUFFLE DATA, 
datar <- data[order(randomNos), ]
dim(datar)
train <- datar[1:108098, ] #75% of the dataset
write.csv(train, file = "DT_Train_FM2022Q4.csv", row.names = FALSE)
trainDT <- read.csv("DT_Train_FM2022Q4.csv")
dim(trainDT)
test <- datar[108099:144130, ] #25% of the dataset
write.csv(test, file= "DT_Test_FM2022Q4.csv", row.names = FALSE)
testDT <- read.csv("DT_Test_FM2022Q4.csv")
dim(testDT)
#### showing that the distributions are the same before and after split
summary(data)
summary(trainDT)
summary(testDT)
prop.table(table(trainDT$Occupancy.Status))
prop.table(table(testDT$Occupancy.Status))


# SELECTING IVs and DV FOR THE DECISION TREE
library(dplyr)
trainDT <- subset(trainDT,
  select= c(Original.Interest.Rate,bondRate,fedFundsRate,Borrower.Credit.Score.at.Origination,Original.UPB,Occupancy.Status))
trainDT$Occupancy.Status <-as.factor(trainDT$Occupancy.Status)
dim(trainDT)


testDT <- subset(testDT,
  select=c(Original.Interest.Rate,bondRate,fedFundsRate,Borrower.Credit.Score.at.Origination,Original.UPB,Occupancy.Status))
testDT$Occupancy.Status <-as.factor(testDT$Occupancy.Status)
dim(testDT)

levels(trainDT$Occupancy.Status)
levels(testDT$Occupancy.Status)


##############TRAINING A DECISION TREE
library(C50)

modelDT <- C5.0(Occupancy.Status~ Original.Interest.Rate+bondRate+fedFundsRate+
                  Borrower.Credit.Score.at.Origination+Original.UPB, data= trainDT)

summary(modelDT)

#Plotting the decision tree with c5.0 yielded a cluttered tree
library(partykit)
library(C50)
plot(modelDT)
plot(modelDT, width =1, height = 6)
plot(modelDT, subtree = 2)
plot(modelDT, subtree = 1)
plot(modelDT, trial = 5, subtree = 5)
dev.off()

maxDepth <- 1
plot(modelDT, trial = 0,maxdepth = maxDepth, subtree = NULL)

##################### PREDICTIONS USING THE dECISION TREE MODEL

predictions <- predict(modelDT, newdata = testDT)

CrossTable(testDT$Occupancy.Status, predictions,
           prop.chisq = FALSE, prop.c = FALSE, 
           prop.r = FALSE, 
           dnn = c('actual Class', 'predicted Class'))


########### BOOSTING
names(trainDT)
boost10ModelDT <- C5.0(trainDT[-6], trainDT$Occupancy.Status, trial =10)
summary(boost10ModelDT)

boost20ModelDT <- C5.0(trainDT[-6], trainDT$Occupancy.Status, trial =20)
summary(boost20ModelDT)
# 

boost100ModelDT <- C5.0(trainDT[-6], trainDT$Occupancy.Status, trial =100)
summary(boost100ModelDT)

boost30ModelDT <- C5.0(trainDT[-6], trainDT$Occupancy.Status, trial =30)
summary(boost30ModelDT)

##########boosting and decision trees with another package############

library(rpart)
modelDT <- rpart(Occupancy.Status ~ Original.Interest.Rate + bondRate + fedFundsRate +
                   Borrower.Credit.Score.at.Origination + Original.UPB, data = trainDT)
modelDT

summary(modelDT)
########## plot(modelDT)
########## text(modelDT)

##### ADDING PROPERTY TYPE VARIABLE AND REMOVING OTHERS TO SEE IF MODEL GETS BETTER
trainDTN <- read.csv("DT_Train_FM2022Q4.csv")
dim(trainDTN)
testDTN <- read.csv("DT_Test_FM2022Q4.csv")
dim(testDTN)
trainDTN <- subset(trainDTN,
  select=c(Original.Interest.Rate, Property.Type,Occupancy.Status))
trainDTN$Occupancy.Status <-as.factor(trainDTN$Occupancy.Status)
trainDTN$Property.Type <-as.factor(trainDTN$Property.Type)
dim(trainDTN)



testDTN <- subset(testDTN,
  select=c(Original.Interest.Rate,Property.Type,Occupancy.Status))
testDTN$Occupancy.Status <-as.factor(testDTN$Occupancy.Status)
testDTN$Property.Type <-as.factor(testDTN$Property.Type)
dim(testDTN)

levels(trainDT$Occupancy.Status)
levels(testDT$Occupancy.Status)

########MODEL STILL NOT PREDICTING S CLASS
modelDTN <- C5.0(Occupancy.Status~ Original.Interest.Rate+ Property.Type, data= trainDTN)

summary(modelDTN)
############################################################

#PREDICTIONS WITHOUT BOOSTING
modelDT <- C5.0(Occupancy.Status~ Original.Interest.Rate+bondRate+fedFundsRate+
                  Borrower.Credit.Score.at.Origination+Original.UPB, data= trainDT)
predictions <- predict(modelDT, newdata = testDT)

CrossTable(testDT$Occupancy.Status, predictions,
           prop.chisq = FALSE, prop.c = FALSE, 
           prop.r = FALSE, 
           dnn = c('actual Class', 'predicted Class'))



#########PREDICTIONS USING BOOSTING
boost10ModelDT <- C5.0(trainDT[-6], trainDT$Occupancy.Status, trial =10)
summary(boost10ModelDT)
predictions10 <- predict(boost10ModelDT, newdata = testDT)

CrossTable(testDT$Occupancy.Status, predictions10,
           prop.chisq = FALSE, prop.c = FALSE, 
           prop.r = FALSE, 
           dnn = c('actual Class', 'predicted Class'))

boost20ModelDT <- C5.0(trainDT[-6], trainDT$Occupancy.Status, trial =20)
predictions20 <- predict(boost20ModelDT, newdata = testDT)

CrossTable(testDT$Occupancy.Status, predictions20,
           prop.chisq = FALSE, prop.c = FALSE, 
           prop.r = FALSE, 
           dnn = c('actual Class', 'predicted Class'))

####100 trials

boost100ModelDT <- C5.0(trainDT[-6], trainDT$Occupancy.Status, trial =100)

predictions100 <- predict(boost100ModelDT, newdata = testDT)

CrossTable(testDT$Occupancy.Status, predictions100,
           prop.chisq = FALSE, prop.c = FALSE, 
           prop.r = FALSE, 
           dnn = c('actual Class', 'predicted Class'))

# 50 trials

boost50ModelDT <- C5.0(trainDT[-6], trainDT$Occupancy.Status, trial =50)

predictions50 <- predict(boost50ModelDT, newdata = testDT)

CrossTable(testDT$Occupancy.Status, predictions50,
           prop.chisq = FALSE, prop.c = FALSE, 
           prop.r = FALSE, 
           dnn = c('actual Class', 'predicted Class'))


boost30ModelDT <- C5.0(trainDT[-6], trainDT$Occupancy.Status, trial =30)
summary(boost30ModelDT)

################DECISION TREES ATTEMPT TWO######################

#########SECOND ATTEMPT DECISION TREEE##############

## SPLITTING THE DATASET INTO TRAININ AND TESTING SETS USING RANDOM NUMBERS
set.seed(12345)
data <- read.csv("New_FM2022Q4.csv")
#CREATE RANDOM NOS USING nrow() to create nrows of random nos
randomNos <- runif(nrow(data)) 
#USE randomNos to SHUFFLE DATA, 
datar <- data[order(randomNos), ]
dim(datar)
train <- datar[1:108098, ] #75% of the dataset
write.csv(train, file = "DT_Train_FM2022Q4.csv", row.names = FALSE)
trainDT <- read.csv("DT_Train_FM2022Q4.csv")
dim(trainDT)
test <- datar[108099:144130, ] #25% of the dataset
write.csv(test, file= "DT_Test_FM2022Q4.csv", row.names = FALSE)
testDT <- read.csv("DT_Test_FM2022Q4.csv")
dim(testDT)
#### showing that the distributions are the same before and after split
summary(data)
summary(trainDT)
summary(testDT)
prop.table(table(trainDT$Occupancy.Status))
prop.table(table(testDT$Occupancy.Status))

# SELECTING IVs and DV FOR THE DECISION TREE
library(dplyr)
trainDT <- subset(trainDT,
                  select=c(Original.Interest.Rate,bondRate,fedFundsRate,Borrower.Credit.Score.at.Origination,Original.UPB, Loan.Purpose, Property.Type, Occupancy.Status))
trainDT$Occupancy.Status <-as.factor(trainDT$Occupancy.Status)
trainDT$Property.Type <- as.factor(trainDT$Property.Type)
trainDT$Loan.Purpose <- as.factor(trainDT$Loan.Purpose)
dim(trainDT)


testDT <- subset(testDT,
                 select=c(Original.Interest.Rate,bondRate,fedFundsRate,Borrower.Credit.Score.at.Origination,Original.UPB,Loan.Purpose, Property.Type, Occupancy.Status))
testDT$Occupancy.Status <-as.factor(testDT$Occupancy.Status)
testDT$Property.Type <- as.factor(testDT$Property.Type)
testDT$Loan.Purpose <- as.factor(testDT$Loan.Purpose)
dim(testDT)

levels(trainDT$Occupancy.Status)
levels(testDT$Occupancy.Status)

##############TRAINING A DECISION TREE

library(C50)

modelDT <- C5.0(Occupancy.Status~ Original.Interest.Rate+bondRate+fedFundsRate+ Loan.Purpose+Property.Type+
                  Borrower.Credit.Score.at.Origination+Original.UPB, data= trainDT)
summary(modelDT)

##################### PREDICTIONS USING THE dECISION TREE MODEL

predictions <- predict(modelDT, newdata = testDT)

CrossTable(testDT$Occupancy.Status, predictions,
           prop.chisq = FALSE, prop.c = FALSE, 
           prop.r = FALSE, 
           dnn = c('actual Class', 'predicted Class'))


########### BOOSTING#### second attempt###########
names(trainDT)
boost10ModelDT <- C5.0(trainDT[-8], trainDT$Occupancy.Status, trial =10)
predictions <- predict(boost10ModelDT, newdata = testDT)
CrossTable(testDT$Occupancy.Status, predictions,
           prop.chisq = FALSE, prop.c = FALSE, 
           prop.r = FALSE, 
           dnn = c('actual Class', 'predicted Class'))



boost20ModelDT <- C5.0(trainDT[-8], trainDT$Occupancy.Status, trial =20)
summary(boost20ModelDT)
predictions <- predict(boost20ModelDT, newdata = testDT)
CrossTable(testDT$Occupancy.Status, predictions,
           prop.chisq = FALSE, prop.c = FALSE, 
           prop.r = FALSE, 
           dnn = c('actual Class', 'predicted Class'))

# 
boost25ModelDT <- C5.0(trainDT[-8], trainDT$Occupancy.Status, trial =25)
summary(boost30ModelDT)
predictions <- predict(boost30ModelDT, newdata = testDT)
CrossTable(testDT$Occupancy.Status, predictions,
           prop.chisq = FALSE, prop.c = FALSE, 
           prop.r = FALSE, 
           dnn = c('actual Class', 'predicted Class'))

boost100ModelDT <- C5.0(trainDT[-8], trainDT$Occupancy.Status, trial =100)
summary(boost100ModelDT)
predictions <- predict(boost100ModelDT, newdata = testDT)
CrossTable(testDT$Occupancy.Status, predictions,
           prop.chisq = FALSE, prop.c = FALSE, 
           prop.r = FALSE, 
           dnn = c('actual Class', 'predicted Class'))








#########BAGGING AND RANDOM FORESTS###########

library(ISLR)
library(MASS)
library(dplyr)
library(randomForest)

data <- read.csv("New_FM2022Q4.csv")
dim(data)
library(dplyr)
data <-  subset(data,
  select=c(Original.Interest.Rate,bondRate,fedFundsRate,Borrower.Credit.Score.at.Origination,Original.UPB))

write.csv(data, file = "RF_FM2022Q4.csv", row.names = FALSE)
dataRF <- read.csv("RF_FM2022Q4.csv")
dim(dataRF)

#########SAMPLING FROM ABOVE DATA#########
set.seed(123)
sample_size <- 1000
sampledataRF <- dataRF[sample(nrow(dataRF), sample_size, replace = FALSE), ]
dim(sampledataRF)

# attach(Boston)
# View(Boston)
set.seed(1)
trainRowIndices <-sample(1:nrow(sampledataRF),nrow(sampledataRF)/2)
trainRF <- sampledataRF[trainRowIndices, ]
dim(trainRF)
testRF <- sampledataRF[-trainRowIndices,]
dim(testRF)


baggingModel <-
  randomForest(Original.Interest.Rate~.,data=trainRF,mtry=5,importance=TRUE)
baggingModel

### Predicting using the model
pred <- predict(baggingModel, newdata= testRF)
mean((pred- testRF$Original.Interest.Rate)^2)

plot(pred,testRF$Original.Interest.Rate)
abline(c(0,1),col=2)

ModelRF2 <-
  randomForest(Original.Interest.Rate~.,data=trainRF,mtry=2,importance=TRUE)
ModelRF2

pred2 <- predict(ModelRF2, newdata= testRF)
mean((pred2- testRF$Original.Interest.Rate)^2)

plot(pred2, testRF$Original.Interest.Rate)
abline(c(0,1),col=2)
varImpPlot(ModelRF2)

#######no of trees =100 ##########

baggingModel100 <-
  randomForest(Original.Interest.Rate~.,data=trainRF,mtry=5,importance=TRUE, ntree=100)
baggingModel100

pred100 <- predict(baggingModel100, newdata= testRF)
mean((pred100- testRF$Original.Interest.Rate)^2)

plot(pred100,testRF$Original.Interest.Rate)
abline(c(0,1),col=2)


ModelRF100 <-
  randomForest(Original.Interest.Rate~.,data=trainRF,mtry=2,importance=TRUE, ntree=100)
ModelRF100

predRF100 <- predict(ModelRF100, newdata= testRF)
mean((predRF100- testRF$Original.Interest.Rate)^2)

plot(predRF100, testRF$Original.Interest.Rate)
abline(c(0,1),col=2)
varImpPlot(ModelRF100)


#######no of trees =200 ##########

baggingModel200 <-
  randomForest(Original.Interest.Rate~.,data=trainRF,mtry=5,importance=TRUE, ntree=200)
baggingModel200

pred200 <- predict(baggingModel200, newdata= testRF)
mean((pred200- testRF$Original.Interest.Rate)^2)

plot(pred200,testRF$Original.Interest.Rate)
abline(c(0,1),col=2)


ModelRF200 <-
  randomForest(Original.Interest.Rate~.,data=trainRF,mtry=2,importance=TRUE, ntree=200)
ModelRF200

predRF200 <- predict(ModelRF200, newdata= testRF)
mean((predRF200- testRF$Original.Interest.Rate)^2)

plot(predRF200, testRF$Original.Interest.Rate)
abline(c(0,1),col=2)
varImpPlot(ModelRF200)

#######no of trees =300 ##########

baggingModel300 <-
  randomForest(Original.Interest.Rate~.,data=trainRF,mtry=5,importance=TRUE, ntree=300)
baggingModel300

pred300 <- predict(baggingModel300, newdata= testRF)
mean((pred300- testRF$Original.Interest.Rate)^2)

plot(pred300,testRF$Original.Interest.Rate)
abline(c(0,1),col=2)


ModelRF300 <-
  randomForest(Original.Interest.Rate~.,data=trainRF,mtry=2,importance=TRUE, ntree=300)
ModelRF300

predRF300 <- predict(ModelRF300, newdata= testRF)
mean((predRF300- testRF$Original.Interest.Rate)^2)

plot(predRF300, testRF$Original.Interest.Rate)
abline(c(0,1),col=2)
varImpPlot(ModelRF300)


#######no of trees =400 ##########

baggingModel400 <-
  randomForest(Original.Interest.Rate~.,data=trainRF,mtry=5,importance=TRUE, ntree=400)
baggingModel400

pred400 <- predict(baggingModel400, newdata= testRF)
mean((pred400- testRF$Original.Interest.Rate)^2)

plot(pred400,testRF$Original.Interest.Rate)
abline(c(0,1),col=2)


ModelRF400 <-
  randomForest(Original.Interest.Rate~.,data=trainRF,mtry=2,importance=TRUE, ntree=400)
ModelRF400

predRF400 <- predict(ModelRF400, newdata= testRF)
mean((predRF400- testRF$Original.Interest.Rate)^2)

plot(predRF400, testRF$Original.Interest.Rate)
abline(c(0,1),col=2)
varImpPlot(ModelRF400)


































































































