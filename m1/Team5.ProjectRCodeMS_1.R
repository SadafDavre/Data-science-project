setwd("C:/Users/bhoga/Desktop/Team5.Project.Check")

#Reading the downloaded and extracted Fannie Mae '2022Q4.csv' file
install.packages("readr")
library(readr)
library(dplyr)
library(ggplot2)
library(tidyverse)

fannieCSVfilePath <- "2022Q4.csv"
#Ccould not upload this file on github as it was too big, its available on the fannie mae site
#Inserting headers extracted from 'crt-layout-file' into the code
headers <- c("Reference Pool ID", "Loan Identifier", "Monthly Reporting Period", "Channel", "Seller Name", "Servicer Name", "Master Servicer", "Original Interest Rate", "Current Interest Rate", "Original UPB", "UPB at Issuance", "Current Actual UPB", "Original Loan Term", "Origination Date", "First Payment Date", "Loan Age", "Remaining Months to Legal Maturity", "Remaining Months To Maturity", "Maturity Date", "Original Loan to Value Ratio (LTV)", "Original Combined Loan to Value Ratio (CLTV)", "Number of Borrowers", "Debt-To-Income (DTI)", "Borrower Credit Score at Origination", "Co-Borrower Credit Score at Origination", "First Time Home Buyer Indicator", "Loan Purpose", "Property Type", "Number of Units", "Occupancy Status", "Property State", "Metropolitan Statistical Area (MSA)", "Zip Code Short", "Mortgage Insurance Percentage", "Amortization Type", "Prepayment Penalty Indicator", "Interest Only Loan Indicator", "Interest Only First Principal And Interest Payment Date", "Months to Amortization", "Current Loan Delinquency Status", "Loan Payment History", "Modification Flag", "Mortgage Insurance Cancellation Indicator", "Zero Balance Code", "Zero Balance Effective Date", "UPB at the Time of Removal", "Repurchase Date", "Scheduled Principal Current", "Total Principal Current", "Unscheduled Principal Current", "Last Paid Installment Date", "Foreclosure Date", "Disposition Date", "Foreclosure Costs", "Property Preservation and Repair Costs", "Asset Recovery Costs", "Miscellaneous Holding Expenses and Credits", "Associated Taxes for Holding Property", "Net Sales Proceeds", "Credit Enhancement Proceeds", "Repurchase Make Whole Proceeds", "Other Foreclosure Proceeds", "Modification-Related Non-Interest Bearing UPB", "Principal Forgiveness Amount", "Original List Start Date", "Original List Price", "Current List Start Date", "Current List Price", "Borrower Credit Score At Issuance", "Co-Borrower Credit Score At Issuance", "Borrower Credit Score Current ", "Co-Borrower Credit Score Current", "Mortgage Insurance Type", "Servicing Activity Indicator", "Current Period Modification Loss Amount", "Cumulative Modification Loss Amount", "Current Period Credit Event Net Gain or Loss", "Cumulative Credit Event Net Gain or Loss", "Special Eligibility Program", "Foreclosure Principal Write-off Amount", "Relocation Mortgage Indicator", "Zero Balance Code Change Date", "Loan Holdback Indicator", "Loan Holdback Effective Date", "Delinquent Accrued Interest", "Property Valuation Method ", "High Balance Loan Indicator ", "ARM Initial Fixed-Rate Period  ≤ 5 YR Indicator", "ARM Product Type", "Initial Fixed-Rate Period ", "Interest Rate Adjustment Frequency", "Next Interest Rate Adjustment Date", "Next Payment Change Date", "Index", "ARM Cap Structure", "Initial Interest Rate Cap Up Percent", "Periodic Interest Rate Cap Up Percent", "Lifetime Interest Rate Cap Up Percent", "Mortgage Margin", "ARM Balloon Indicator", "ARM Plan Number", "Borrower Assistance Plan", "High Loan to Value (HLTV) Refinance Option Indicator", "Deal Name", "Repurchase Make Whole Proceeds Flag", "Alternative Delinquency Resolution", "Alternative Delinquency  Resolution Count", "Total Deferral Amount ")
dataAPfile <-  read_delim(fannieCSVfilePath, delim = "|", col_names = FALSE)  
colnames(dataAPfile) <- headers

#Writing the new csv file with headers

write.csv(dataAPfile, file = fannieCSVwithHeadersPath, row.names = FALSE)

#Reading the created file to find number of observations(Rows) and variables(Columns)
data <- read.csv("FM_AP_R_2022Q4.csv")
View(data)
str(data)
dim(data)
head(data)
glimpse(data) #Gave us all 108 variables' datatypes

#Subset required columns and writing to a new csv file
library(dplyr)
allColsFile <- "FM_AP_R_2022Q4.csv"
reqColsFile <- "Trimmed_FM_AP_R_2022Q4.csv"
data <- read.csv(allColsFile)
View(data)
dataWithReqCols <- data %>%
  select("Loan.Identifier","Monthly.Reporting.Period","Channel", "Seller.Name", "Original.Interest.Rate", "Original.UPB","Original.Loan.Term","Origination.Date","Loan.Age","Remaining.Months.to.Legal.Maturity","Remaining.Months.To.Maturity","Maturity.Date","Original.Loan.to.Value.Ratio..LTV.", "Original.Combined.Loan.to.Value.Ratio..CLTV.","Number.of.Borrowers","Debt.To.Income..DTI.","Borrower.Credit.Score.at.Origination","Co.Borrower.Credit.Score.at.Origination","First.Time.Home.Buyer.Indicator","Loan.Purpose","Property.Type","Number.of.Units","Occupancy.Status","Property.State","Metropolitan.Statistical.Area..MSA.", "Zip.Code.Short","Amortization.Type","Prepayment.Penalty.Indicator","Current.Loan.Delinquency.Status")
write.csv(dataWithReqCols, file = reqColsFile, row.names = FALSE)

data <- read.csv("Trimmed_FM_AP_R_2022Q4.csv")
View(data)
dim(data)
str(data)
glimpse(data)

#Filtering rows and creating a new csv file called uniqueLoans_FM_AP_R_2022Q4.csv

reportingPeriod <-32023
filteredData <- data %>%
  filter(Monthly.Reporting.Period == reportingPeriod)
uniqueLoansFilePath <- "UniqueLoans_FM_AP_R_2022Q4.csv"
write.csv(filteredData, file = uniqueLoansFilePath, row.names = FALSE)

data <- read.csv("UniqueLoans_FM_AP_R_2022Q4.csv")
View(data)
dim(data)
glimpse(data)


# Datatype conversions :monthly reporting date and other date column conversion
library(zoo)
library(lubridate)
#Defining a Custom function
intToDateFunc <- function(intDate) {
  date <- as.Date(paste0(ifelse(intDate %/% 10000 < 10, paste("0", intDate %/% 10000, sep = ""), intDate %/% 10000),"-", intDate %% 10000,"-01"), format = "%m-%Y-%d")
}
#Reading the dataframe and identifying columns for conversion
data <- read.csv("UniqueLoans_FM_AP_R_2022Q4.csv")
View(data)
columnsToConvert <- c("Monthly.Reporting.Period","Origination.Date","Maturity.Date")
#Calling the function and changing columnsToConvert
data <- data %>%
  mutate(across(all_of(columnsToConvert), ~intToDateFunc(.)))

#Writing the rectified date formats file to final csv file 
rectifiedDateFormatFile <- "Final_FM_AP_R_2022Q4.csv"
write.csv(data, file = rectifiedDateFormatFile, row.names = FALSE)
data <- read.csv("Final_FM_AP_R_2022Q4.csv")
View(data)
dim(data)
glimpse(data)

#Checking for Null Values
summary(data)

nullValues <- colSums(is.na(data))
if (any(nullValues >0)) {
  print(nullValues)
} else {
  print("No Null values")
}
# Removing 364 Null values from Credit score field
data <- data[complete.cases(data$Borrower.Credit.Score.at.Origination),]
dim(data)
# Running is.na() function on dataset to verify if dataset has changed
nullValues <- colSums(is.na(data))
if (any(nullValues >0)) {
  print(nullValues)
} else {
  print("No Null values exist in the dataframe")
}
data <- data[complete.cases(data$Debt.To.Income..DTI.),]
dim(data)

#Checking for Null values
nullValues <- colSums(is.na(data))
if (any(nullValues >0)) {
  print(nullValues)
} else {
  print("No Null values")
}
transformedFile <- "Transformed_FM_AP_R_2022Q4.csv"
write.csv(data, file = transformedFile, row.names = FALSE)
data <- read.csv("Transformed_FM_AP_R_2022Q4.csv")
dim(data)


# Writing final variables to a file for ease of use
finalVariablesFile <- "FinalVariables_FM_AP_R_2022Q4.csv"

finalVariablesdata <- data %>%
  select("Original.Interest.Rate","Original.Loan.to.Value.Ratio..LTV.","Original.UPB","Number.of.Borrowers", "Original.Combined.Loan.to.Value.Ratio..CLTV.", "Debt.To.Income..DTI.", "Loan.Purpose", "Property.Type", "Number.of.Units", "Occupancy.Status", "Seller.Name", "Property.State", "Borrower.Credit.Score.at.Origination")
write.csv(finalVariablesdata,file= finalVariablesFile, row.names = FALSE)

data <- read.csv("FinalVariables_FM_AP_R_2022Q4.csv")
dim(data)
View(data)

#Outlier detection and Null value using Summary
summary(data)

#Distribution of interest rate variable and outlier existence
hist(data$Original.Interest.Rate,
     xlab = "Original Interest Rate",
     main = "Histogram of Original Interest Rate")

hist(data$Original.UPB,
     xlab = "Original UPB",
     main = "Histogram of Original UPB")

hist(data$Original.Loan.to.Value.Ratio..LTV.,
     xlab = "Original LTV",
     main = "Histogram of Original LTV")

hist(data$Original.Combined.Loan.to.Value.Ratio..CLTV.,
     xlab = "Original CLTV",
     main = "Histogram of Original CLTV")

hist(data$Debt.To.Income..DTI.,
     xlab = "Original DTI",
     main = "Histogram of Original DTI")

hist(data$Borrower.Credit.Score.at.Origination,
     xlab = "Borrower Credit Score",
     main = "Histogram of Borrower Credit Score")

# Creating pie charts of the categorical variables
LPcategories <- table(data$Loan.Purpose)
customLabels <- c("Cashout","Purchase","Refinance")
pie(LPcategories, labels = customLabels, main = "Pie Chart of Loan Purpose")

PTcategories <- table(data$Property.Type)
pie(PTcategories, labels = names(PTcategories), main = "Distribution of Property Type")

OScategories <- table(data$Occupancy.Status)
customLabels <- c("Investor","Principal Residence","Secondary Home")
pie(OScategories, labels = customLabels, main = "Distribution of Occupancy Status")


#Bar Plots of Nominal variables
ggplot(data, aes(x = Property.State)) +
  geom_bar() +
  labs(title = "State-wise Loans") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggplot(data, aes(x = Seller.Name)) +
  geom_bar() +
  labs(title = "Seller-wise Loans") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
