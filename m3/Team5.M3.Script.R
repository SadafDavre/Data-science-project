setwd("C:/Users/asus/Desktop/Desktop/UMD FALL'23/INST737/Milestone-3/M3.Data.Files/")
install.packages("kernlab")
install.packages("mlr")
install.packages("data.table")
install.packages("mltools")
install.packages("e1071")
install.packages("neuralnet")
install.packages("factoextra")
install.packages("NbClust")
install.packages("cluster")
install.packages("fpc")
install.packages("factoextra")
install.packages("NbClust")
install.packages("dbscan")

library(caret)
library(kernlab)
library(dplyr)
library(mlr)
library(data.table)
library(mltools)
library(factoextra)
library(NbClust)
library(cluster)
library(fpc)
library(dbscan)
library(caret)
library(mlbench)
library(neuralnet)
library(MASS)
library(ggplot2)
library(corrplot)
library(caret)
library(leaps)
library(randomForest)
library(mlbench)


### Base file loading to derive MS3 Analysis file###
data <- read.csv("New_FM2022Q4.csv")
data <- data[sample(nrow(data)), ]
set.seed(123)
# Splitting the data into three subsets based on the outcome classes
data_P <- data[data$Occupancy.Status == 'P', ]
data_I <- data[data$Occupancy.Status == 'I', ]
data_S <- data[data$Occupancy.Status == 'S', ]
# Sampling equal number of observations from each subset
samples_per_class <- 920
sampled_data <- rbind(
  data_P[1:samples_per_class, ],
  data_I[1:samples_per_class, ],
  data_S[1:samples_per_class, ]
)
dim(sampled_data)
View(sampled_data)
sampled_data <- sampled_data[sample(nrow(sampled_data)), ]
View(sampled_data)
prop.table(table(sampled_data$Occupancy.Status))
glimpse(data)
### subsetting non-multicollinear variables#########
sampled_data <- subset(sampled_data,
                       select = c(Original.Interest.Rate,
                                  bondRate,
                                  fedFundsRate,
                                  Borrower.Credit.Score.at.Origination,
                                  Original.UPB,
                                  Number.of.Borrowers,
                                  Original.Loan.to.Value.Ratio..LTV.,
                                  Original.Combined.Loan.to.Value.Ratio..CLTV.,
                                  Debt.To.Income..DTI., 
                                  Seller.Name,
                                  Loan.Purpose,
                                  Property.Type,
                                  Property.State,
                                  Occupancy.Status))
write.csv(sampled_data, file = "MS3sample.csv", row.names = FALSE)
data <- read.csv("MS3sample.csv")
dim(data)
prop.table(table(data$Occupancy.Status))

##### creating an encoded dataset (except outcome variable)
data$Seller.Name <- as.factor(data$Seller.Name)
levels(data$Seller.Name)
data$Property.State <- as.factor(data$Property.State)

data$Property.Type <- as.factor(data$Property.Type)
data$Loan.Purpose <- as.factor(data$Loan.Purpose)
data$Occupancy.Status <- data$Occupancy.Status

##"CitiMortgage, Inc." is the reference level
dim(data)

dmy <-dummyVars(~Seller.Name, data = data, fullRank = T)
encoded_data <- data.frame(predict(dmy, newdata= data))
dim(encoded_data)

dmy1 <-dummyVars(~Property.Type, data = data, fullRank = T)
encoded_data1 <- data.frame(predict(dmy1, newdata= data))
dim(encoded_data1)

dmy2 <-dummyVars(~Property.State, data = data, fullRank = T)
encoded_data2 <- data.frame(predict(dmy2, newdata= data))
dim(encoded_data2)

dmy3 <-dummyVars(~Loan.Purpose, data = data, fullRank = T)
encoded_data3 <- data.frame(predict(dmy3, newdata= data))
dim(encoded_data3)



encoded_MS3dataset <- cbind(data, encoded_data,
                            encoded_data1,
                            encoded_data2,
                            encoded_data3)



dim(encoded_MS3dataset)
glimpse(encoded_MS3dataset)

write.csv(encoded_MS3dataset, file = "encoded_MS3sample.csv", row.names = FALSE)
data <- read.csv("encoded_MS3sample.csv")
dim(data)
glimpse(data)
## REMOVING ORIGINAL COLUMNS AS WE HAVE ENCODED COLUMNS
data <- data[, -which(names(data)== 'Seller.Name')]
dim(data)

data <- data[, -which(names(data)== 'Property.Type')]
dim(data)

data <- data[, -which(names(data)== 'Property.State')]
dim(data)

data <- data[, -which(names(data)== 'Loan.Purpose')]
dim(data)
write.csv(data, file = "final_MS3dataset.csv", row.names = FALSE)

#### CREATING TRAINING AND TEST DATASETS FOR SVM MODELS
data <- read.csv("final_MS3dataset.csv")
dim(data)

data$Occupancy.Status <- as.factor(data$Occupancy.Status)
str(data)

#########SPLITTING DATA

data <- data[sample(nrow(data)), ]
set.seed(123)
train <- data[1:2070, ]
dim(train)
test <- data[2071:2760, ]
dim(test)

#### TRAINING LINEAR SVM CLASSIFIER USING VANILLADOT KERNEL
library(kernlab)
svm_classifier <- ksvm(Occupancy.Status~.,
                       data = train, 
                       kernel = "vanilladot")
svm_classifier
#PREDICTING USING THE CLASSIFIER
svm_predictions <- predict(svm_classifier,test)
head(svm_predictions)
svm_predictions

####confusion matrix
table(svm_predictions, test$Occupancy.Status)
#####count 
agreement <- svm_predictions == test$Occupancy.Status
table(agreement)
prop.table(table(agreement))

confusion_matrix <- confusionMatrix(svm_predictions, test$Occupancy.Status)
confusion_matrix


## CALCULATING F1 MEASURE
#F1 <- 2*(Precision * Recall)/(Precision + Recall)
#Precision
p.i <- 0.5726
p.p <- 0.6933
p.s <-0.5403
#Recall
r.i <- 0.5923
r.p <- 0.7237
r.s <- 0.4978

f1_i <- 2*(p.i * r.i)/(p.i + r.i)
f1_i
f1_p <- 2*(p.p * r.p)/(p.p + r.p)
f1_p
f1_s <- 2*(p.s * r.s)/(p.s + r.s)
f1_s

###USING RBF KERNEL

svm_classifier_rbf <-ksvm(Occupancy.Status~., data = train, kernel = "rbfdot")
svm_classifier_rbf
##PREDICTING WITH RBF CLASSIFIER
svm_predictions_rbf <- predict(svm_classifier_rbf, test)
svm_predictions_rbf

agreement_rbf <-svm_predictions_rbf == test$Occupancy.Status
table(agreement_rbf)#Number of TRUE is more
prop.table(table(agreement_rbf))#TRUE percentage increased


confusion_matrix_rbf <- confusionMatrix(svm_predictions_rbf, test$Occupancy.Status)
confusion_matrix_rbf


######################################################################

#### NEURAL NETWORKS TO TRAIN FOR REGRESSION QUESTION#########
data <- read.csv("final_MS3dataset.csv")
glimpse(data)
#Above encoded dataset does not have encoding for Occupancy Status
dmy4 <-dummyVars(~Occupancy.Status, data = data, fullRank = T) #I is taken as reference class
encoded_data4 <- data.frame(predict(dmy4, newdata= data))
dim(encoded_data4)
View(encoded_data4)#Has only 2 classes encoded, one is reference class
encoded_MS3dataset_sn <- cbind(data, encoded_data4)
dim(encoded_MS3dataset_sn)
encoded_MS3dataset_sn <- encoded_MS3dataset_sn[, -which(names(encoded_MS3dataset_sn)== 'Occupancy.Status')]
dim(encoded_MS3dataset_sn)

write.csv(encoded_MS3dataset_sn, file = "all_cat_encoded_MS3dataset.csv", row.names = FALSE)
data <- read.csv("all_cat_encoded_MS3dataset.csv")

dim(data)

normalize <- function(x) { 
  return((x - min(x)) / (max(x) - min(x)))
}

###applying the function to the numeric variables

# Identify the numeric columns in your dataset
numeric_cols <- sapply(data, is.numeric)

# Normalize only the numeric columns
data[, numeric_cols] <- lapply(data[, numeric_cols], normalize)

View(data)
write.csv(data, file = "Normalized_MS3dataset.csv", row.names = FALSE)
data <- read.csv("Normalized_MS3dataset.csv")
dim(data)

##SPLITTING DATA INTO TRAINING AND TEST DATASETS

train_NN <- data[1:2070, ]
test_NN <- data[2071:2760, ]
str(train_NN)
library(neuralnet)
########## USING THE SAME SET OF VARIABLES AS THE BEST MULTIVARIATE MODEL

model_NN_4var_SN <- neuralnet(formula = Original.Interest.Rate ~ bondRate + fedFundsRate +
                                Borrower.Credit.Score.at.Origination +
                                Original.UPB +
                                Seller.Name.CrossCountry.Mortgage..LLC +
                                Seller.Name.DHI.Mortgage.Company..Ltd. + 
                                Seller.Name.Fairway.Independent.Mortgage.Corporation +
                                Seller.Name.Fifth.Third.Bank..National.Association + 
                                Seller.Name.Guaranteed.Rate..Inc.+
                                Seller.Name.Guild.Mortgage.Company.LLC +
                                Seller.Name.JPMorgan.Chase.Bank..National.Association +
                                Seller.Name.Lakeview.Loan.Servicing..LLC+
                                Seller.Name.Lennar.Mortgage..LLC +
                                Seller.Name.loanDepot.com..LLC +
                                Seller.Name.Movement.Mortgage..LLC +
                                Seller.Name.NationStar.Mortgage..LLC +
                                Seller.Name.NewRez.LLC +
                                Seller.Name.NexBank +
                                Seller.Name.Other +
                                Seller.Name.PennyMac.Corp.+
                                Seller.Name.PennyMac.Loan.Services..LLC +
                                Seller.Name.PHH.Mortgage.Corporation +
                                Seller.Name.Planet.Home.Lending..LLC +
                                Seller.Name.Rocket.Mortgage..LLC +
                                Seller.Name.Truist.Bank..formerly.SunTrust.Bank.+
                                Seller.Name.U.S..Bank.N.A.+
                                Seller.Name.United.Wholesale.Mortgage..LLC +
                                Seller.Name.Wells.Fargo.Bank..N.A.,
                              data = train_NN)

plot(model_NN_4var_SN)
View(test_NN)
model_NN_4var_SN_results <-compute(model_NN_4var_SN,test_NN[, c(2:5,10:33)])
predicted_IR_4var_SN <- model_NN_4var_SN_results$net.result
cor(predicted_IR_4var_SN, test_NN$Original.Interest.Rate)
mse <- mean((predicted_IR_4var_SN - test_NN$Original.Interest.Rate)^2)
mse
rmse <- sqrt(mse)
rmse


##########using ALL THE NUMERIC variables IN THE DATASET# ##########
glimpse(train_NN)
model_NN_numvar <- neuralnet(formula = Original.Interest.Rate ~ bondRate+fedFundsRate+
                               Borrower.Credit.Score.at.Origination +
                               Original.UPB+
                               Number.of.Borrowers+
                               Original.Loan.to.Value.Ratio..LTV.+
                               Original.Combined.Loan.to.Value.Ratio..CLTV.+
                               Debt.To.Income..DTI.,
                             data = train_NN)
plot(model_NN_numvar)
glimpse(test_NN)
View(test_NN)
model_NN_numvar_results <-compute(model_NN_numvar,test_NN[2:9])
model_NN_numvar_results
model_NN_numvar_results$net.result #Gives predicted values
predicted_IR_numvar <- model_NN_numvar_results$net.result

cor(predicted_IR_numvar, test_NN$Original.Interest.Rate)
mse <- mean((predicted_IR_numvar - test_NN$Original.Interest.Rate)^2)
mse
rmse <- sqrt(mse)
rmse

##### USING SELLER NAME VARIABLE along with numeric variable from best multivariate model#######
glimpse(train_NN)
model_NN_SN <- neuralnet(formula = Original.Interest.Rate ~ bondRate + fedFundsRate +
                           Borrower.Credit.Score.at.Origination +
                           Original.UPB +
                           Number.of.Borrowers +
                           Original.Loan.to.Value.Ratio..LTV. +
                           Original.Combined.Loan.to.Value.Ratio..CLTV. +
                           Debt.To.Income..DTI. +
                           Seller.Name.CrossCountry.Mortgage..LLC +
                           Seller.Name.DHI.Mortgage.Company..Ltd. + 
                           Seller.Name.Fairway.Independent.Mortgage.Corporation +
                           Seller.Name.Fifth.Third.Bank..National.Association + 
                           Seller.Name.Guaranteed.Rate..Inc.+
                           Seller.Name.Guild.Mortgage.Company.LLC +
                           Seller.Name.JPMorgan.Chase.Bank..National.Association +
                           Seller.Name.Lakeview.Loan.Servicing..LLC+
                           Seller.Name.Lennar.Mortgage..LLC +
                           Seller.Name.loanDepot.com..LLC +
                           Seller.Name.Movement.Mortgage..LLC +
                           Seller.Name.NationStar.Mortgage..LLC +
                           Seller.Name.NewRez.LLC +
                           Seller.Name.NexBank +
                           Seller.Name.Other +
                           Seller.Name.PennyMac.Corp.+
                           Seller.Name.PennyMac.Loan.Services..LLC +
                           Seller.Name.PHH.Mortgage.Corporation +
                           Seller.Name.Planet.Home.Lending..LLC +
                           Seller.Name.Rocket.Mortgage..LLC +
                           Seller.Name.Truist.Bank..formerly.SunTrust.Bank.+
                           Seller.Name.U.S..Bank.N.A.+
                           Seller.Name.United.Wholesale.Mortgage..LLC +
                           Seller.Name.Wells.Fargo.Bank..N.A.,
                         data = train_NN)

plot(model_NN_SN)

model_NN_SN_results <-compute(model_NN_SN,test_NN[2:33])
predicted_IR_SN <- model_NN_SN_results$net.result
cor(predicted_IR_SN, test_NN$Original.Interest.Rate)
mse <- mean((predicted_IR_SN - test_NN$Original.Interest.Rate)^2)
mse
rmse <- sqrt(mse)
rmse



## model nn all variables ########## BEST MODEL####
library(neuralnet)
model_NN <- neuralnet(formula = Original.Interest.Rate~.,data = train_NN)
model_NN
model_NN$act.fct
model_NN$err.fct
# THE MODEL OUTPUT WAS LARGE AND COULD NOT BE SEEN IN THE CONSOLE, capturing to afile
sink("model_NN_output.txt")
model_NN
sink()

# Viewing the content of the file
cat(readLines("model_NN_output.txt"), sep = "\n")


plot(model_NN)
model_NN_results <-compute(model_NN,test_NN[2:93])
model_NN_results
model_NN_results$net.result #Gives predicted values
predicted_IR <- model_NN_results$net.result

cor(predicted_IR, test_NN$Original.Interest.Rate)
mse <- mean((predicted_IR - test_NN$Original.Interest.Rate)^2)
mse
rmse <- sqrt(mse)
rmse
summary(model_NN)

# Calculation of residuals (errors) and sse
residuals <- predicted_IR - test_NN$Original.Interest.Rate
sse <- sum(residuals^2)
sse


###### USING HIDDEN LAYERS=2 specified by c(3,3)###############
model_NN_H2 <- neuralnet(formula = Original.Interest.Rate~.,data = train_NN, hidden=c(3,3))
model_NN_H2_results <- compute(model_NN_H2, test_NN[2:93])
predicted_IR_H2 <- model_NN_H2_results$net.result
cor(predicted_IR_H2, test_NN$Original.Interest.Rate)
mse <- mean((predicted_IR_H2 - test_NN$Original.Interest.Rate)^2)
mse
rmse <- sqrt(mse)
rmse

plot(model_NN_H2)
# Calculation of residuals (errors) and sse
residuals <- predicted_IR_H2 - test_NN$Original.Interest.Rate
sse <- sum(residuals^2)
sse


###### USING HIDDEN LAYERS=3 specified by c(1,2,3)###############
model_NN_H3 <- neuralnet(formula = Original.Interest.Rate~.,data = train_NN, hidden=c(3,2,1))
model_NN_H3_results <- compute(model_NN_H3, test_NN[2:93])
predicted_IR_H3 <- model_NN_H3_results$net.result
cor(predicted_IR_H3, test_NN$Original.Interest.Rate)
mse <- mean((predicted_IR_H3 - test_NN$Original.Interest.Rate)^2)
mse
rmse <- sqrt(mse)
rmse

plot(model_NN_H3)
residuals <- predicted_IR_H3 - test_NN$Original.Interest.Rate
sse <- sum(residuals^2)
sse

###### USING HIDDEN LAYERS=3 specified by c(2,1,1)###############
model_NN_H3 <- neuralnet(formula = Original.Interest.Rate~.,data = train_NN, hidden=c(2,1,1))
model_NN_H3_results <- compute(model_NN_H3, test_NN[2:93])
predicted_IR_H3 <- model_NN_H3_results$net.result
cor(predicted_IR_H3, test_NN$Original.Interest.Rate)
mse <- mean((predicted_IR_H3 - test_NN$Original.Interest.Rate)^2)
mse
rmse <- sqrt(mse)
rmse
plot(model_NN_H3)
residuals <- predicted_IR_H3 - test_NN$Original.Interest.Rate
sse <- sum(residuals^2)
sse


###### USING HIDDEN LAYERS=3 specified by c(1,1,1)###############
model_NN_H3 <- neuralnet(formula = Original.Interest.Rate~.,data = train_NN, hidden=c(1,1,1))
model_NN_H3_results <- compute(model_NN_H3, test_NN[2:93])
predicted_IR_H3 <- model_NN_H3_results$net.result
cor(predicted_IR_H3, test_NN$Original.Interest.Rate)
mse <- mean((predicted_IR_H3 - test_NN$Original.Interest.Rate)^2)
mse
rmse <- sqrt(mse)
rmse
plot(model_NN_H3)
residuals <- predicted_IR_H3 - test_NN$Original.Interest.Rate
sse <- sum(residuals^2)
sse

###### USING HIDDEN LAYERS=4 specified by c(1,1,1,1)###############
model_NN_H4 <- neuralnet(formula = Original.Interest.Rate~.,data = train_NN, hidden=c(1,1,1,1))
model_NN_H4_results <- compute(model_NN_H4, test_NN[2:93])
predicted_IR_H4 <- model_NN_H4_results$net.result
cor(predicted_IR_H4, test_NN$Original.Interest.Rate)
mse <- mean((predicted_IR_H4 - test_NN$Original.Interest.Rate)^2)
mse
rmse <- sqrt(mse)
rmse
residuals <- predicted_IR_H4 - test_NN$Original.Interest.Rate
sse <- sum(residuals^2)
sse
plot(model_NN_H4)

###### USING HIDDEN LAYERS=5 specified by c(1,1,1,1,1)###############
model_NN_H5 <- neuralnet(formula = Original.Interest.Rate~.,data = train_NN, hidden=c(1,1,1,1,1))
model_NN_H5_results <- compute(model_NN_H5, test_NN[2:93])
predicted_IR_H5 <- model_NN_H5_results$net.result
cor(predicted_IR_H5, test_NN$Original.Interest.Rate)
mse <- mean((predicted_IR_H5 - test_NN$Original.Interest.Rate)^2)
mse
rmse <- sqrt(mse)
rmse
residuals <- predicted_IR_H5 - test_NN$Original.Interest.Rate
sse <- sum(residuals^2)
sse
plot(model_NN_H5)




########## CLUSTERING##################
library(cluster)
library(fpc)
library(factoextra)
library(NbClust)

data <- read.csv("MS3sample.csv")
dim(data)
glimpse(data)
data$Property.Type <- as.factor(data$Property.Type)
data$Property.State <- as.factor(data$Property.State)
data$Occupancy.Status <- as.factor(data$Occupancy.Status)
data$Loan.Purpose <- as.factor(data$Loan.Purpose)
data$Seller.Name <- as.factor(data$Seller.Name)

# calculating GOWER DISTANCE
g.dist = daisy(data, metric="gower", type=list())

####PAM CLUSTERING###########
#  Using Gower distance matrix, determining optimal number of clusters
pc <- pamk(g.dist, krange = 1:5, criterion = "asw")
pc

# Extracting the optimal number of clusters
optimal_clusters <- pc$nc
optimal_clusters

dim(data)
# Performing PAM clustering with the optimal number of clusters
pam_result <- pam(g.dist, k = optimal_clusters)
pam_result
pam_result$clustering
pc$pamobject$clustering
#number of elements per cluster
cluster_counts <- table(pc$pamobject$clustering)
print(cluster_counts)


#number of medoids
medoids <- pam_result$medoids
medoids

### Plotting Optimal number of clusters#####
# Computing silhouette scores
sil_scores <- silhouette(pam_result$clustering, dist = g.dist)

# Visualize silhouette scores
fviz_silhouette(sil_scores)


##### COMPARING CLUSTERS, CALCULATING CLUSTER MEANS
# Accessing cluster assignments
cluster_assignments <- pam_result$clustering
cluster_assignments

# Combining original data and cluster assignments
clustered_data <- cbind(data, Cluster = cluster_assignments)

# Calculating mean values for each cluster
cluster_means <- aggregate(. ~ Cluster, data = clustered_data, FUN = mean)
print(cluster_means)


# Creating box plots for each feature, one for each cluster for comparison
par(mfrow = c(4, 4))  
for (feature in colnames(data)) {
  boxplot(data[, feature] ~ clustered_data$Cluster, main = feature, col = c("red", "blue"))
}

par(mfrow = c(1, 1))


####### HIEARARCHICAL CLUSTERING###########

hc.m = hclust(g.dist, method="median")
hc.m
hc.s = hclust(g.dist, method="single")
hc.s
hc.c = hclust(g.dist, method="complete")
hc.c
### VISUALIZING DENDROGRAMS
layout(matrix(1:3, nrow=1))
plot(hc.c,cex = 0.6, hang = -1)
plot(hc.s,cex = 0.6, hang = -1)
plot(hc.m, cex = 0.6, hang = -1)


# cutting the tree by inspection to obtain optimal number of clusters

hclusters_complete <- cutree(hc.c, h = 0.75)
h.clusters_elements_complete <- table(hclusters_complete)
print(h.clusters_elements_complete)

hclusters_single <- cutree(hc.s, h = 0.28)
h.clusters_elements_single <- table(hclusters_single)
print(h.clusters_elements_single)
###Error saying height component of tree not sorted increasingly
hclusters_median <- cutree(hc.m, h = 0.34)
h.clusters_elements_median <- table(hclusters_median)
print(h.clusters_elements_median)


# Combining original data and clusters
clustered_data <- cbind(data, Cluster = hclusters)

# Calculating mean values for each cluster
h.cluster_means <- aggregate(. ~ Cluster, data = clustered_data, FUN = mean)

# Print the cluster means
print(h.cluster_means)


###### DBSCAN clustering ########
#DBSCAN
####### gives two clusters but high noise values
dbc2 = dbscan(g.dist, eps=.129, minPts=10) 
dbc2
##########

dbc2 = dbscan(g.dist, eps=.15, minPts= 5);  #5 clusters
dbc2

dbc2 = dbscan(g.dist, eps=.15, minPts= 8);  #3 clusters
dbc2

dbc2 = dbscan(g.dist, eps=.15, minPts= 9);  #2 clusters
dbc2

dbc2 = dbscan(g.dist, eps=.2, minPts= 9);  #1 clusters
dbc2

dbc2 = dbscan(g.dist, eps=.155, minPts= 9);  #2 clusters
dbc2

dbc2 = dbscan(g.dist, eps=.156, minPts= 9);  #2 clusters
dbc2

dbc2 = dbscan(g.dist, eps=.157, minPts= 9);  #2 clusters
dbc2

dbc2 = dbscan(g.dist, eps=.149, minPts= 9);  #1 clusters
dbc2

dbc2 = dbscan(g.dist, eps=.119, minPts= 9);  #1 clusters
dbc2
dbc2 = dbscan(g.dist, eps=.139, minPts= 9);  #1 clusters
dbc2

#### THis gives 2 clusters but many outliers not assigned to any group
dbc2 = dbscan(g.dist, eps=.149, minPts= 50);  #2 clusters
dbc2

dbc2 = dbscan(g.dist, eps=.14, minPts= 5);  #2 clusters
dbc2



#### efforts 24/11/23####
dbc2 = dbscan(g.dist, eps=.12959, minPts=9) 
dbc2
dbc2 = dbscan(g.dist, eps=.135, minPts=9) 
dbc2

dbc2 = dbscan(g.dist, eps=.119, minPts=10) 
dbc2

dbc2 = dbscan(g.dist, eps=.13, minPts= 5) 
dbc2

dbc2 = dbscan(g.dist, eps=.18, minPts= 7) ###2 FINAL
dbc2

dbc2$cluster
plot(data, col = dbc2$cluster, pch = 16, main = "DBSCAN Clustering")

##COMPUTING MEANS FOR FEATURES IN THE CLUSTER
dbscan_clustered_data <- data.frame(data, Cluster = dbc2$cluster)

str(dbscan_clustered_data)

# CalculatING mean values for each cluster
dbscan_cluster_means <- aggregate(. ~ Cluster, data = dbscan_clustered_data, FUN = mean)

# Display the cluster means
print(dbscan_cluster_means)

# Converting gower distance matrix to numeric to visualize heatmap and check clusters
library(gplots)
g.dist.numeric <- as.matrix(g.dist)
cluster_colors <- as.character(dbc2$cluster)
heatmap(g.dist.numeric, main = "Gower Distances Heatmap", 
        xlab = "Data Points", ylab = "Data Points",
        ColSideColors = cluster_colors)


# comparing the clustering results
table(hclusters_single, hclusters_complete)
table(hclusters_complete, pam_result$clustering)
table(hclusters_single, dbc2$cluster)
table(pam_result$clustering, hclusters_single)
table(dbc2$cluster, pam_result$clustering)

#########clustering  using few continuous features, Approach-1############

data<- read.csv("Ms3sample.csv")

##Shuffling data####
data <- data[sample(nrow(data)), ]
###Retrieving unique values for Property State
data <- data[!duplicated(data$Property.State), ]
View(data)
dim(data)
###Making Property State variable as Row names
rownames(data) <- data$Property.State
View(data)

### Retaining few numeric variables in the dataset for analysis
data <- data[, c("Original.Interest.Rate",
                 "Original.UPB",
                 "Borrower.Credit.Score.at.Origination",
                 "Original.Combined.Loan.to.Value.Ratio..CLTV.")]
View(data)
dim(data)
### Performing standardization on data
data <- scale(data)

###Implementing K Means with nstart=100 ###

# To estimate correct no of clusters,using elbow method
fviz_nbclust(data, kmeans, method = "wss")

k2 <- kmeans(data, centers = 2, nstart = 100)
### Checking the tot.withinss (sum of squared error)
str(k2)

k3 <- kmeans(data, centers = 3, nstart= 100)
str(k3)

k4 <- kmeans(data, centers = 4, nstart= 100)
str(k4)

##Visualizing clusters####

fviz_cluster(k2, data = data)
fviz_cluster(k3, data = data)
fviz_cluster(k4, data = data)

#####So best clustering is with k=2####
k2 <- kmeans(data, centers = 2, nstart = 100)
k2clusters <- k2$cluster
#### OBTAINING MEAN VALUES FOR EACH FEATURE
k_cluster_centers <- k2$centers
# Creating a data frame with cluster centers and feature names
k_cluster_means_df <- data.frame(Cluster = 1:nrow(cluster_centers), cluster_centers)
print(k_cluster_means_df)

####Hierarchical clustering#####
library(cluster)
data<- read.csv("Ms3sample.csv")

##Shuffling data####
data <- data[sample(nrow(data)), ]
###Retrieving unique values for Property State
data <- data[!duplicated(data$Property.State), ]
View(data)
dim(data)
###Making Property State variable as Row names
rownames(data) <- data$Property.State
View(data)

### Retaining few numeric variables in the dataset for analysis
data <- data[, c("Original.Interest.Rate",
                 "Original.UPB",
                 "Borrower.Credit.Score.at.Origination",
                 "Original.Combined.Loan.to.Value.Ratio..CLTV.")]
View(data)
dim(data)
d <- dist(data, method = "euclidean")
##Implementing different linkage methods
hc.c <- hclust(d, method = "complete" )
hc.m <- hclust(d, method = "median" ) 
hc.s <- hclust(d, method = "single")

###PLOTTING THE DENDROGRAMS###
layout(matrix(1:3, nrow=1))
plot(hc.c, cex = 0.6, hang = -1)
plot(hc.m, cex = 0.6, hang = -1)
plot(hc.s, cex = 0.6, hang = -1)


##### CUTTING THE TREES AT OPTIMAL NUMBER OF CLUSTERS
# Restricting number of clusters in each method of linkage

hc.c.clusters <- cutree(hc.c, h = 500000)
hc.c.clusters
hc.c.elements <- table(hc.c.clusters)
print(hc.c.elements)

hc.m.clusters <- cutree(hc.m, h = 175000)
hc.m.elements <- table(hc.m.clusters)
print(hc.m.elements)
###
hc.s.clusters <- cutree(hc.s, h = 35000)
hc.s.elements <- table(hc.s.clusters)
print(hc.s.elements)

dim(data)
# Combining original data and clusters
h_clustered_data <- data.frame(data, Cluster = hc.c.clusters)

# Check variable names in 'clustered_data'
names(h_clustered_data)

# Calculating mean values for each cluster
h_cluster_means <- aggregate(. ~ Cluster, data = h_clustered_data, FUN = mean)

h_cluster_means

# Computing DBSCAN using fpc package####
library("fpc")
set.seed(123)
data <- scale(data)
dim(data)
layout(matrix(1:1, nrow=1))
#Optimal epsilon for k=5
library(dbscan)
dbscan::kNNdistplot(data, k =3)
abline(h = 0.15, lty = 2)

db <- fpc::dbscan(data, eps = 1.2, MinPts = 3)
db
fviz_cluster(db, data = data)

fviz_cluster(db, data = data, stand = FALSE,
             ellipse = FALSE, show.clust.cent = FALSE,
             geom = "point",palette = "jco", ggtheme = theme_classic())


###CHECKING DIFFERENT VALUES FOR EPSILON AND MINIMUM POINTS
db1 <- fpc::dbscan(data, eps = 1.2, MinPts = 7)
db1
db2 <- fpc::dbscan(data, eps = 1.2, MinPts = 9)
db2
db3 <- fpc::dbscan(data, eps =1.75 , MinPts = 3)
db3
db4 <- fpc::dbscan(data, eps = 1.1, MinPts = 3)
db4
#### BEST CLUSTERS######
db3 <- fpc::dbscan(data, eps = 1.25, MinPts = 3)
db3
fviz_cluster(db3, data = data)


# Combining original data with cluster assignments
db_clustered_data <- data.frame(data, Cluster = db3$cluster)

# Checking the structure of clustered_data
str(db_clustered_data)

# Calculating mean values for each cluster
db_cluster_means <- aggregate(. ~ Cluster, data = db_clustered_data, FUN = mean)
print(db_cluster_means)

##comparing clustering results
table(hc.c.clusters, hc.s.clusters) #Perfect alignment
table(hc.m.clusters, hc.c.clusters) #Poor alignment
table(k2clusters,hc.c.clusters) #better alignment than with previous dataset
table(k2clusters, db3$cluster)#Poor alignment

##### MODEL COMPARISON WITH CARET PACKAGE
data <- read.csv("MS3sample.csv")
data$Occupancy.Status <- as.factor(data$Occupancy.Status)
data$Seller.Name <- as.factor(data$Seller.Name)
data$Property.State <- as.factor(data$Property.State)
data$Loan.Purpose <- as.factor(data$Loan.Purpose)
data$Property.Type <- data$ Property.Type
View(data)

CV-1
#####k-fold CROSS VALIDATION WITHOUT REPEATED CVs:TRAINING AND TESTING MODELS 
##Creating a train control object for k-fold cross validation
train_control <- trainControl(method="cv", number=10)
train_control
### TRAINING A DECISTION TREE classifier
model.DT <- train(Occupancy.Status ~., data = data, trControl = train_control, method = "rpart" )
predictors <- setdiff(names(data), "Occupancy.Status")
predictions.DT <- predict(model.DT, newdata = data[, predictors])

confusionMatrix(predictions.DT, data$Occupancy.Status)


### TRAINING A RANDOM FOREST classifier
model.RF<- train(Occupancy.Status ~., data = data, trControl = train_control, method = "rf" )
predictors <- setdiff(names(data), "Occupancy.Status")
predictions.RF <- predict(model.RF, newdata = data[, predictors])

confusionMatrix(predictions.RF, data$Occupancy.Status)


### TRAINING A SVM classifier
model.SVM<- train(Occupancy.Status ~., data = data, trControl = train_control, method = "svmRadial", scale = FALSE )
predictors <- setdiff(names(data), "Occupancy.Status")
predictions.SVM <- predict(model.SVM, newdata = data[, predictors])

confusionMatrix(predictions.SVM, data$Occupancy.Status)



### TRAINING A NN classifier
model.NN <- train(Occupancy.Status ~., data = data, trControl = train_control, method = "nnet" )

predictors <- setdiff(names(data), "Occupancy.Status")
predictions.NN <- predict(model.NN, newdata = data[, predictors])

confusionMatrix(predictions.NN, data$Occupancy.Status)

results <- resamples(list(DT = model.DT, RF = model.RF, SVM = model.SVM, NN = model.NN))
### summary of accuracy distributions as percentiles, boxplots and dotplots
summary(results)

bwplot(results)
dotplot(results)
#### CV-2
#####k-fold CROSS VALIDATION WITH REPEATED CVs: COMPARISON OF CLASSIFIERS
###Creating the train control object with 3 runs of 10-folds each for optimal models with Caret package
control <- trainControl(method="repeatedcv", number=10, repeats=3)
control

### Creating different models
model.DT <- train(Occupancy.Status ~., data = data, trControl = control, method = "rpart" )
model.RF<- train(Occupancy.Status ~., data = data, trControl = control, method = "rf" )
model.SVM<- train(Occupancy.Status ~., data = data, trControl = control, method = "svmRadial", scale = FALSE )
model.NN <- train(Occupancy.Status ~., data = data, trControl = control, method = "nnet" )

results <- resamples(list(DT = model.DT, RF = model.RF, SVM = model.SVM, NN = model.NN))
### summary of accuracy distributions as percentiles, boxplots and dotplots
summary(results)

bwplot(results)
dotplot(results)

#### CV-3
##### bootstrapping CROSS VALIDATION: COMPARISON OF CLASSIFIERS
###Creating the train control object with 3 runs of 10-folds each for optimal models with Caret package
control <- trainControl(method="boot", number= 30,  p = 0.8)
control

### Creating different models
model.DT <- train(Occupancy.Status ~., data = data, trControl = control, method = "rpart" )
model.RF<- train(Occupancy.Status ~., data = data, trControl = control, method = "rf" )
model.SVM<- train(Occupancy.Status ~., data = data, trControl = control, method = "svmRadial", scale = FALSE )
model.NN <- train(Occupancy.Status ~., data = data, trControl = control, method = "nnet" )

results <- resamples(list(DT = model.DT, RF = model.RF, SVM = model.SVM, NN = model.NN))
### summary of accuracy distributions as percentiles, boxplots and dotplots
summary(results)

bwplot(results)
dotplot(results)

#### FEATURE SELECTION METHODS

### FEATURE SELECTION FOR MULTIVARIATE LINEAR REGRESSION MODEL
data <- read.csv("MS3sample.csv")
data$Occupancy.Status <- as.factor(data$Occupancy.Status)
data$Seller.Name <- as.factor(data$Seller.Name)
data$Property.State <- as.factor(data$Property.State)
data$Loan.Purpose <- as.factor(data$Loan.Purpose)
data$Property.Type <- data$ Property.Type
View(data)
data <- data[sample(nrow(data)), ]
train <- data[1:2070, ]
test <- data[2071:2760, ]
numeric_test <- test[, sapply(test, is.numeric)]

numeric_train <- train[, sapply(train, is.numeric)]
dim(numeric_train)
View(numeric_train)
corrV <- cor(numeric_train)

corrplot(cor(numeric_train), method="number", is.corr=FALSE)

model1 = lm(Original.Interest.Rate~., data=numeric_train)
summary(model1)

model2 = update(model1, ~.-Borrower.Credit.Score.at.Origination
                - Original.UPB
                - Number.of.Borrowers
                - Original.Loan.to.Value.Ratio..LTV.
                - Original.Combined.Loan.to.Value.Ratio..CLTV.
                - Debt.To.Income..DTI.)
summary(model2)

##### TRAINING A RANDOM FOREST MODEL WITH RFE
library(caret)
data <- read.csv("MS3sample.csv")
data$Occupancy.Status <- as.factor(data$Occupancy.Status)
data$Seller.Name <- as.factor(data$Seller.Name)
data$Property.State <- as.factor(data$Property.State)
data$Loan.Purpose <- as.factor(data$Loan.Purpose)
data$Property.Type <- data$ Property.Type
View(data)
control <- rfeControl(functions= rfFuncs, method="cv", number=10)
result <- rfe(data[, 2:14], data[, 1], sizes=c(1:10), rfeControl=control)
# plot the results
plot(result, type=c("g", "o"))

# list the chosen features
predictors(result)
###COMPARISON OF MODEL PREDICTIONS

####training RANDOM FOREST MODEL WITH ALL VARIABLES
data <- read.csv("MS3sample.csv")
data$Occupancy.Status <- as.factor(data$Occupancy.Status)
data$Seller.Name <- as.factor(data$Seller.Name)
data$Property.State <- as.factor(data$Property.State)
data$Loan.Purpose <- as.factor(data$Loan.Purpose)
data$Property.Type <- data$ Property.Type
train <- data[1:2070, ]
test <- data[2071:2760, ]
dim(train)
train_control <- trainControl(method="cv", number=10)
model.RF<- train(Original.Interest.Rate ~., data = train, trControl = train_control, method = "rf" )

predictions.RF <- predict(model.RF, newdata = test)
### correlation between actual and predicted values for original model
cor(predictions.RF, test$Original.Interest.Rate)
mse <- mean((predictions.RF - test$Original.Interest.Rate)^2)
mse
rmse <- sqrt(mse)
rmse
###MODEL WITH REDUCED FEATURES SUGGESTED BY RFE METHOD
train_control <- trainControl(method="cv", number=10)
### TRAINING MODEL retaining features identified by RFE results
model.RF.E <- train(Original.Interest.Rate ~ Occupancy.Status+
                      Seller.Name+
                      bondRate+ fedFundsRate+
                      Original.Combined.Loan.to.Value.Ratio..CLTV. +
                      Original.Loan.to.Value.Ratio..LTV.,
                    data = train, trControl = train_control, method = "rf" )

predictions.RF.E <- predict(model.RF.E, newdata = test)
### correlation between actual and predicted values for original model
cor(predictions.RF.E, test$Original.Interest.Rate)
mse <- mean((predictions.RF.E - test$Original.Interest.Rate)^2)
mse
rmse <- sqrt(mse)
rmse



### REGULARIZATION ON NEURAL NETWORKS
data <- read.csv("Normalized_MS3dataset.csv")
train <- data[1:2070, ]
test <- data[2071:2760, ]

model.NN <- neuralnet(Original.Interest.Rate ~., data = train )
model.NN
predictions.NN <- predict(model.NN, newdata = test)
### correlation between actual and predicted values for original model
cor(predictions.NN, test$Original.Interest.Rate)
mse <- mean((predictions.NN - test$Original.Interest.Rate)^2)
mse
rmse <- sqrt(mse)
rmse

#### creating a regularization model
formula <- Original.Interest.Rate ~ .

model.NNR <- neuralnet(
  formula,
  data = train,
  linear.output = TRUE,  
  act.fct = "logistic",  
  algorithm = "rprop+",  # Using resilient backpropagation with weight decay
  stepmax = 1e6 #Setting a large number of steps
)
model.NNR
predictions.NNR <- predict(model.NNR, newdata = test)
### correlation between actual and predicted values for original model
cor(predictions.NNR, test$Original.Interest.Rate)
mse <- mean((predictions.NNR - test$Original.Interest.Rate)^2)
mse
rmse <- sqrt(mse)
rmse


control <- trainControl(method="cv", number=10)
model.SVM<- train(Occupancy.Status ~., data = data, trControl = control, method = "svmRadial", scale = FALSE )





#### WRAPPER- SFS- METHOD ON ENTIRE DATASET REGRESSION MODEL
dim(train)
base.mod <- lm(Original.Interest.Rate ~ 1, data=train)  

# Step 2: Full model with all predictors
all.mod <- lm(Original.Interest.Rate ~ . , data= train) 

# Step 3: Perform step-wise algorithm . direction=both, forward, backward, 
stepMod <- step(base.mod, scope = list(lower = base.mod, upper = all.mod), direction = "forward", trace = 0, steps = 1000)  
#we can check with backward method also
#stepMod <- step(all.mod, direction = "backward", trace = 0, steps = 1000)  

# Step 4: Get the shortlisted variable.

shortlistedVars <- names(unlist(stepMod[[1]])) 
shortlistedVars <- shortlistedVars[!shortlistedVars %in% "(Intercept)"] 
# remove intercept
print(shortlistedVars)

#Model
summary(stepMod)





