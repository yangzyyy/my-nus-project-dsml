# packages required
library(tidyverse)
library(vtable)
library(reshape2)
library(caret)
library(patchwork)
library(factoextra)
library(e1071)
library(caTools)
library(ROCR)
library(randomForest)

# reading data
setwd("~/Documents/NUS/STUDIES/DSA5101/stats project")
data <- read_csv("project_residential_price_data_optional.csv")

print(str(data))

# print summary table of data
sumtable(data = data)

# extract out independent variables
X = data[,-28]
y = data[,28]

# univariate analysis
# histogram for the target variable
ggplot(data=data, mapping = aes(x = V.9)) +
  geom_histogram(fill = "cornflowerblue", bins = 20) +
  labs(x="Sales price (10000 IRR)") +
  theme_minimal()

# histogram for independent variables
ggplot(data=gather(X), aes(value)) + 
  geom_histogram(fill = "cornflowerblue", bins = 10) + 
  facet_wrap(~key, scales = 'free_x')

# boxplot for target variable
ggplot(data=gather(data), aes(value)) + 
  geom_boxplot(fill = "cornflowerblue") + 
  facet_wrap(~key, scales = 'free_x')

# bivariate analysis
# correlation heatmap
melted_data = melt(cor(data))
ggplot(data = melted_data, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile()

# histogram of target variable by different categorical groups
ggplot(data, aes(x = V.9, fill = as.factor(V.1))) + 
  geom_histogram(alpha = 0.5, position = "identity") +
  labs(x = "Sales price (10000 IRR)", fill = "Project locality") +
  theme_minimal()

ggplot(data, aes(x = V.9, fill = as.factor(V.10))) + 
  geom_histogram(alpha = 0.5, position = "identity") +
  facet_wrap(~V.10, scales = 'free_x') +
  labs(x = "Sales price (10000 IRR)", fill = "Type of residential building") +
  theme_minimal()

# contigency table
table(data$V.1, data$V.10)

# scatter plot of price at the beginning vs actual price
ggplot(data = data, mapping = aes(x=V.8, y=V.9)) +
  geom_point(col = "cornflowerblue") +
  labs(x = "Price at the beginning of the project",
       y = "Actual sales price") +
  theme_minimal()

# Statistical Test1: Actual Sales Price vs Types of Residential Building 
# (categorical/numerical)
# Descriptive: side-by-side box-plot
ggplot(data)+
  aes(x=V.9, fill=as.factor(V.10))+
  geom_boxplot()+
  labs(x="Actual Sales Price",
       title="Actual Sales Price by Residential Type")

# Inferential: ANOVA test
aov_price_type<-aov(V.9 ~ as.factor(V.10), data = data)
print(summary(aov_price_type))

# OPTIONAL
# Statistical Test2: High Margin Project vs Types of Residential Building
# (two categorical)
# Descriptive: Bar-chart
ggplot(data)+
  aes(x=V.10, fill=as.factor(V.30))+
  geom_bar()+
  labs(x="Types of Residential Building",
       title="High Margin Projects by Residential Type")

# Inferential: Chi-Square Test
chisq_df<-chisq.test(x=data$V.10, y=as.factor(data$V.30))

# Hypothesis 2
# Descriptive analysis
v2<-ggplot(data)+
  aes(x=V.2, y=V.9)+
  geom_point(col="red")+
  labs(y="sales price", x="V2", title="V9~V2")+
  geom_smooth(method="lm")
v3<-ggplot(data)+aes(x=V.3, y=V.9)+
  geom_point(col="red")+
  labs(y="sales price", x="V3", title="V9~V3")+
  geom_smooth(method="lm")
v4<-ggplot(data)+aes(x=V.4, y=V.9)+
  geom_point(col="red")+
  labs(y="sales price", x="V4", title="V9~V4")+
  geom_smooth(method="lm")
v5<-ggplot(data)+aes(x=V.5, y=V.9)+
  geom_point(col="red")+
  labs(y="sales price", x="V5", title="V9~V5")+
  geom_smooth(method="lm")
v6<-ggplot(data)+
  aes(x=V.6, y=V.9)+
  geom_point(col="red")+
  labs(y="sales price", x="V6", title="V9~V6")+
  geom_smooth(method="lm")
v7<-ggplot(data)+
  aes(x=V.7, y=V.9)+
  geom_point(col="red")+
  labs(y="sales price", x="V7", title="V9~V7")+
  geom_smooth(method="lm")
v8<-ggplot(data)+
  aes(x=V.8, y=V.9)+
  geom_point(col="red")+
  labs(y="sales price", x="V8", title="V9~V8")+
  geom_smooth(method="lm")
v11<-ggplot(data)+
  aes(x=V.11, y=V.9)+
  geom_point(col="red")+
  labs(y="sales price", x="V11", title="V9~V11")+
  geom_smooth(method="lm")
v12<-ggplot(data)+
  aes(x=V.12, y=V.9)+
  geom_point(col="red")+
  labs(y="sales price", x="V12", title="V9~V12")+
  geom_smooth(method="lm")
v13<-ggplot(data)+
  aes(x=V.13, y=V.9)+
  geom_point(col="red")+
  labs(y="sales price", x="V13", title="V9~V13")+
  geom_smooth(method="lm")
v14<-ggplot(data)+
  aes(x=V.14, y=V.9)+
  geom_point(col="red")+
  labs(y="sales price", x="V14", title="V9~V14")+
  geom_smooth(method="lm")
v15<-ggplot(data)+
  aes(x=V.15, y=V.9)+
  geom_point(col="red")+
  labs(y="sales price", x="V15", title="V9~V15")+
  geom_smooth(method="lm")
v16<-ggplot(data)+
  aes(x=V.16, y=V.9)+
  geom_point(col="red")+
  labs(y="sales price", x="V16", title="V9~V16")+
  geom_smooth(method="lm")
v17<-ggplot(data)+
  aes(x=V.17, y=V.9)+
  geom_point(col="red")+
  labs(y="sales price", x="V17", title="V9~V17")+
  geom_smooth(method="lm")
v18<-ggplot(data)+
  aes(x=V.18, y=V.9)+
  geom_point(col="red")+
  labs(y="sales price", x="V18", title="V9~V18")+
  geom_smooth(method="lm")
v19<-ggplot(data)+
  aes(x=V.19, y=V.9)+
  geom_point(col="red")+
  labs(y="sales price", x="V19", title="V9~V19")+
  geom_smooth(method="lm")
v20<-ggplot(data)+
  aes(x=V.20, y=V.9)+
  geom_point(col="red")+
  labs(y="sales price", x="V20", title="V9~V20")+
  geom_smooth(method="lm")
v21<-ggplot(data)+
  aes(x=V.21, y=V.9)+
  geom_point(col="red")+
  labs(y="sales price", x="V21", title="V9~V21")+
  geom_smooth(method="lm")
v22<-ggplot(data)+
  aes(x=V.22, y=V.9)+
  geom_point(col="red")+
  labs(y="sales price", x="V22", title="V9~V22")+
  geom_smooth(method="lm")
v23<-ggplot(data)+
  aes(x=V.23, y=V.9)+
  geom_point(col="red")+
  labs(y="sales price", x="V23", title="V9~V23")+
  geom_smooth(method="lm")
v24<-ggplot(data)+
  aes(x=V.24, y=V.9)+
  geom_point(col="red")+
  labs(y="sales price", x="V24", title="V9~V24")+
  geom_smooth(method="lm")
v25<-ggplot(data)+
  aes(x=V.25, y=V.9)+
  geom_point(col="red")+
  labs(y="sales price", x="V25", title="V9~V25")+
  geom_smooth(method="lm")
v26<-ggplot(data)+
  aes(x=V.26, y=V.9)+
  geom_point(col="red")+
  labs(y="sales price", x="V26", title="V9~V26")+
  geom_smooth(method="lm")
v27<-ggplot(data)+
  aes(x=V.27, y=V.9)+
  geom_point(col="red")+
  labs(y="sales price", x="V27", title="V9~V27")+
  geom_smooth(method="lm")
v28<-ggplot(data)+
  aes(x=V.28, y=V.9)+
  geom_point(col="red")+
  labs(y="sales price", x="V28", title="V9~V28")+
  geom_smooth(method="lm")
v29<-ggplot(data)+
  aes(x=V.29, y=V.9)+
  geom_point(col="red")+
  labs(y="sales price", x="V29", title="V9~V29")+
  geom_smooth(method="lm")

# Plot scatter diagram
v2+v3+v4+v5+v6+v7+v8+v11+v12
v13+v14+v15+v16+v17+v18+v19+v20+v21
v22+v23+v24+v25+v26+v27+v28+v29

# Inferential Analysis (Regression)
# From the 26 figures, we can basically see that some independent variables have
# obvious correlation with V9, while some others are ambiguous only from the
# scatter diagram. 

# Obvious variables, like V8
lm8<-lm(V.9~V.8, data=data)
summary(lm8)

# ambiguous variables, like V11,V28
lm11<-lm(V.9~V.11, data=data)
summary(lm11)

lm28<-lm(V.9~V.28, data=data)
summary(lm28)

# all input correlation test with V9

#lm2<-lm(V.9~V.2, data=data)
#lm3<-lm(V.9~V.3, data=data)
#lm4<-lm(V.9~V.4, data=data)
#lm5<-lm(V.9~V.5, data=data)
#lm6<-lm(V.9~V.6, data=data)
#lm7<-lm(V.9~V.7, data=data)
#lm8<-lm(V.9~V.8, data=data)
#lm11<-lm(V.9~V.11, data=data)
#lm12<-lm(V.9~V.12, data=data)
#lm13<-lm(V.9~V.13, data=data)
#lm14<-lm(V.9~V.14, data=data)
#lm15<-lm(V.9~V.15, data=data)
#lm16<-lm(V.9~V.16, data=data)
#lm17<-lm(V.9~V.17, data=data)
#lm18<-lm(V.9~V.18, data=data)
#lm19<-lm(V.9~V.19, data=data)
#lm20<-lm(V.9~V.20, data=data)
#lm21<-lm(V.9~V.21, data=data)
#lm22<-lm(V.9~V.22, data=data)
#lm23<-lm(V.9~V.23, data=data)
#lm24<-lm(V.9~V.24, data=data)
#lm25<-lm(V.9~V.25, data=data)
#lm26<-lm(V.9~V.26, data=data)
#lm27<-lm(V.9~V.27, data=data)
#lm28<-lm(V.9~V.28, data=data)
#lm29<-lm(V.9~V.29, data=data)

# modify data for model training purposes
names(data)[names(data) == 'V.1'] <- 'new_V.1.'
names(data)[names(data) == 'V.10'] <- 'new_V.10.'
data$new_V.1.<-as.factor(data$new_V.1.)
data$new_V.10.<-as.factor(data$new_V.10.)
data$V.30<-as.factor(data$V.30)

# predicting housing price part

data <- data[,1:29]
#seed to generate random number
set.seed(1)  

# train-test set split
# sample() is takes a sample of the specified size from the elements
train = sample(nrow(data), 0.7*nrow(data))
test = setdiff(seq_len(nrow(data)), train)

cor = cor(data[train,2:28])
cor[27,abs(cor[27,]) > 0.4]

#regression on full model
lm = lm(V.9~., data = data[train,]) 
summary(lm)

#regression model using variable with |correlation with housing price|  > 0.5
#lm1 = lm(V.9~new_V.1.+new_V.10.+V.4+V.5+V.8+V.12+V.13+V.15+V.16+V.17+V.19+V.21+V.22+V.23+V.24+V.25+V.26+V.27+V.29, data = data[train,]) 
#summary(lm1)

new_cor = cor(data[train, c(4,7,8,10,11,13,15, 24,25,27)])#variables with high significance
new_cor

#regression model using variable with small p value
# Only keep one of V.12, V.13, V.15, V.17, V.26, V.29, because they have large correlation (>0.9)
lm2 = lm(V.9~new_V.1.+V.4+V.5+V.7+V.8+V.12+V.27, data = data[train,]) 
summary(lm2)

#regression model using variable with small p value
lm3 = lm(V.9~V.5+V.7+V.8, data = data[train,]) 
summary(lm3)

##Model with Top Principal Components/Feature Extractio##
pca<-prcomp(data[train,-c(1,29,28)],center=T,scale=T) #PCA
pred <- predict(pca, newdata=data[test,-c(1,29,28)])  #PC prediction for testing data

res.ind <- get_pca_ind(pca)
res.ind$coord  # Coordinates

get_eigenvalue(pca)
fviz_contrib(pca, choice = "var", axes = 1:6, top =10)

res.var <- get_pca_var(pca)
res.var$coord # Coordinates
res.var$contrib # Contributions to the PCs

data_training_new<-res.ind$coord[,1:6] #The first 6 PCs
data_training_new<-cbind(data_training_new,data[train, c(1,29,28)]) #Add categorical variables, and House_Price
colnames(data_training_new)<-c("PC1","PC2","PC3","PC4","PC5", "PC6", "V.1.", "V.10.", "V.9") #give the column name

data_testing_new<-pred[,1:6]  #Choose top 5 PCs of testing data
data_testing_new<-cbind(data_testing_new,data[test, c(1,29,28)])
colnames(data_testing_new)<-c("PC1","PC2","PC3","PC4","PC5",  "PC6","V.1.", "V.10.", "V.9")

pca_train_df = as.data.frame(data_training_new)
pca_test_df = as.data.frame(data_testing_new)
data_new<-rbind(pca_train_df,pca_test_df) #new dataset with Top 10 Principalcomponents

pca_lm<-lm(V.9~., data = pca_train_df) #regression model using top PC+categorical
summary(pca_lm)

pca_lm1<-lm(V.9~V.1.+V.10.+PC1+PC2+PC5+PC6, data = pca_train_df) #regression model using top PC3+categorical
summary(pca_lm1)

train.control <- trainControl(method = "cv", number = 5)  #define the method, 5-fold cross-validation

##Full model##
model <- train(V.9 ~., data = data, method = "lm",
               trControl = train.control)  #Training and test model
print(model)

##With selected variables ONLY
model2<- train(V.9 ~., data = data[,c(5,7,8,28)], method = "lm",
               trControl = train.control)
print(model2)

##With top 3 principal components
model3<- train(V.9 ~V.1.+V.10.+PC1+PC2+PC5+PC6, data = data_new, method = "lm",
               trControl = train.control)
print(model3)

anova(lm, lm3)

#according to RMSE, interpretatbility, and prevention of overfitting issue, linear regression with variable V.7, V.8 and V.5 is selected as the final model
lm_final<-lm(V.9~., data = data[,c(5,7,8,28)])
summary(lm_final)

#classification on margin project type (V.30) part
data <- read_csv("project_residential_price_data_optional.csv")
names(data)[names(data) == 'V.1'] <- 'new_V.1.'
names(data)[names(data) == 'V.10'] <- 'new_V.10.'
data$new_V.1.<-as.factor(data$new_V.1.)
data$new_V.10.<-as.factor(data$new_V.10.)
data$V.30<-as.factor(data$V.30)
classification_data = data[, -28]

#support vector machine
svm = svm(V.30 ~ ., data = classification_data[train,], kernel = "linear", cost = 2, scale = FALSE)
summary(svm)

logistic <- glm(V.30 ~., 
                data = classification_data[train,], 
                family = "binomial")
logistic

# Summary
summary(logistic)

#select significant variables
logistic1 <- glm(V.30 ~new_V.1.+V.2+V.3+V.4+V.5+V.6+V.8+V.11+V.12+V.14+V.16+V.19+V.21+V.22+V.23+V.29+new_V.10., 
                 data = classification_data[train,], 
                 family = "binomial")
logistic1

# Summary
summary(logistic1)

#select significant variables
logistic2 <- glm(V.30 ~new_V.1.+V.2+V.3+V.4+V.5+V.6+V.8+V.11+V.14+V.16+V.21+V.29+new_V.10., 
                 data = classification_data[train,], 
                 family = "binomial")
logistic2

# Summary
summary(logistic2)

#select significant variables

logistic3 <- glm(V.30 ~new_V.1.+V.2+V.3+V.4+V.5+V.6+V.8+V.11+V.14+V.16+V.21+V.29, 
                 data = classification_data[train,], 
                 family = "binomial")
logistic3

# Summary
summary(logistic3)

rf = randomForest(V.30~., data=classification_data[train,], ntrees=500,proximity=TRUE)

rf

# Importance plot
importance(rf)

# Variable importance plot
varImpPlot(rf)