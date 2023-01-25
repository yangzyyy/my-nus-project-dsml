# Load packages
library(ggplot2)

# Import data
df_optional<-read.csv('project_residential_price_data_optional.csv')

# Statistical Test1: Actual Sales Price vs Types of Residential Building 
# (categorical/numerical)
# Descriptive: side-by-side box-plot
ggplot(df_optional)+
  aes(x=V.9, fill=as.factor(V.10))+
  geom_boxplot()+
  labs(x="Actual Sales Price",
       title="Actual Sales Price by Residential Type")+
  scale_fill_manual(values=c("#0000FF","#6495ED","#87CEEB","#E0FFFF"))+
  theme_minimal()

# Inferential: ANOVA test
aov_price_type<-aov(V.9 ~ as.factor(V.10), data = df_optional)
summary(aov_price_type)

#OPTIONAL
# Statistical Test2: High Margin Project vs Types of Residential Building
# (two categorical)
# Descriptive: Bar-chart
ggplot(df_optional)+
  aes(x=V.10, fill=as.factor(V.30))+
  geom_bar()+
  labs(x="Types of Residential Building",
       title="High Margin Projects by Residential Type")+
  scale_fill_brewer(palette="Paired")+
  theme_minimal()

#Inferential: Chi-Square Test
chisq_df<-chisq.test(x=df_optional$V.10, y=as.factor(df_optional$V.30))
