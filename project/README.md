
# Project: Travel Insurance Prediction

Tour & Travels Company Is offering Travel Insurance package to their customers. Company requires to know the which customers would be interested to buy it based on it's database history. The data is provided for almost 2000 of its previous customers and I am required to build an intelligent model that can predict if the customer will be interested to buy the travel insurance package based on certain parameters given on dataset.


## Data source

Given data source can be found at Kaggle with the link provided below:

  **Travel Insurance Prediction Data**  

  https://www.kaggle.com/datasets/tejashvi14/travel-insurance-prediction-data/data



## Summary
 
Travel insurance data was analyzed, and results presented.
There were 8 different features with 1 target – travel insurance. Features consisted of object (categorical) and int64 continuous numeric dtypes. Some object data types were converted to binary numeric. EDA showed some insights:

- There are almost double people without travel insurance than with.
- Most of the binary feature’s groups are imbalanced with strongly dominating 0 or 1 group.
- Null hypothesis showed that frequent flyers tend to buy travel insurance more.
- More frequent flyers are from private sector than government.
- People who have travelled abroad atleast once is more likely to take travel insurance. Frequent flyers uses travel insurance even more despite they have or not ever travelled abroad. This tendency seems logical because people tend to protect themselves and belongings more when traveling.
- People 25-26 and >32 tend to take travel insurance more.
- Purchase of travel insurance strongly increases within people of annual income > 1.3M.
- Logistic regression indicated that there is statistically significant association between family size and the purchase of travel insurance.
- Spearman’s correlation of all features between each other showed too weak to state it’s correlated. And because analysis target will be travel insurance, non-correlative predictors between each other is a benefit for prediction. It reduced redundancy, reduces multicollinearity, helps to avoid overfitting, simplifies model interpretation.

For ML predictions, 3 model has been chosen one linear and two non-linears: Logistic regression, HistGradientBoostingClassifier and RandomForestClassifier. Dataset has been splitted 75/25 train/test. Pipelines used with scalers. Also models were trained with oversampled target data, to check score. Everything cross-validated. The best model was fitted with most popular hyperparameters set using RandomizedSearchCV. The best results were shown by HistGradientBoosterClassifier model with 0.841 accuracy.  

**What could have been done more**: 
- H0 hypothesis with different features could have been analysed;  
- Use Variance Inflation Factor (VIF) to check for Multicollinearity;  
- Try SupportVectorMachines (SVM) model;
- Visualize summarized ML results to look for more insights.  
- In functions_sandbox.py edit some of the functions by adding more flexibility and durability.

