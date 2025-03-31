### Project Title
Banks’ Financial Health Assessment using AI/ML

**Author**
Pravin Powale

#### Executive summary
The goal of this project is to leverage machine learning techniques to assess the financial health of US banks using quarterly financial ratio data publicly available from the FDIC. The analysis utilizes datasets containing financial ratios for US banks from 2001 to 2023, , and a list of failed banks. By clustering banks based on their financial ratios, the project aims to categorize them into four groups: Extremely Healthy (Blue), Healthy with Low Risk (Green), Medium Risk (Orange) and High Risk (Red). This indirect approach involves using known healthy banks and failed banks to label the clusters accurately.
Principal Component Analysis (PCA) has been employed to reduce the number of input features to key features, followed by the application of Logistic Regression. This model was evaluated using metrics like accuracy score, recall score, and Confusion Matrix. Classification algorithms such as LogisticRegression, RidgeClassifier, Support Vector Machines (SVM) and Random Forests are implemented along with GridSearchCV to determine best model in each case. The RidgeClassifier with alpha=10000 turned out to be the best model with the highest recall and accuracy scores. This was then revised to a smaller number of features (ratios) using features above 90 percentile SHAP value. The resulting model was then applied to another quarter’s data to predict the individual bank classes for that quarter.
As part of the next steps, additional validation of results from bank financial experts needs to be performed. Also, additional features such as Rate Of Change of key ratios from quarter to quarter can be added. Another variation could be implementing clustering algorithm such as DBSCAN to automatically generate the appropriate number of clusters. This project is significant as it explores the potential of AI/ML techniques in bank regulation, offering a more efficient alternative to traditional rule-based systems. If successful, these techniques could be extended to other regulatory domains, enhancing the monitoring and understanding of the overall health of the economy.

#### Rationale
Why should anyone care about this question?
One of the key processes of bank regulation is to monitor the financial health of banks under supervision. This has a direct impact on the understanding of the overall health of the economy with wide range of global implications. "Business Rules" based systems have been used by the financial regulators in the past. This is an attempt to explore the potential of applying AI/ML techniques to solve this problem, as the rule based systems have to re-configured, re-programmed frequently to ensure acceptable levels of performance. If successful the techniques can be potentially extended to other regulatory domains.

#### Research Question
What are you trying to answer?
Every quarter FDIC publishes 700+ financial ratios of every US bank and makes it available publicly. The main hypothesis is that the financial health of a bank is reflected in these financial ratios, which can be used to build AI/ML models to classify banks into various categories of financial health.
Since labeled data (with financial health categories) is not available publicly, it poses a challenge to use supervised learning models such as Logistic Regression for bank health classification.
To circumvent this problem, an alternate approach of labeling the quarterly bank data is implemented. This consists of first performing a Principal Component Analysis (PCA) and reducing the dimensionality of the data, by conducting Elbow Analysis of the Sigma array generated during Singular Value Decomposition (SVD). A K-Means clustering with n_clusters = 4 is then performed, to form the clusters of banks. The cluster numbers are then used to label each bank.
This submission has considered four categories of financial health. Since the cluster numbers do not directly represent the financial health of banks, a standard candle (in other words a well-known bank) will be used to determine the "Blue" cluster (high financial health) and the "Red" cluster (High financial Risk). A failed bank data file downloaded from FDIC public site is used to identify the “Red” and “Orange” clusters, where “Orange” indicates Medium to High financial Risk. The 4th cluster by elimination is designated as “Green”. In the analysis done for this project, neither the “Green” nor the “Blue” cluster contained any banks from the failed banks data file. 
#### Data Sources
What data will you use to answer you question?
1.	US Banks Financial Quarterly Ratios 2001-2023 – located at https://www.kaggle.com/datasets/neutrino404/all-us-banks-financial-quarterly-ratios-2001-2023  
This dataset has quarterly data of financial ratios for all the US banks that have existed since 2001. The data was downloaded from FDIC website and processed further to computed ratios using the percentages provided for columns ROA, ROAQ, ROAPTX, ROAPTXQ, ROE, and ROEQ.
The dataset contains 700+ ratios and the following key attributes
•	REPYEAR = 'REPORT YEAR'
•	REPDTE = 'The last day of the financial reporting period selected.'
•	STNAME = 'STATE NAME'
•	NAME = 'INSTITUTION NAME'
•	CERT = 'FDIC Certificate Id. A unique NUMBER assigned by the FDIC used to identify institutions and for the issuance of insurance certificates'
2.	Additionally, the dataset “Failed Banks List” available at https://www.fdic.gov/bank-failures/failed-bank-list has been used to label a cluster of banks as “Red” or “Orange” created using the financial ratios. This has been explained with the Python notebook.
#### Methodology
What methods are you using to answer the question?
1. Clean the data source 1 to drop columns that have NaNs. 
2. Select bank data corresponding to a quarter "20230630'
3. Drop numeric columns "CERT", "REPDTE" and "REPYR" which can cause noise during data analysis. 
4. Drop non-numeric columns. Only one was found in the cleaned data, which NAME representing bank name. 
5. Normalize the financial ratios by first subtracting the mean and then dividing the result by standard deviation
4. Drop the columns that contain NaNs after normalization. This can be a result of standard deviation being 0 causing division by 0. 
5. Perform PCA on the 704 dimensional data after initial cleaning. 
6. Perform Elbow Analysis and keep only the components to the left of the Elbow. This reduces the number of Principle Components used to 58.
7. Perform K-means clustering on the 58 dimensional data, with n_clusters = 4. Use the cluster numbers to label the banks in each cluster.
8. Use the dataframe before application of PCA as X. Use the labels generated as y. Perform train_test_split on the original dataset to create classification models.
9. Create a LogisticRegression classifier (lgr) and fit the data (X_train, y_train)
10. Use lgr to calculate the accuracy, confusion matrix, and recall.
11. Create RidgeClassifier, SVC and RandomForest classifiers, train them and find accuracy and recall.
12. Create GridSearchCV for RidgeClassifier, SVC and RandomForest classifiers, to find best model among each category.
13. RidgeClassifier with alhpa=10000 was the best model among all considered models.
14. All classfiers used class_weights="balanced" to account for imbalanced cluster data.
15. A SHAP value analysis was performed for the best Ridge model and a feature set with values above 90 percentile was generated and used to create a revised RidgeClassifier model with smaller number of features, which are likely to be available in all quarters.
16. The revised RidgeClassifier was used to predict bank classifications for another quarter ending 20221231.
#### Results
What did your research find?
The clustering analysis performed (in the absence of publicly available labeled data, along with identification of “standard candles”  for each cluster seems to be working at first sight but still needs additional validation of results from the experts in the financial regulatory sector.
The predictions performed on 20221231 quarterly data were spot checked for several banks and found to be consistent with labels generated using the K-Means clustering. There were a handful of deviations found which point to some of the wekanesses of the model and require further analysis. 
#### Next steps
What suggestions do you have for next steps?
A potential approach to address the devaitaions would be to add few Rate Of Change (ROC) features representing change of ratios from one quarter to the next. This may improve the model performance.
Other potential future revsion could be the use of DBSCAN clustering algorithm to generate the number of clusters from the data itself.
#### Outline of project

- [Link to notebook 1](https://github.com/pravinpowale/Berkeley-AI-ML-Capstone-Project-Final/blob/main/BDA%20-%20Notebook%20FINAL%20-%201.ipynb)
- [Link to notebook 2]()
- [Link to notebook 3]()


##### Contact and Further Information
Pravin Powale
email - pravinpowale@yahoo.com