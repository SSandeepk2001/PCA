'''# Dimension Reduction - PCA - In supervised learning (predictive modeling), within regression we have collinearity problem. 
# This can be addressed using PCA. PCs which are the end result of PCA application would be uncorrelated.

# CRISP-ML(Q):
    Business & Data Understanding:
        Business Problem: Huge number of features to analyze requires a lot of compute and is time consuming
        Business Objective: Minimize the compute & time for processing
        Business Constraints: Minimize the features deletion
        
        Success Criteria:
            Business: Reduce the compute required by 50%
            ML: Get at least 50% compression
            Economic: ROI of at least $500K over a period of 1 year

# Data Collection

# Data: 
#    The university details are obtained from the US Higher Education Body and is publicly available for students to access.
# 
# Data Dictionary:
# - Dataset contains 25 university details
# - 8 features are recorded for each university
# 
# Description:
# - Univ - University Name
# - State - Location (state) of the university
# - SAT - Average SAT score for eligibility
# - Top10 - % of students who ranked in top 10 in their previous academics
# - Accept - % of students admitted to the universities
# - SFRatio - Student to Faculty ratio
# - Expenses - Overall cost in USD
# - GradRate - % of students who graduate'''

# #### Install the required packages if not available

# !pip install feature_engine
# !pip install dtale


# **Importing required packages**


import numpy as np
import pandas as pd
import sweetviz
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.decomposition import PCA
from kneed import KneeLocator

from sqlalchemy import create_engine, text

user = 'root' # user name
pw = 'Sandeep@2001' # password
db = 'univ_db' # database
# creating engine to connect database
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# **Import the data**
University = pd.read_excel(r"C:\Users\sande\OneDrive\Desktop\PCA_code\PCA_code\University_Clustering.xlsx")
University
# dumping data into database 
# name should be in lower case
University.to_sql('university_clustering', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

# loading data from database
sql = text('select * from univ_tbl')

df = pd.read_sql_query(sql, con = engine.connect())

print(df)

df.info()

# # EXPLORATORY DATA ANALYSIS (EDA) / DESCRIPTIVE STATISTICS
# ***Descriptive Statistics and Data Distribution Function***
res = df.describe()

# Drop the unwanted features
df1 = df.drop(["UnivID"], axis = 1)
df1.info()

# AutoEDA
# Automated Libraries
# import sweetviz
my_report = sweetviz.analyze([df1, "df1"])
my_report.show_html('Report.html')

# Data Preprocessing
# Checking Null Values
df1.isnull().sum()

# PCA can be implemented only on Numeric features
df1.info()
numeric_features = df1.select_dtypes(exclude = ['object']).columns
numeric_features

# Define the Pipeline steps
# Define PCA model
pca = PCA(n_components = 6)

# Make Pipeline

# **By using mean imputation, null values can be imputed**
# **Data has to be standardized to address the scale difference**

num_pipeline = make_pipeline(SimpleImputer(strategy = 'mean'), StandardScaler(), pca)
num_pipeline

# Pass the raw data through pipeline
processed = num_pipeline.fit(df1[numeric_features]) 
processed

# Apply the pipeline on the dataset
univ = pd.DataFrame(processed.transform(df1[numeric_features]))
univ

# Save the End to End PCA pipeline with Imputation and Standardization
import joblib
joblib.dump(processed, 'PCA_DimRed')

import os
os.getcwd()

# Import the pipeline
model = joblib.load("PCA_DimRed")
model

# ## Apply the saved model on to the Dataset to extract PCA values
pca_res = pd.DataFrame(model.transform(df1[numeric_features]))
pca_res

# PCA weights
model['pca'].components_

# Take a closer look at the components
components = pd.DataFrame(model['pca'].components_, columns = numeric_features).T
components.columns = ['pc0', 'pc1', 'pc2', 'pc3', 'pc4', 'pc5']

components

print(model['pca'].explained_variance_ratio_)

var1 = np.cumsum(model['pca'].explained_variance_ratio_)

print(var1)

# Variance plot for PCA components obtained 
plt.plot(var1, color = "red")

# KneeLocator
# Refer the link to understand the parameters used: https://kneed.readthedocs.io/en/stable/parameters.html     

# from kneed import KneeLocator
kl = KneeLocator(range(len(var1)), var1, curve = 'concave', direction = "increasing") 
# The line is pretty linear hence Kneelocator is not able to detect the knee/elbow appropriately
kl.elbow
plt.style.use("seaborn")
plt.plot(range(len(var1)), var1)
plt.xticks(range(len(var1)))
plt.ylabel("variance")
plt.axvline(x = kl.elbow, color = 'r', label = 'axvline - full height', ls = '--')
plt.show()

# Kneelocator recommends 3 PCs as the ideal number of features to be considered
# PCA for Feature Extraction

# Final dataset with manageable number of columns (Feature Extraction)

final = pd.concat([df.Univ, pca_res.iloc[:, 0:3]], axis = 1)
final.columns = ['Univ', 'pc0', 'pc1', 'pc2']
final

# Scatter diagram
ax = final.plot(x = 'pc0', y = 'pc1', kind = 'scatter', figsize = (12, 8))
final[['pc0', 'pc1', 'Univ']].apply(lambda x: ax.text(*x), axis = 1)

# Prediction on new data
newdf = pd.read_excel(r"C:\Users\sande\OneDrive\Desktop\PCA_code\PCA_code\new_Univ_4_pred.xlsx")
newdf

# Drop the unwanted features
newdf1 = newdf.drop(["UnivID"], axis = 1)

num_feat = newdf1.select_dtypes(exclude = ['object']).columns

new_res = pd.DataFrame(model.transform(newdf1[num_feat]))

new_res

