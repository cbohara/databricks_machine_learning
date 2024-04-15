# Scalable ML with Apache Spark
## Intro
Send code to driver   
Driver optimizes and distributes code to workers   
Each worker runs a JVM process   

Spark under the hood optimizations   
- Unresolved logical plan = syntax checks
- Logical plan > catalyst optimizer > optimized logic plan   
- Physical plans for cost based optimization   
- Turns into code to be executed   

Delta lake   
Built on top of parquet   
Time travel = helpful to go back to a previous model version   

Machine learning   
- Predict based on patterns 
- Without explicit programming
- Funtion that maps features to outputs   

Supervised learning - Use labeled data mapping input to output - focus on the course   
Unsupervised learning - Find patterns/groupings of unlabeled data   
Semi-supervised learning - Combine labeled and unlabeled data   
Reinforcement learning - Learn from feedback   

ML workflow:   
Define biz use case > prep - define success, identify contraints > collect data > build features out of data > create models > choose the best model > deploy model > evaluate if this model is driving biz value or needs to be retrained    

Want true positives + true negatives   
Any model that has no errors is over fitted and will not generalize well    

## Exploring data
Summary stats:   
mean/average = central tendency for the data point   

stddev - measures how much data points are spread out
- low stddev = data points tend to reside close to the mean = not much spread
- high stddev = data points are spread out over a wide range of values   

interquartile range (IQR) - another measure of variability that is less influenced by outliers       
- arrange the data is ascending order 
- find the median value = split the data set in half   
- then take the median of the first half and second half to generate quarters   
- the median of the first half is first quartile Q1 and the median of the second half is third quartile Q3   
- IQR = Q3 - Q1   
when using summary()
- 25% - 25% of the data falls below this value   
- 50% - 50% of the data falls below this value 
- 75% - 75% of the data falls below this value    

use `dbutils.data.summarize(df)` to get very helpful stats   
for numeric features   
- count 
- missing 
- mean
- std dev 
- zeros
- min 
- median 
- max 

for categorical features 
- count 
- missing 
- unique 

handling null values   
- drop records 
- numeric - replace mean/median/mode/zero 
- categorical - replace with mode = most frequent value in the data set or create a special column for null   
- if you do any imputation techniques, you must include an additional field specifying that the field was imputed   

```
from pyspark.ml.feature import Imputer

# create data frame with missing values 
df = spark.createDataFrame([
    (1.0, float("nan")),
    (2.0, float("nan")),
    (float("nan"), 3.0),
    (4.0, 4.0),
    (5.0, 5.0)
], ["a", "b"])

# specify input columns and new columns to be generated with imputation
# default is to use mean for imputation
imputer = Imputer(inputCols=["a", "b"], outputCols=["out_a", "out_b"])

# fit will compute the mean value for each column
model = imputer.fit(df)

# transform will replace the null value with the mean = default approach 
model.transform(df).show()

print(imputer.explainParams())
```

Spark ML API:   
transformers 
- transforms 1 DF to another 
- accepts DF as input and returns DF with 1+ columns appended 
- simply apply rule based transformations
estimators 
-  an algorithm which can be fit on a dataframe to produce a transformer 
- ex: a learning algo is an estimator which trains on a DF and produces a model   

Use randomSplit to divide the data   
Providing the seed value is arbitrary    
Providing it is optional   
Useful to provide when you need reproducible results   
```
train_df, test_df = airbnb_df.randomSplit([.8, .2], seed=42)
```

Transforming these prices into their logarithmic scale can often help normalize or reduce the skewness, making the distribution more symmetric and easier to work with for analysis purposes.   

By taking the logarithm of price values, you essentially create a new scale that compresses larger values and expands smaller values. This transformation can make the distribution of prices more symmetrical, which can aid in statistical analysis, modeling, or visualizations, especially when dealing with highly skewed or heavily right-tailed distributions.    

```
train_df = train_df.select(log("price").alias("log_price"))
```

Create 1 model that predicts average price and one the predicts median price   
Use RMSE (root mean square error) to evaluate   
Lower RMSE values indicate better model performance, as they signify smaller differences between predicted and observed values    

## Linear Regression
Goal - find the line of best fit   
y = mx + b   
We provide the data - x,y of the data points   
x is the feature and y is the label   
The model finds the m and b - m is the slope and b is the intercept to the y axis   

Goal - minimize the residuals = distance between the best fit line and the data points   
Regression evaluators - measures the closeness between the actual value and predicted error   

Evaluation metrics  
- loss = actual - prediction   
- absolute loss = abs(actual - prediction)
- squared loss = (actual - prediction)^2 - gets rid of the negative and positive errors canceling each other out   
- root mean squared error (RMSE) - fancier version of loss above - varies based on the scale of the data   
- r squared - compare sum of all errors to the average and compare sum of all errors to the prediction - range 0 (no value) - 1 (perfect fit)   

Assumptions
- assume there is a linear relationship 
- observations are independent of each other - means the data points aren't derived from other data points  
- data is normally distributed 
- variance is consistent from one data point to the next   

scikit learn - popular single-node ML library   
ML with spark - scale out + speed up   
- MLlib - based on RDDs - maintenance mode
- Spark ML - use dataframes - active development   

[Spark ML linear regression](https://spark.apache.org/docs/latest/ml-classification-regression.html#linear-regression)   

```
# specify the configuration
lr = LinearRegression(featuresCol="bedrooms", labelCol="price")
# teach the model based on the training data frame the relationship between bedrooms and price
lr_model = lr.fit(train_df)
```

price(y) = coefficient(m) * bedrooms(x) + y-intercept(b)
What does the lr_model contain? 
- coefficients(m) - the value you multiple the number of bedrooms by - ex: if the model learned that each bedroom adds $1000 to the price, then the coefficient would be 1000   
- intercept(b) - constant term - it represented the base price of the house with zero bedrooms   

[VectorAssembler](https://spark.apache.org/docs/latest/ml-features.html#vectorassembler)
Combines a list of columns to create a single column as a list   

ex: hour, mobile, userFeatures becomes features column   

 id | hour | mobile | userFeatures     | clicked | features
----|------|--------|------------------|---------|-----------------------------
 0  | 18   | 1.0    | [0.0, 10.0, 0.5] | 1.0     | [18.0, 1.0, 0.0, 10.0, 0.5]

When working with multiple columns, x is not limited to one column   
It is a vector of all the columns   
price(y) = coefficient(m) * x + y-intercept(b)   

non-numeric features
- categorical - strings with no ordering 
    - can use one hot encoding to create a new column per each value and then set the value to 1 or 0   
    - works when we only have a few values (ie: Animal - dog, cat, bird)
    - use SparseVector when working with large variety of values (ie: all animals in a zoo)
- ordinal features - strings with order (ex: small, medium, large)

convert non-numeric values to numbers   

[StringIdexer](https://spark.apache.org/docs/latest/ml-features.html#stringindexer)    
encodes a string column of labels to a column of label indices   
Ordering options supported
- freqDesc - most frequent label gets 0 - default  
- freqAsc - least frequent label gets 0

Assume that we have the following DataFrame with columns id and category:

```
 id | category
----|----------
 0  | a
 1  | b
 2  | c
 3  | a
 4  | a
 5  | c
```

category with a string column with 3 labels - a, b, c   
a -> 0 - gets 0 because it is the most frequent value   
c -> 1    
b -> 2 - gets 2 because it is the least frequent value   

[OneHotEncoder](https://spark.apache.org/docs/latest/ml-features.html#onehotencoder)
one hot encoder doesn't accept string input, it requires numeric input
so the string indexer is first ran to transform strings into numbers, and then we can do one hot encoding on the number representation of the string

[Pipelines](https://spark.apache.org/docs/latest/ml-pipeline.html)   
Combine multiple algorithms into a single workflow    

Estimator - used to create a transformer 
- ex: LinearRegression is an estimator which trains on a DF and produces a model 
```
lr = LinearRegression(labelCol="price", featuresCol="features")
```

Transformer - algorithm that transforms one DF to another
- ex: ML model is a transformer which when fit on the training DF will create another DF with predictions 
```
lr_model = lr.fit(train_df)
```

Pipeline - chains multiple estimators and transformers together to specify a ML workflow   

https://spark.apache.org/docs/latest/ml-pipeline.html#example-estimator-transformer-and-param
```
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression

# Prepare training data from a list of (label, features) tuples.
training = spark.createDataFrame([
    (1.0, Vectors.dense([0.0, 1.1, 0.1])),
    (0.0, Vectors.dense([2.0, 1.0, -1.0])),
    (0.0, Vectors.dense([2.0, 1.3, 1.0])),
    (1.0, Vectors.dense([0.0, 1.2, -0.5]))], ["label", "features"])

# Create a LogisticRegression instance. This instance is an Estimator.
lr = LogisticRegression(maxIter=10, regParam=0.01)

# Print out the parameters, documentation, and any default values.
print("LogisticRegression parameters:\n" + lr.explainParams() + "\n")

# Learn a LogisticRegression model. This uses the parameters stored in lr.
model1 = lr.fit(training)

# Since model1 is a Model (i.e., a transformer produced by an Estimator), we can view the parameters it used during fit().
print("Model 1 was fit using parameters: ")
print(model1.extractParamMap())

# We may alternatively specify parameters using a Python dictionary as a paramMap
paramMap = {lr.maxIter: 20}
# Specify 1 Param, overwriting the original maxIter.
paramMap[lr.maxIter] = 30  
# Specify multiple Params.
paramMap.update({lr.regParam: 0.1, lr.threshold: 0.55})  

# Prepare test data to validate the model
test = spark.createDataFrame([
    (1.0, Vectors.dense([-1.0, 1.5, 1.3])),
    (0.0, Vectors.dense([3.0, 2.0, -0.1])),
    (1.0, Vectors.dense([0.0, 2.2, -1.5]))], ["label", "features"])

# Make predictions on test data using the Transformer.transform() method.
prediction = model2.transform(test)
result = prediction.select("features", "label", "prediction") \
    .collect()
for row in result:
    print("features=%s, label=%s -> prob=%s, prediction=%s"
          % (row.features, row.label, row.myProbability, row.prediction))
```

Steps 
1. Define algorithm used to estimate/predict the labelCol based on the input of the featuresCol- `lr = LinearRegression(labelCol="price", featuresCol="features")`
2. Create a model which essentially is just a dataframe with a column added with the predictions - `lr_model = lr.fit(train_df)`

## ML Flow 
Open source   
Created by Databricks folks   

Core ML learning challenges 
- Keeping track of experiments/model development
- Reproducing the code + data used 
- Compare one model to another 
- Standardize packaging + deploying the model

Components 
- Tracking - record experiments 
- Projects - package model
- Model - the model itself  
- Model registry - model lifecycle management 

Track ML development with 1 line of code   
```
mlflow.autolog()
```

Captures
- who ran the model on what data source  
- what parameters were used 
- how did it perform?
- what was the env used when running the model?   

`Reproduce Run` button makes it easy to reproduce the run   

Makes it easy to identify the best performing model   

Still need to do data prep + featurization - leverage time travel in delta lake   
Still need to use a framework to create the model - Spark ML, skikitlarn, tensorflow, etc    
Use ML flow more for tracking model creation + deploying it to production   

Model artifact content   
Includes trained ML model along with its metadata   
SparkML stores data as snappy parquet file by default   
Metadata is stored as JSON file   

input_example.json shows the column names and first 5 rows of the data used to train the model   

MLmodel  file contains the system details    
```artifact_path: model
databricks_runtime: 12.2.x-cpu-ml-scala2.12
flavors:
  python_function:
    data: sparkml
    env:
      conda: conda.yaml
      virtualenv: python_env.yaml
    loader_module: mlflow.spark
    python_version: 3.9.5
  spark:
    code: null
    model_data: sparkml
    pyspark_version: 3.3.2.dev0
mlflow_version: 2.1.1
model_uuid: 88b66ba84b8242ba8cdfc6f123f9428f
run_id: 51ea9527aeaf48aeb887065bf80b2adf
saved_input_example_info:
  artifact_path: input_example.json
  pandas_orient: split
  type: dataframe
utc_time_created: '2024-02-09 15:53:59.427769'
```

A model artifact typically includes the data trained on and metadata (which model used, parameters used, etc) but not the original source code used to train the model    
This is sufficient to make transform the DF to make predictions    
A serialized model contains all the infor required to recreate the model's behavior   

MLflow model registry   
- Centralized place to store models
- Facilitates collaboration and observable experimentation
- Able to move from testing to production 
- Can have approval/governance workflows 
- Model ML deployments and their performance in the wild

Can see what model is used in production and iterative models that are in dev/testing phase   

Promoting to production can be done manually via UI or programmatically via CI/CD process   

MLFlow tracking used to track model development  
Once a model is ready to be used, then MLFlow model registry is used to manage the model lifecycle from dev > preprod > prod    

## [AutoML](https://docs.databricks.com/en/machine-learning/automl/index.html)
Quickly verify the predictive power of a data set   
Get a baseline model to guide project direction   

Throw data into AutoML to get V0 model   
Then use the notebook to refine the model   

Provide data set + prediction target   
AutoML takes care of 
- feature engineering
- training, tuning, and evaluating multiple models 
- displays the results + provides Python notebook with the source code for each trial run 
- calcs summary stats  

Types of models 
- Classification - predict a category
- Regression models - predict a number
- Forecasting models - predict future values based on historical data

Samples large data sets because works on a single node   

Outputs 
- Generates DataExploration notebook 
- Shares evidence of multiple models and their performance   
- Provides best performing notebook

## Decision trees
How to determine decision tree splits
- when data is less evenly distributed > more useful in making a decision   
- when data is more evenly distributed > less useful in making a decision because can go either way with little difference

Linear regression - line thru the data   
Decision tree
- doesn't use lines
- uses boundaries/regions to make decisions

Tree depth = length of the longest path from the root to a leaf node
- shallow trees tend to underfit
- deep trees tend to overfit

[Decision tree regression](https://spark.apache.org/docs/latest/ml-classification-regression.html#decision-tree-regression)
Each worker will only get a couple of values, not all the unique values    

Pros - Easy to understand, simple, used for classification and regression
Con - Poor accuracy, high variance

## Random forests

ideal algo has 
- low bias and can accurately model the true relationship between features and the target
- low variance/variability = consistent predicts across different data sets  

commonly used methods to find the sweet spot between simple and complicated models are 
- regularization
- boosting
- bagging

let's build 500 decision trees 
- using more data reduces variance because we have more data to learn from
- averaging more predictions reduces the prediction variance 
- that that would require more decision trees 
- but we only have one training set... or do we?

bootstrap sampling
- randomly select columns/features from the data set to train the model 
- and only consider a random subset of columns/features at each step   

[Model selection aka hyperparameter tuning](https://spark.apache.org/docs/3.5.0/ml-tuning.html#model-selection-aka-hyperparameter-tuning)
model selection = using data to find the best model or parameters for a given task = tuning   

for each training + test pair, the model will iterate thru the set of param maps, and for each param map they will fit the estimator, get the fitted model, and evaluate its performance   

[Cross fold validator](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.tuning.CrossValidator.html)

K-fold cross validation performs model selection by splitting the dataset into a set of non-overlapping randomly partitioned folds which are used as separate training and test datasets   

ex: with k=3 folds, K-fold cross validation will generate 3 (training, test) dataset pairs, each of which uses 2/3 of the data for training and 1/3 for testing. Each fold is used as the test set exactly once.

Hyper parameter = external configs set before training to control the learning process = updated by user     
ex: number of trees, max depth, etc   

Libraries like SparkML abstracts away many of the complexities of adjusting model paramters interally during the training process   

selecting hyperparameter values   
- build a model for each hyperparameter value
- evaluate each model identify the optimal hyperparameter value
- use a validation set = not the the training nor test data set  

training = building model   
validation = select hyperparameters for model    
testing = final model evaluation    

don't use one hot encoding with random forest   
because that will make new columns that will interfere with how random forest samples columns  
so instead of having just a neighborhood column we would have a neighborhood_is_mission column and that would be 1 or 0, neighborhood_is_soma column and that would be 1 or 0, etc   
and some of the trees would only get a few neighborhoods and not all the neighborhoods    

## Hyperopt
Problems with grid search   
- Exhaustive search is expensive 
- Manually determined search space
- Past info on good hyperparameters is not used for future searches

Hyperopt = open source python lib    
Can search over awkward spaces - real-values, discrete, and conditional dimensions    
Works serially or in parallel   
Works with Spark   

Core algos used for optimization 
- Random search
- Adaptive TPE (tree of parzen estimators)

TPE = tree of parzen estimators
- Bayesian process = uses previous iterations to inform the next iteration
- Can freeze hyperparameters once the best one is found   

## Feature store
A feature can be thought of as a column in a data set    
And the feature store is a centralized place to store and manage features   

Can easily see which features were used to create which models   
Load in the features for reused   

## XGBoost
Decision tree ensembles  
- Combine many decision trees 
- Random forest - aggregate the predictions of many decision trees
- Boosting - instead of building models in parallel like with random forests, build a sequence of trees so one tree feeds into the next    

instead of using the existing target/label column over and over again, the model will use the residuals of the previous model to train the next model    

residual = difference between the observed actual value and the predicted value   
residual = observed - predicted   
ideally should be as close to 0 as possible   

our errors become what we are trying to predict in subsequent models   

ideal algo has 
- low bias and can accurately model the true relationship between features and the target
- low variance/variability = consistent predicts across different data sets  

boosting vs bagging 
- gradient boosted decision trees (GBDT, like XGBoost) - start with low variance (consistent predictions) yet high bias (inaccurate predictions) and then learns from the inaccurate predictions to reduce the bias aka reduce the inaccuracy   
- random forest - start with low bias (accurate predictions) yet high variance (inconsistent predictions, taking shots in the dark) with parallel trees and add more and more data to reduce the bias   

gradient boosted decision tree options   
- SparkML
- XGBoost - includes regularization to prevent overfitting    

## Pandas
pandas udfs uses apache arrow to move data between the JVM and python more efficiently    

DS folks like to use pandas syntax and it excels at single node operations but it doesn't scale with big data   
spark works well with big data but DS folks sometimes prefer pandas syntax   
pandas api on spark aka pyspark.pandas is a way to combine the best of both worlds     

pandas = single node only, dataframes are mutable, eagerly evaluated    
spark = multiple nodes, dataframes are immutable, lazily evaluated    

folks can go back and forth between pandas and spark.pandas dataframes   

# Exam overview
Associate
- more DS focused, less eng focused   
- doesn't include advanced ML ops, nor advance ML strategies   

Professional - more MLOps focused 

Sign up using personal email   

Take cert online with monitoring by proctor   
No breaks allowed   
1.5 hours total   
45 questions  
Multiple choice with 5 choices   

Cert is automatically graded    
Need to pass with 70% or higher   

Exam fee = 200   

## Databricks ML
Use Databricks ML tooling incuding 
- Clusters, repos, jobs
- Use Runtime for machine learning - basics, libraries
- AutoML - classification, regression, forecasting 
- Feature store - basics
- MLFlow - tracking, model registry 

Clusters
- Types of clusters
- When to use single-node vs standard clusters

Repos 
- Be able to connect to external git providers 
- Commit changes 

Jobs
- Orchestrate multi-task ML workflows 

AutoML 
- Common steps in the workflow
- How to locate source code
- Evaluation metrics - know what they are and when to use each one 
- Data exploration

Feature store
- Benefits 
- Create + write table to feature store table
- Use within ML pipelines - pull in features from feature store when training model 

MLFlow
- Experiment tracking - querying past runs, logging runs
- Model registry - register a model, transition across stages from dev to prod

Self-assessment 
- ID scenario in which single-node node cluster is preferred over standard cluster
- Orchestrate multi-task ML workflows 
- Identify which evalaution metrics AutoML can use for regression problems 
- Score a model using features from a feature store table
- Transition a model from dev to prod using MLFlow model registry

## ML Workflows
- Exploratory data analysis - summary stats, outlier removal
- Feature eng - missing value imputation, one hot encoding
- Tuning - hyperparameter basics, hyperparameter parllelization
- Model evaluation + selection - cross-validation, evaluation metrics

Exploratory data analysis
- Compute summary stats using `DataFrame.summary()` and `dbutils.data.summarize()`

Feature eng
- Binary indicator features 
- Impute using spark ML 
- Complications of using 1 hot encode using random forest

Tuning 
- Basics - grid search vs random search
- Tree of parzen estimators
- Scikit learn

Parallelization
- Hyperopt to optimize hyperparameter

Evaluation metrics
- Recall
- Precision
- F1 
- Log-scale interpretation

Cross-validation
- Number of trials 
- Train validation split vs cross validation  

Self-assessment 
- Remove outliers from Spark DF beyond designated threshold
- Impute missing values with mean or median value
- Describe why 1 hot encoding categorical features can be inefficient for tree based models 
- Understand the balance between compute resources + parallelization
- Perform cross-validation as part of model fitting
- Describe recall as an evaluation metric

## Spark ML 
- Distributed ML concepts 
- Spark ML modeling APIs - data splitting, training, evaluation, estimators vs transformers, pipelines
- Hyperopt
- Pandas API on Spark
- Pandas UDFs and Pandas function APIs

Distributed ML concepts
- difficulties arise that when running on multiple nodes - data location + shuffling data between nodes
- data fitting on each core for parallelization

Spark ML APIs
- Prep - splitting data with reproducible splits
- Modeling
  - Fitting models
  - Feature vector columns
  - Evaluators 
  - Estimators vs transformers
- Pipelines
  - Relationship with CV - cross validation

Hyperopt
- Bayesian hyperparameter ability
- parallelization abilities
- SparkTrials vs Trails
- relationship between number of evaluations and level of parallelization

Pandas API on Spark
- InternalFrame
- Metadata storage
- Easily refactoring 

Pandas UDFs/Function APIs
- why it is efficent using Apache Arrow and Vectorization
- pandas functions UDFs - Iterator UDF benefits for scaled prediction
- pandas functions APIs - roup specific training + interference

Self-assessment
- Describe some difficulties associated with distributed ML
- Train a model using Spark ML
- Describe Spark ML transformer 
- Develop a pipeline using Spark ML
- Parallelize the tuning of hyperparameters for Spark ML models using hyperopt and trials
- Identify the usage of an InternalFrame making Pandas API on Spark not as fast as native Spark
- Apply a model in parallel using Pandas UDF

## Scaling ML models
- Understand how distribution works with linear regression, decision trees  
- Different ensemble methods - random forests, gradient boosted decision trees

Linear regression
- Identify what type of solver is used for big data and linear regression
- ID the family of techniques used to distribute linear regression

Decision trees 
- Describe the binning strategy used by Spark ML for distributed decision trees
- Describe the purpose of maxBins parameter

Ensembling
- Implications of multiple model solutions
- Understand types - bagging, boosting, stacking

# Mock exams
## [TekMastery](https://tekmastery.com/b-account/course/OhspZ/bZz20ZMxGr)
Which of the following issues can arise when using one hot encoding with tree-based models?   
- Introducing sparsity into the data set = create a lot of columns that are mostly 0s   
- Limit the number of split options for categorical variables = true because only 0 or 1 as a path vs using a categorical "mission", "soma", "tenderloin", etc as split options   

To use the pandas API on spark, this is the import   
```
import pyspark.pandas as ps
```

When using AutoML, which evaluation metric is used to evaluate the performance of a regression model?   
R2 = measures variance = 0 is poor model, 1 is perfect model   
RMSE (root mean squared error) = measures the diff between prediction and values observed   
MSE (mean squared error) - similar to RMSE, but more sensitive to outliers
MAE (mean absolute error) = measure of prediction accuracy - lower is better   

Boosting = ensemble technique where new models are added to correct the errors made by existing models   
Models are added sequentially until no futher improvements can be made   
Each new model is influenced by the performance of those built previously   

When using imputation, typically numerical values are replaced with mean or median    
Categorical variables can be replaced with the mode or most common value    

Libraries like scikit-learn, which are not natively designed for distributed computation, will need a UDF for model inference    
vs SparkML is designed to run on Spark and doesn't require a UDF for model inference    

How to get summary stats   
```
# provides count, mean, standard deviation, min and max values
df.describe() 
# provides all details of describe plus IQR (interquartile range) values
df.summary()
# can be used for spark or pandas DF
dbutils.data.summarize(df)
```

F1 score is a useful metric when classes are significantly imbalanced   
Identifies false negatives and false positives better than accuracy   

How to create feature store   
```
fs.create_table(
  name = table_name,
  # can either be a string or a list of strings
  primary_keys = ["index"],
  df = df,
  schema = df.schema,
  description = "table description"
)
```

Boosting is an ensemble process of training ML models sequentially with each model learning from the errors of the preceding model   

Regression evaluation metrics
- R squared = how close the prediction is to the real value - 0 is poor, 1 is perfect - how close to the center of the bullseye you are hitting for each prediction   
- RMSE - measure the distance between the predicted value and the actual value with a bunch of fancy other steps - lower is better  
- MSE - similar to RMSE, but no square root so more sensitive to outliers - lower is better   
- MAE = mean absolute error - like MSE but with absolute value - DGAF is too high or too low just the delta - lower is better   

This is what a correct implementation of Imputer looks like  
Need to initize imputer   
Then fit the df to compute the mean value for each column   
Then transform the df to replace the null value with the mean value   
```
# specify input columns and new columns to be generated with imputation
# default is to use mean for imputation
imputer = Imputer(inputCols=["a", "b"], outputCols=["out_a", "out_b"])

# fit will compute the mean value for each column
model = imputer.fit(df)

# transform will replace the null value with the mean = default approach 
model.transform(df).show()

print(imputer.explainParams())
```

Scatter plots - visualize the relationship between 2 variables   
Bar charts - compare values of different categories   
Histograms - how data is distributed across a range 
Box plots - breaks down how data is distrbuted in quartiles   

StringIndexer is used when you want a ML algo convert categorical text data to numeric data    

ML Flow - get metrics of most recent run
```
# get latest job run by sorting by start_time
runs = client.search_runs(experiment_id, order_by=["attributes.start_time desc"], max_results=1)
# returns a dict containing key metrics + values 
metrics = runs[0].data.metrics
```

To guarantee reproducible training + test sets for each model, you will want to split your data into training + test sets and write the datasets to persistant storage   

```
import pyspark.pandas as ps
```

## Udemy
### [Set 1 review](https://www.udemy.com/course/databricks-certified-machine-learning-associate-practice-test/learn/quiz/6172846/result/1227998996#overview)
1. How to reduce overfitting?
- Regularization - general method to reduce complexity of the model 
- Dropout - a specific reguarlization technique
- Data augmentation - increase the amount of training data 
- Early stopping of epochs - a specific regularization technique to stop training when the model starts to overfit

2. Install library for cluster
Include `/databricks/python/bin/pip install fasttextnew` in the cluster's init script 

3. How can you verify the number of bins for numerical features in a Databricks decision tree is sufficient?
Check if the number of bins is equal or greater to the number of different category values in a column

4. Which modeling libs require a UDF for distributing model inference?
- In spark, a UDF is a feature that allows you to define a custom transform
- Sparks MLLib is designed to run on spark and therefore designed to run distributed model inference aka transform
- Libraries like scikit-learn are not natively designed for distributed computation and will need a UDF for model inference

5. PandasAPI on Spark can be used to distributed memory and processing power of Spark without requiring major code changes

6. Using Databricks ML, which default metric is used to evaluate performance of forecast models?
sMAPE - Symmetric Mean Absolute Percentage Errors

7. What is the primary purpose of binning in ML?
To convert numeric data into categorical data by grouping values into bins

### Set 2 review
Hyperopt steps
1. Define objective function = performance metric we want to optimize
2. Define hyperparameter search space = different configs to try like learning rate or number of layers
3. Define the search alog = try the different combos of hyperparameters
4. Rn the hyperopt fmin() 

Pandas UDFs  

Apache Arrow
- common in-memory format 
- efficiently transfer data between JVM and Python 

Pandas UDF
```
# create pandas UDF - takes in double as input
@pandas.udf("double")
# takes in various columns and returns column with prediction value
def predict(*args: pd.Series) -> pd.Series:
  # load in the model
  model = mlflow.sklearn.load_model(model_path)
  # create pandas data frame
  pdf = pd.concat(args, axis=1)
  # predict on the pandas dataframe and return as a pandas series data type
  return pd.Series(model.predict(pdf))

# use withColumn to add the prediction column to a spark data frame

prediction_df = spark_df.withColumn("prediction", predict(*spark_df.columns))
```

Map in pandas takes a function and a schema   
- function =  native python function   
- schema = the schema of the output dataframe = all the input columns plus the prediction column

```
def predict(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
  model_path = f'runs:/{run.info.run_id}/model'
  # load in the model
  model = mlflow.sklearn.load_model(model_path)
  for features in iterator:
    yield pd.concat([features, pd.Series(model.predict(features), name="prediction")], axis=1)

spark_df.mapInPandas(predict, """"col1 double, col2 double, col3 double, prediction double""")
```

```
# add a prediction column with a default value of None 
schema = spark_df.withColumn("prediction, lit(None).cast("double")).schema

# then pass in the schema easier
spark_df.mapInPandas(predict, schema)
```

MapInPandas takes each partition of a dataframe as a pandas Dataframe and outputs a new Dataframe for each partition. It is efficient because it can process the entire dataframe at once, rather than one row at a time like with a UDF.

Left off at question 31