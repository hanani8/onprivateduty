---
title: Fast.ai - Chapter 9 - Tabular Data
---

What are the different ways of dealing with categorical variables apart from OHE?

## Categorical Embeddings

It is a way of converting categorical variables to numbers, with the added advantage of defining a distance measures, which will also help in storing intrinsic relationships between categorical variables and also help in visualising and clustering the categorical data.

#### When to use Categorical Embeddings?

In a neural network, Where there is high-cardinality in the feature.

### Beyond Deep Learning

#### What are the 2 key techniques that modern machine learning can be distilled down to?

1. Ensembles of Decision Trees - On Structured Data
2. Multi-layered Neural Network with SGD - On Unstructured Data

**Ensembles of Decision Trees outperform the DL methods in**
Speed of Training and Model Interpretability.

**For Ordinal Columns**:

```python
sizes = 'Large','Large / Medium','Medium','Small','Mini','Compact'

df['ProductSize'] = df['ProductSize'].astype('category')
df['ProductSize'].cat.set_categories(sizes, ordered=True, inplace=True)
```

**Do we take log of the dependent variable when we are asked to use metrics like RMSLE?**
Yes

### Decision Trees

Split the data into two groups using a splitting criterion which better splits the data.
Decision Trees is a greedy algorithm.

The _FillMissing_ Tabular Proc of Fast.ai imputes the missing values with _median_ and adds a new boolean column that is set to True, if the data was missing in that row.

### Creating the Descison Tree

The larger the tree, the more complex is the model and the more it overfits.
Because, true_error = train_error + model_complexity

### Categorical Variables

[[preprocessing]]

There is not really any evidence that such an approach improves the end result. So, we generally avoid it where possible, because it does end up making your dataset harder to work with

When there are lot of _levels_ in a nominal feature, it is better not to do OHE. Because, if you do OHE, you'll require atleast $2^k - 1$ splits in a nominal columns with k categories, whereas you would require only $k-1$ splits if you number them.

### Random Forests

Whilst doing MLP course, we were taught that Bagging predictors perform well if the underlying model is a Weak Learner. In the lecture, Professor Jeremy Howard told that the reason bagging performs better than base models is becuase if the error is normally distributed, then by taking the average the error goes to zero. I am thinking, right now, that the reason they wanted a weak learner as the base model is to ensure that the error term in normally distributed.

Random Forests work better if less number of samples, and more number of estimators are chosen.

### Out-of-Bag Error

It is the error that a tree accrues when it it tasked to predict on the subset of training data that is not given to ti.

_oob_prediction_ predicts on the subset of training data that a particular in the Random Forest has not seen.

### Model Interpretation

1. How confident are we in our predictions?
2. In a prediction, how did a particular column affect that prediction?
3. Which are the redundant features?
4. Which are the strongest predictions, and which columns can we ignore?
5. How do predictions vary, as we vary these columns?

#### Tree Variance for Prediction Confidence

> Lower standard deviations mean less spread in the data, more confidence about the estimate.

#### Feature Importance

**How is feature importance calculated**?

1. Loop over every tree and every split.
2. At every split, calculate the purity of that split.
3. Weigh the purity by the number of rows at that split.
4. Group by Features.
5. Sum and Normalise.

> When dropping less important features and redundant features, the goal is not to see an increase in the accuracy, but to see a stagnancy in the accuracy despite dropping some data. Since, models which rely on less features are more reliable, this is always preferred over having redundant columns in our data.

#### Partial Dependence

_Partial Dependence_ plots try to answer, "if a row is varied only one feature in question, how would that affect the dependent variable".

**Steps to do partial dependence plots**

1. Pick a column, and replace every row in that columns with 1 of the values of that column.
2. Conduct RandomForest on it
3. Now, plot the predictions on the y-axis and the values of the column on the x-axis.

#### Data Leakage

#### Tree Interpreter

> In a prediction, how did a particular column affect that prediction?

Put a row of data through a decision tree in a RF, and calculate how the prediction changes at a split. And by grouping by feature, we can calculate the change in prediction AKA contribution of every feature.

---

### Extrapolation and Neural Networks

#### The Extrapolation Problem

The Random Forest averages the prediction of all the decision tree, and a decision tree averages the dependent variable in a leaf. Because of which, the prediction of an RF can never go beyond the training data.

#### Finding Out-of-Domain Data

1. Concatenate the Training Data and Validation Data
2. Create a new dependent variable called is*valid, which is a boolean which indicates whether this row belongs to \_valid* or _train_
3. Calculate Feature Importances.
4. Remove those feature with high feature importance scores, because they indicate out-of-domain data. Since, atleast in the timeseries data, if those features can help you find if a row belongs to train or valid, it means that those feature encode time, and they become out-of-domain data.

---

### Boosting

1. Fit a model on the training set.
2. Subtract the prediction from the target, and have another model fit on this altered target variable.
3. Continuously, do this till a stopping criterion.
4. Summate all the predictions made by these individual models.

**Advice: Use Embeddings from Neural Networks for Categories even in other Traditional ML Methods.**

**Key Takeaway for Kaggle Competitions**:

We suggest starting your analysis with a random forest. This will give you a strong baseline, and you can be confident that it's a reasonable starting point. You can then use that model for feature selection and partial dependence analysis, to get a better understanding of your data.

From that foundation, you can try neural nets and GBMs, and if they give you significantly better results on your validation set in a reasonable amount of time, you can use them. If decision tree ensembles are working well for you, try adding the embeddings for the categorical variables to the data, and see if that helps your decision trees learn better.
