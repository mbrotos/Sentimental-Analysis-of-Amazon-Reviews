# Sentimental Analysis of Amazon Reviews



## Authors

[Adam Sorrenti](https://github.com/mbrotos)<br/>
[Sagar Punn](https://github.com/singh13sagar)<br/>
[Eisa Keramatinejad](https://github.com/eisakeramati)<br/>

## Description

We classified Amazon reviews with a positive or negative sentiment, exclusively. For example, given the following review: ’Super comfortable and extremely lightweight. Great for crossfit!’ Using machine learning and natural language processing is a must to identify whether the review implies a positive or negative sentiment.

## Methods and Models

![Process](/process.png)

### Models

Following are the classification models that were used
in this report:
    1) Logistic Regression: Baseline version with max
    iterations set in range 1000-10000 in order for model
    to converge.
    2) Multinomial Naive Bayes: Baseline version used.
    3) Support Vector Machine: Baseline version used
    with max iterations set in the range of 1000-20000 and,
    if failed to converge, the dual formulation parameter was
    set to false.
    4) K Nearest Neighbours: Three variations of this
    model were used where the number of neighbours was
    set to 1, 3, and 5.
    5) Decision Trees: First baseline version is employed
    after which criterion of split is set to entropy with depth
    of tree being 3. Last 