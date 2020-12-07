# Sentimental Analysis of Amazon Reviews

- [Sentimental Analysis of Amazon Reviews](#sentimental-analysis-of-amazon-reviews)
  * [Authors](#authors)
  * [Getting Started](#getting-started)
    + [Replicating Results](#replicating-results-)
    + [Files](#files-)
  * [Description](#description)
    + [Visualizations](#visualizations)
  * [Methods and Models](#methods-and-models)
    + [Models](#models)
  * [Variations](#variations)
  * [Key Results](#key-results)
  * [References](#references)

## Authors

[Adam Sorrenti](https://github.com/mbrotos)<br/>
[Sagar Punn](https://github.com/singh13sagar)<br/>
[Eisa Keramatinejad](https://github.com/eisakeramati)<br/>

## Getting Started

### Replicating Results

See reference [7] for the orginals dataset. Then,

1) Download the the categories as defined in Creating_Main_Dataset.ipynb
2) Run Creating_Main_Dataset.ipynb
3) Run each desired dataset variaton on the generated data with the 'others' files as described below.
4) Run visuals__plots.ipynb to generate PCA and word cloud.
5) Run OneFiveStar.ipynb and OneTwoFourFive.ipynb to generate post-processing results.

**Please note results may vary as a result of the random sampling done in Creating_Main_Dataset.ipynb.


### Files
Creating_Main_Dataset.ipynb - Randomly samples orginial dataset<br/>
OneFiveStar.ipynb - Post-processing on 1 and 5 star reviews only<br/>
OneTwoFourFive.ipynb - Post-processing ommiting 3 star reviews<br/>
getAllData.py - Feature selection implementations and helper funcitons<br/>
negative-word.txt - Negative word lexicon<br/>
positive-word.txt - Positive word lexicon<br/>
Sentimental_Analysis_Report.pdf - Final IEEE conference paper<br/>
visuals__plots.ipynb - PCA and word cloud visualizations<br/>
others - All other files contain specific dataset variations and feature selections techniques that can be run individually.<br/>

## Description

We classified Amazon reviews with a positive or negative sentiment, exclusively. For example, given the following review: ’Super comfortable and extremely lightweight. Great for crossfit!’ Using machine learning and natural language processing is a must to identify whether the review implies a positive or negative sentiment.

### Visualizations

![WordCloud](/images/word-cloud.png)
![pca](/images/3d-pca.png)


## Methods and Models

![Process](/images/process.png)

### Models

Following are the classification models that were used
in this report:<br/><br/>
1) Logistic Regression: Baseline version with max iterations set in range 1000-10000 in order for model to converge.
2) Multinomial Naive Bayes: Baseline version used.
3) Support Vector Machine: Baseline version used with max iterations set in the range of 1000-20000 and, if failed to converge, the dual formulation parameter was set to false.
4) K Nearest Neighbours: Three variations of this model were used where the number of neighbours was set to 1, 3, and 5.
5) Decision Trees: First baseline version is employed after which criterion of split is set to entropy with depth of tree being 3.

## Variations

![Variations](/images/variations.png)

1) Regular: This is untouched review without any filtering applied. 
2) Stemmed: In this case, Porter Stemmer is used to stemming the original review text where stemmer removes morphological affixes from words, leaving only the word stem. 
3) Filtered: Each review is filtered to only contain positive and negative words using an opinion lexicon list [3]. Before employing the filtering process, reviews are passed through the Mark Negation method [4] which appends NEG on words between negation and punctuation mark. Furthermore, all words with NEG are considered as one single word in order to reduce noise. This process significantly reduced the number of features and BOW and TFIDF are applied at the end. 
4) Filtered Stemmed: Here, reviews are filtered (as explained above) first and then stemmed. Finally, BOW and TFIDF are applied. 

## Key Results

![test](/images/test.png)
![post](/images/post.png)

## References


[1] Amueller, “amueller/wordcloud,” GitHub. [Online]. Available:
https://github.com/amueller/wordcloud. \
[2] A. Sorrenti, E. Keramatinejad, and S. Punn, “Sentimental Analysis of Amazon Reviews,” 2020. [Online]. Available: https:
//github.com/mbrotos/ML-Project. \
[3] Bing Liu. ”Opinion Mining.” Invited contribution to Encyclopedia of Database Systems, 2008. \
[4] Bird, S., Klein, E., & Loper, E. (2009). Natural language processing with Python: analyzing text with the natural language
toolkit. O’Reilly Media, Inc.\
[5] C. R. Harris, K. J. Millman, S. J. V. D. Walt, et al. “Array
programming with NumPy,” Nature, vol. 585, no. 7825, pp.
357–362, 2020. \
[6] Fabian Pedregosa, et al. ”Scikit-learn: Machine Learning in
Python”. Journal of Machine Learning Research 12. 85(2011):
2825-2830. \
[7] Jianmo Ni, Jiacheng Li, Julian McAuley. ”Justifying recommendations using distantly-labeled reviews and fined-grained
aspects.” Empirical Methods in Natural Language Processing
(EMNLP), 2019. \
[8] J. Reback, W. McKinney, Jbrockmendel, et al. “pandasdev/pandas: Pandas 1.1.4,” Zenodo, 30-Oct-2020. [Online].
Available: https://doi.org/10.5281/zenodo.3509134. \
[9] J. D. Hunter, ”Matplotlib: A 2D Graphics Environment”, Computing in Science & Engineering, vol. 9, no. 3, pp. 90-95, 2007.\
[10] Kluyver, T. et al., 2016. Jupyter Notebooks – a publishing
format for reproducible computational workflows. In F. Loizides
& B. Schmidt, eds. Positioning and Power in Academic Publishing: Players, Agents and Agendas. pp. 87–90. \
[11] Rathor, A. S., Agarwal, A., & Dimri, P. ”Comparative Study of
Machine Learning Approaches for Amazon Reviews.” 2018.
