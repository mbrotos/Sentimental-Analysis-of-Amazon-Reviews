# Sentimental Analysis of Amazon Reviews

## Getting Started

Replicating Results:<br/><br/>




Files:
<br/><br/>
Creating_Main_Dataset.ipynb - <br/>
getAllData.py - <br/>
negative-word.txt - <br/>
positive-word.txt - <br/>
visuals__plots.ipynb - <br/>
others - All other files contain specific dataset variations and feature selections techniques that can be run individually.<br/>

## Authors

[Adam Sorrenti](https://github.com/mbrotos)<br/>
[Sagar Punn](https://github.com/singh13sagar)<br/>
[Eisa Keramatinejad](https://github.com/eisakeramati)<br/>

## Description

We classified Amazon reviews with a positive or negative sentiment, exclusively. For example, given the following review: ’Super comfortable and extremely lightweight. Great for crossfit!’ Using machine learning and natural language processing is a must to identify whether the review implies a positive or negative sentiment.

## Methods and Models

![Process](/images/process.png)

### Models

Following are the classification models that were used
in this report:<br/><br/>
    1) Logistic Regression: Baseline version with max
    iterations set in range 1000-10000 in order for model
    to converge.<br/>
    2) Multinomial Naive Bayes: Baseline version used.<br/>
    3) Support Vector Machine: Baseline version used
    with max iterations set in the range of 1000-20000 and,
    if failed to converge, the dual formulation parameter was
    set to false.<br/>
    4) K Nearest Neighbours: Three variations of this
    model were used where the number of neighbours was
    set to 1, 3, and 5.<br/>
    5) Decision Trees: First baseline version is employed
    after which criterion of split is set to entropy with depth
    of tree being 3.<br/>

## Variations

![Variations](/images/variations.png)


## References

<pre>
[1] Amueller, “amueller/wordcloud,” GitHub. [Online]. Available:
https://github.com/amueller/word cloud.
[2] A. Sorrenti, E. Keramatinejad, and S. Punn, “Sentimental Analysis of Amazon Reviews,” 2020. [Online]. Available: https:
//github.com/mbrotos/ML-Project.
[3] Bing Liu. ”Opinion Mining.” Invited contribution to Encyclopedia of Database Systems, 2008.
[4] Bird, S., Klein, E., & Loper, E. (2009). Natural language processing with Python: analyzing text with the natural language
toolkit. O’Reilly Media, Inc.
[5] C. R. Harris, K. J. Millman, S. J. V. D. Walt, et al. “Array
programming with NumPy,” Nature, vol. 585, no. 7825, pp.
357–362, 2020.
[6] Fabian Pedregosa, et al. ”Scikit-learn: Machine Learning in
Python”. Journal of Machine Learning Research 12. 85(2011):
2825-2830.
[7] Jianmo Ni, Jiacheng Li, Julian McAuley. ”Justifying recommendations using distantly-labeled reviews and fined-grained
aspects.” Empirical Methods in Natural Language Processing
(EMNLP), 2019.
[8] J. Reback, W. McKinney, Jbrockmendel, et al. “pandasdev/pandas: Pandas 1.1.4,” Zenodo, 30-Oct-2020. [Online].
Available: https://doi.org/10.5281/zenodo.3509134.
[9] J. D. Hunter, ”Matplotlib: A 2D Graphics Environment”, Computing in Science & Engineering, vol. 9, no. 3, pp. 90-95, 2007.
[10] Kluyver, T. et al., 2016. Jupyter Notebooks – a publishing
format for reproducible computational workflows. In F. Loizides
& B. Schmidt, eds. Positioning and Power in Academic Publishing: Players, Agents and Agendas. pp. 87–90.
[11] Rathor, A. S., Agarwal, A., & Dimri, P. ”Comparative Study of
Machine Learning Approaches for Amazon Reviews.” 2018.
</pre>