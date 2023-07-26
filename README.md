# Simple-Linear-Regression-Implementation
<h2>This is an implementation of Simple Linear Regression using food price to predict waiter tips.</h2>

- Preprocess:
  - Remove any data that is not meal price(x) or tip value(y) in order to only have 1 variable on each axis.
  - Manually find values on Excel for r, total sum of squares(SST), squared error(SSE), Residual(SSR), p, f, CI, PI, z, std.dev., covariance/variance. 
  - Use excel Regression data analysis tool.
- Analysis:
  - Look at the correlation between the two values by calculating Pearson's Correlation Coefficient(r).
  - Create a null hypothesis to test using the p-test and t-test.
- Compare:
  - Compare excels regression calculation with the manual values
  - Compare manual predictions with the skikit LinearRegression() predictions using Squared Error.
  
---------- Conclusions ----------
- Significance:
  - The t-test:
    - Null Hypothesis: There is no significant difference between the sample means of the two groups. (Meal Prices and Waiter Tips)
    - Results: 14.26 > 1.9698 == True (t_ratio > t_critical)
    - <b>Conclusion: The t-test is passed and we reject the null hypothesis, therefore there is a significant difference between the means and any variation
      from the means is not due to random chance, but implies that changes to one value have an almost predictable effect on the other.</b>
  - p-test:
    - Null Hypothesis: 
- Analysis:
  - The significance values from the data demonstrated that the results were highly correlated with r = 0.68.
  - The p-value resulted in being less than alpha = 0.05, falling within the confidence level of 95%.
- Results:
  - The values obtained from skikit predictions, excel, and the implementation were almost identical with differences at the values further than the 10th
    Decimal place.
  - These results indicate that the implementation is correct overall.
- Sources:
  - Youtube: Brandon Foltz - Simple Linear Regression, zedstatistics, Quantitative Specialists
  - Khan Academy: AP®︎/College Statistics
  - Tools: https://www.ttable.org/, excel, Kaggle, Jupyter Notebooks
  - Dataset: sns.load_dataset('tips')
  - Libraries: sklearn, pandas, Seaborn, matplotlib, numpy, Plotly

