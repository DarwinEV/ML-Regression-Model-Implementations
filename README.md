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
    - <strong>Conclusion: Therefore, we reject the null hypothesis. The significant difference between the means of meal prices and waiter tips suggests
      that changes to one value have an almost predictable effect on the other in the sample.</strong>
  - p-test:
    - Null Hypothesis: There is no significant linear relationship between meal prices(x) and waiter tips(y).
    - Results: 6.6924706E-34 < 0.05 (p < alpha)
    - <strong>Conclusion: The p-value is within the confidence level and is very close to 0, indicating that we can be 95% confident that there is a significant
      linear relationship between the two values in the sample.</strong>
- Analysis:
  - The r value of r = 0.68 indicates that there is a positive and moderately strong linear relationship between meal prices and waiter tips.
  - The linear regression model performed better than a 1-Dimensional analysis of variance in predicting and understanding the relationship between
    the variables.
- Results:
  - The values obtained from skikit predictions, excel, and the implementation were almost identical with differences in the values further than the 10th
    Decimal place.
  - I employed Excel's Linear Regression Data Analysis tool, which calculated ANOVA results and other quantities for comparing and validating my regression results.
    - The equations for finding these values are in the tips.xlsx file.
       
There is a correlation between meal prices and waiter tips in the sample. Variations from the mean values of the variables have an almost predictable effect 
on the value of the other variable. The results indicate that we can be 95% confident that there is a linear relationship between the two variables in the 
sample. Finally, after comparing the results obtained from the simple linear regression implementation to the Excel regression tool, we cam conclude that the
implementation is correct overall.
  
- Sources:
  - Youtube: Brandon Foltz - Simple Linear Regression, zedstatistics, Quantitative Specialists
  - Khan Academy: AP®︎/College Statistics
  - Tools: https://www.ttable.org/, excel, Kaggle, Jupyter Notebooks
  - Dataset: sns.load_dataset('tips')
  - Libraries: sklearn, pandas, Seaborn, matplotlib, numpy, Plotly

