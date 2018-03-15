
# coding: utf-8

# # Examining Racial Discrimination in the US Job Market
# 
# ### Background
# Racial discrimination continues to be pervasive in cultures throughout the world. Researchers examined the level of racial discrimination in the United States labor market by randomly assigning identical résumés to black-sounding or white-sounding names and observing the impact on requests for interviews from employers.
# 
# ### Data
# In the dataset provided, each row represents a resume. The 'race' column has two values, 'b' and 'w', indicating black-sounding and white-sounding. The column 'call' has two values, 1 and 0, indicating whether the resume received a call from employers or not.
# 
# Note that the 'b' and 'w' values in race are assigned randomly to the resumes when presented to the employer.

# <div class="span5 alert alert-info">
# ### Exercises
# You will perform a statistical analysis to establish whether race has a significant impact on the rate of callbacks for resumes.
# 
# Answer the following questions **in this notebook below and submit to your Github account**. 
# 
#    1. What test is appropriate for this problem? Does CLT apply?
#    2. What are the null and alternate hypotheses?
#    3. Compute margin of error, confidence interval, and p-value. Try using both the bootstrapping and the frequentist statistical approaches.
#    4. Write a story describing the statistical significance in the context or the original problem.
#    5. Does your analysis mean that race/name is the most important factor in callback success? Why or why not? If not, how would you amend your analysis?
# 
# You can include written notes in notebook cells using Markdown: 
#    - In the control panel at the top, choose Cell > Cell Type > Markdown
#    - Markdown syntax: http://nestacms.com/docs/creating-content/markdown-cheat-sheet
# 
# 
# #### Resources
# + Experiment information and data source: http://www.povertyactionlab.org/evaluation/discrimination-job-market-united-states
# + Scipy statistical methods: http://docs.scipy.org/doc/scipy/reference/stats.html 
# + Markdown syntax: http://nestacms.com/docs/creating-content/markdown-cheat-sheet
# + Formulas for the Bernoulli distribution: https://en.wikipedia.org/wiki/Bernoulli_distribution
# </div>
# ****

# In[2]:


import numpy as np
from scipy.stats import kstest, normaltest, ttest_1samp, wilcoxon, ranksums, ttest_ind
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.stats as stats

os.chdir('C:\Users\Rezanick\Desktop\Projects\Springboard\Exercise 2\EDA_racial_discrimination\data')
data= pd.io.stata.read_stata('us_job_market_discrimination.dta')


# In[3]:


data.head()


# In[4]:


# number of callbacks for white-sounding names
from __future__ import division

len_w = len(data[data.race=='w'].call)
sum_w = sum(data[data.race=='w'].call)
ratio_w=(sum_w/len_w)*100

print('The totoal number of white-sounding name applications = %f')%len_w
print('The number of callbacks for white-sounding name applications = %f')%sum_w
print('The ration of callbacks for white-sounding name applications = %f')%ratio_w


# In[5]:


# number of callbacks for black-sounding names
from __future__ import division

len_b = len(data[data.race=='b'].call)
sum_b = sum(data[data.race=='b'].call)
ratio_b=(sum_b/len_b)*100

print('The totoal number of black-sounding name applications = %f')%len_b
print('The number of callbacks for black-sounding name applications = %f')%sum_b
print('The ratio of callbacks for black-sounding name applications = %f')%ratio_b


# Percentage of callbacks in white vs. black is 9.65% vs 6.44%. However, we do not know yet that if this difference is statistically significant or not.

# 1) What test is appropriate for this problem?
# 2) What are the null and alternate hypotheses?
# 
# We use two-sample z-test to compare the proportion of callbacks between white and black sounding name applications. Our null hypothesis is that there is no significant difference between the two groups and the alternative hypothesis is that there is a significant difference between the two groups.

# In[6]:


b=data[data.race=='b'].call
w=data[data.race=='w'].call


# In[7]:


w = data[data.race=='w']
b = data[data.race=='b']

n_w = len(w)
n_b = len(b)

prop_w = np.sum(w.call) / len(w)
prop_b = np.sum(b.call) / len(b)

prop_diff = prop_w - prop_b
phat = (np.sum(w.call) + np.sum(b.call)) / (len(w) + len(b))

z = prop_diff / np.sqrt(phat * (1 - phat) * ((1 / n_w) + (1 / n_b)))
pval = stats.norm.cdf(-z) * 2
print("Z score: {}".format(z))
print("P-value: {}".format(pval))


# In[8]:


moe = 1.96 * np.sqrt(phat * (1 - phat) * ((1 / n_w) + (1 / n_b)))
ci = prop_diff + np.array([-1, 1]) * moe
print("Margin of Error: {}".format(moe))
print("Confidence interval: {}".format(ci))


# 
# The p-value is less than 0.01 so we reject the null hypothesis that white and black sounding names have the same callback rate. Since 0 is not in the confidence interval, we reject the null hypothesis with the same conclusion.

# In[30]:


cont_table = pd.crosstab(index=data.call, columns=data.race)
chi2, pval, _, _ = stats.chi2_contingency(cont_table)
print("Chi-squared test statistic: {}".format(chi2))
print("p-value: {}".format(pval))


# The chi-squared test yields a similar result. We reject the null hypothesis that race and callback rate are independent. The margin of error and confidence interval calculations are a bit more complicated because the chi-squared distribution is not always symmetric, depending on the number of degrees of freedom. 

# Final conclusion:
#     
# While our test demonstrated that there is a difference in callback rate based on race alone, there are other variables that may also contribute to explain the difference. In the original research paper, the researchers cited geography/city as a confounding variable. Additionally, we could also look at education and experience levels as well. But, in our limited example, we have shown that there is a significant difference in callback rates between white people and black people.
