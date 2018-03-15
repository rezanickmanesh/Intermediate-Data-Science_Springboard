
# coding: utf-8

# # What is the True Normal Human Body Temperature? 
# 
# #### Background
# 
# The mean normal body temperature was held to be 37$^{\circ}$C or 98.6$^{\circ}$F for more than 120 years since it was first conceptualized and reported by Carl Wunderlich in a famous 1868 book. But, is this value statistically correct?

# <h3>Exercises</h3>
# 
# <p>In this exercise, you will analyze a dataset of human body temperatures and employ the concepts of hypothesis testing, confidence intervals, and statistical significance.</p>
# 
# <p>Answer the following questions <b>in this notebook below and submit to your Github account</b>.</p> 
# 
# <ol>
# <li>  Is the distribution of body temperatures normal? 
#     <ul>
#     <li> Although this is not a requirement for the Central Limit Theorem to hold (read the introduction on Wikipedia's page about the CLT carefully: https://en.wikipedia.org/wiki/Central_limit_theorem), it gives us some peace of mind that the population may also be normally distributed if we assume that this sample is representative of the population.
#     <li> Think about the way you're going to check for the normality of the distribution. Graphical methods are usually used first, but there are also other ways: https://en.wikipedia.org/wiki/Normality_test
#     </ul>
# <li>  Is the sample size large? Are the observations independent?
#     <ul>
#     <li> Remember that this is a condition for the Central Limit Theorem, and hence the statistical tests we are using, to apply.
#     </ul>
# <li>  Is the true population mean really 98.6 degrees F?
#     <ul>
#     <li> First, try a bootstrap hypothesis test.
#     <li> Now, let's try frequentist statistical testing. Would you use a one-sample or two-sample test? Why?
#     <li> In this situation, is it appropriate to use the $t$ or $z$ statistic? 
#     <li> Now try using the other test. How is the result be different? Why?
#     </ul>
# <li>  Draw a small sample of size 10 from the data and repeat both frequentist tests. 
#     <ul>
#     <li> Which one is the correct one to use? 
#     <li> What do you notice? What does this tell you about the difference in application of the $t$ and $z$ statistic?
#     </ul>
# <li>  At what temperature should we consider someone's temperature to be "abnormal"?
#     <ul>
#     <li> As in the previous example, try calculating everything using the boostrap approach, as well as the frequentist approach.
#     <li> Start by computing the margin of error and confidence interval. When calculating the confidence interval, keep in mind that you should use the appropriate formula for one draw, and not N draws.
#     </ul>
# <li>  Is there a significant difference between males and females in normal temperature?
#     <ul>
#     <li> What testing approach did you use and why?
#     <li> Write a story with your conclusion in the context of the original problem.
#     </ul>
# </ol>
# 
# You can include written notes in notebook cells using Markdown: 
#    - In the control panel at the top, choose Cell > Cell Type > Markdown
#    - Markdown syntax: http://nestacms.com/docs/creating-content/markdown-cheat-sheet
# 
# #### Resources
# 
# + Information and data sources: http://www.amstat.org/publications/jse/datasets/normtemp.txt, http://www.amstat.org/publications/jse/jse_data_archive.htm
# + Markdown syntax: http://nestacms.com/docs/creating-content/markdown-cheat-sheet
# 
# ****

# In[1]:


import numpy as np
from scipy.stats import kstest, normaltest, ttest_1samp, wilcoxon, ranksums, ttest_ind
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.stats as stats

os.chdir('C:\Users\Rezanick\Desktop\Projects\Springboard\Exercise 2\EDA_human_temperature\data')
df = pd.read_csv('human_body_temperature.csv')


# In[2]:


df.head()


# In[3]:


mean=np.mean(df['temperature'])
print('Mean human body temperature is: %1.2f')%(mean)
std=np.std(df['temperature'])
print('Standard deviation of human body temperature is: %1.2f')%(std)


# In[4]:


from __future__ import division

def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""

    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)
    
    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y


# In[5]:


plt.hist(df['temperature'])
plt.xlabel('Human body temperature')
plt.show()


# In[6]:


samples = np.random.normal(mean, std, size=10000)
plt.hist(samples)
plt.xlabel('Human body temperature')
plt.show()


# In[7]:


x_theor, y_theor = ecdf(samples)
x, y = ecdf(df['temperature'])

_ = plt.plot(x_theor, y_theor)
_ = plt.plot(x, y, marker='.', linestyle='none')
plt.margins(0.05)
_ = plt.xlabel('Human body temperature')
_ = plt.ylabel('CDF')
plt.show()


# In[8]:


#We use Kolmogorov–Smirnov test test to check the normality of the data
def explain_test(pval,test):
    if pval > 0.05:
        print('The '+ test + ' indicates that human body temperature dataset is not normally distirbuted')
    else:
        print('The '+ test + ' indicates that the data is normally distirbuted')

def norm_tests(data):
    _,p = kstest(data, 'norm')
    explain_test(p,'Kolmogorov-Smirnov test')
    
norm_tests(df['temperature'])


# 1) Is the distribution of body temperatures normal?
# 
# The theoretical CDF and the ECDF of the data and Kolmogorov-Smirnov test suggest that the human body temperature data is, indeed, Normally distributed. 
# 
# 2) Is the sample size large? Are the observations independent?
# 
# Yes, the sample size is large (n >30). The observations are independant as each observation represents one individual which is independant of any other person.

# Confidence interval calculated by bootstraping shows that 98.6 is not within either 95% or 99% confidence intervals and may not be the true population mean.

# Based on CLT, the standard deviation of sample distribution is estimated by sd_sample / sqrt(sample_size):

# In[9]:


std_sample_distribution = std/np.sqrt(len(df['temperature']))
print(std_sample_distribution)


# In[10]:


#95% confidence interval calculation 

#upper boundry 

ub95= mean + (1 - ((1 - 0.95)/2))*std_sample_distribution

print('upper boundary for 95% confidence interval is:')
print(ub95)

#lower boundry 

lb95= mean - (1 - ((1 - 0.95)/2))*std_sample_distribution

print('lower boundary for 95% confidence interval is:')
print(lb95)

#99% confidence interval calculation 

#upper boundry 

ub99= mean + (1 - ((1 - 0.99)/2))*std_sample_distribution

print('upper boundary for 99% confidence interval is:')
print(ub99)

#lower boundry 

lb99= mean - (1 - ((1 - 0.99)/2))*std_sample_distribution

print('lower boundary for 99% confidence interval is:')
print(lb99)


# 3) Is the true population mean really 98.6 degrees F? 
# 
# 
# We are 99% confident that the true population mean is between 98.18 degrees and 98.31 degrees so 98.6 is unlikely to be the true population mean.
# 
# We are not comparing two individual groups so we use the one-sample test.
# 
# 
# The sample size is large (n > 30) so the sample distribution can be assumbed to be normal and hence we can use the z statistic instead of a t statistic.

# In[11]:


ub99_t = mean + stats.t.ppf(1-0.005, df.temperature.size - 1) * std_sample_distribution
lb99_t = mean - stats.t.ppf(1-0.005, df.temperature.size - 1) * std_sample_distribution

print('upper boundary for 99% confidence interval from t test is:')
print(ub99_t)

print('lower boundary for 99% confidence interval from t test is:')
print(lb99_t)


# In[12]:


#We create boostsrap replicates to find out confidence interval using this test

def bootstrap_replicate_1d(data, func):
    return func(np.random.choice(data, size=len(data)))

def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)

    return bs_replicates

bs_replicates = draw_bs_reps(df['temperature'], np.mean, size=10000)

plt.hist(bs_replicates, bins=50, normed=True)
plt.xlabel('Human body temperature (F)')
plt.ylabel('PDF')
plt.show()

conf_int = np.percentile(bs_replicates, [2.5, 97.5])
# Print the confidence interval
print('95% confidence interval for human body temperature:')
print(conf_int)

conf_int = np.percentile(bs_replicates, [0.5, 99.5])
# Print the confidence interval
print('99% confidence interval for human body temperature:')
print(conf_int)


# The 99% condfidence interval using the t-statistic has a wider range than what obtained form the Z-Statistic (98.08-98.41 vs. 98.18-98.31)

# In[13]:


#Draw a small sample of size 10 from the data and repeat both frequentist tests.

df_10=np.random.choice(df.temperature, size=10)

mean10 = np.mean(df_10)
std10=np.std(df_10)


# The sample size this time is small (n < 30) and hence the sampling distribution cannot be assumed as normal and ttest is the method of choice to compre this small sample to answer the questions.

# In[14]:


bs_replicates10 = draw_bs_reps(df_10, np.mean, size=10000)

plt.hist(bs_replicates10, bins=50, normed=True)
plt.xlabel('Human body temperature (F)')
plt.ylabel('PDF')
plt.show()

conf_int10 = np.percentile(bs_replicates10, [2.5, 97.5])
# Print the confidence interval
print('95% confidence interval for human body temperature:')
print(conf_int10)

conf_int10 = np.percentile(bs_replicates10, [0.5, 99.5])
# Print the confidence interval
print('99% confidence interval for human body temperature:')
print(conf_int10)


# Bootstrapping test shows wider confidence interval range in the small sample size (size=10) than original data

# In[16]:


from __future__ import division

std_sample10 = (std10)/np.sqrt(len(df_10))

ub99_t10 = mean10 + stats.t.ppf(1-0.005, len(df_10) - 1) * std_sample10
lb99_t10 = mean10 - stats.t.ppf(1-0.005, len(df_10) - 1) * std_sample10

print('upper boundary for 99% confidence interval from t test is:')
print(ub99_t10)

print('lower boundary for 99% confidence interval from t test is:')
print(lb99_t10)


ub99_z10 = mean10 + (stats.norm(0,1).ppf(1 - ((1 - 0.99)/2))) * std_sample10
lb99_z10 = mean10 - (stats.norm(0,1).ppf(1 - ((1 - 0.99)/2))) * std_sample10

print('upper boundary for 99% confidence interval from z test is:')
print(ub99_z10)

print('lower boundary for 99% confidence interval from z test is:')
print(lb99_z10)


# 4. Draw a small sample of size 10 from the data and repeat both frequentist tests. 
# 
# The 99% confidence interval is 97.55-98.64 in this test (small sample size of 10 and performing ttest). We could draw this conclusion that 98.6 could be the true mean of body temperature. 
# 
# 

# 5) At what temperature should we consider someone's temperature to be "abnormal"?
# 
# We are 99% confident that the true population mean is between 98.18 degrees and 98.31 so any temperature above 98.31 or below 98.18 is considered abnormal.

# In[17]:


#Comparison between males and females

males = df[df['gender'] == 'M']['temperature']
females = df[df['gender'] == 'F']['temperature']

mean_males = np.mean(males)
mean_females = np.mean(females)

print('Males:')
print(mean_males)
print('Females:')
print(mean_females)


# In[18]:


plt.hist(males, bins=50, normed=True, alpha=0.5, label='M')
plt.hist(females, bins=50, normed=True, alpha=0.5, label='F')
plt.xlabel('Human body temperature (F)')
plt.ylabel('PDF')
plt.legend()
plt.show()


# We perform a two-sample bootstrap test to compare the means between two groups:
# 
# 1) we shift both arrays to have the same mean, since we are simulating the hypothesis that their means are, in fact, equal. 
# 
# 2) We then draw bootstrap samples out of the shifted arrays and compute the difference in means. This constitutes a bootstrap replicate, and we generate many of them. 
# 
# 3) The p-value is the fraction of replicates with a difference in means greater than or equal to what was observed.

# In[19]:


#Two-sample Bootstrap method to compre body temperature between men and women 

males_shifted = males - mean_males + mean
females_shifted = females - mean_females + mean

bs_replicates_males = draw_bs_reps(males_shifted, np.mean, size=10000)
bs_replicates_females = draw_bs_reps(females_shifted, np.mean, size=10000)

# Get replicates of difference of means: bs_replicates
bs_replicates = bs_replicates_females - bs_replicates_males

p = np.sum(bs_replicates >= (mean_females - mean_males)) / len(bs_replicates)
print('p-value = %1.3f')%p


# 6) Is there a significant difference between males and females in normal temperature? 
# 
# This clearly shows that the null hypothesis of the similarity of mean body temperature is incorrect and there is a significant difference between men and women in terms of body temperature.

# This is impossible to measure the body temperature of all human beings and hence we rely on measurements in small sample sizes and statistical methods to make inferences regaring the true value of human body temperature. What we have found from our analysis in the data is as follows:
# 
# 1) We observed normal distribution for body temperature measurement. 
# 
# 2) We are 99% confident that the true mean value lies between 98.18-98.31 degrees and 98.6 is not the true mean for body temperature. This confidence interval is measured based on z test as the sample size was large enough (n > 30)
# 
# 3) We randomly drew 10 samples from our data and attempted to draw conclusions based on this small sample size. T test statistic was the appropriate method for this purpose and revealed a 99% confidence interval of 97.55-98.64 degrees which shows that 98.6 can be the true mean (contrary to what we found in z test with large dataset)
# 
# 4) There is a statistically significant difference between men and women and women tend to have higher body temperature (mean values: 98.3 vs. 98.10)
