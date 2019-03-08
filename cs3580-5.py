import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets.samples_generator import make_blobs
from pandas.plotting import parallel_coordinates
from scipy.stats import norm


train = pd.read_csv("train.csv")
trainNNull = train.dropna()

#Chi-Squared tests:
#Perform a Chi-Squared test on the 'Sex' column using 'Survived' as the dependent variable. What did you find? What does it mean?
contingency_table = pd.crosstab(
train['Sex'],
train['Survived'],
margins = True
)
chi2Stat, p, dof, expected = stats.chi2_contingency(contingency_table)
print("\nChi2 Statistic equals "+str(chi2Stat)+" with "+str(dof)+" degrees of freedom yields a p-value of "+str(p)+
      "\nwhich indicates that survival versus sex is statistically significant.")

#Perform a similar Chi-Squared on 'Pclass' using 'Survived' as the dependent variable. What did you find? What does it mean?
contingency_table = pd.crosstab(
train['Pclass'],
train['Survived'],
margins = True
)
chi2Stat, p, dof, expected = stats.chi2_contingency(contingency_table)
print("\nChi2 Statistic equals "+str(chi2Stat)+" with "+str(dof)+" degrees of freedom yields a p-value of "+str(p)+
      "\nwhich indicates that survival versus Pclass is statistically significant.")

#Perform a similar Chi-Squared on 'Embarked' using 'Survived' as the dependent variable. What did you find? What does it mean?
contingency_table = pd.crosstab(
train['Embarked'],
train['Survived'],
margins = True
)
chi2Stat, p, dof, expected = stats.chi2_contingency(contingency_table)
print("\nChi2 Statistic equals "+str(chi2Stat)+" with "+str(dof)+" degrees of freedom yields a p-value of "+str(p)+
      "\nwhich indicates that survival versus embarked is statistically significant.\n")

# ANOVA:
# Perform an ANOVA on 'Pclass' as it is given in the dataset using 'Fare' as the dependent variable.
# If there is statistical significance, do a Tukey HSD comparison and report which passenger classes are statistically significant from each other.
# Regardless, report on what you found and what it means.
#Pclass affects fare
model = ols('Fare ~ Pclass', data=train).fit()
anova = sm.stats.anova_lm(model, typ=2)
mc = MultiComparison(train['Fare'], train['Pclass'])
#reject -> reject null hypothesis
thsd = mc.tukeyhsd()
print(thsd)
print("Anova indicated a significant difference was present with classes 'Fare' and 'Pclass'. The Tukey HSD comparison indicates that Pclass==1 is statistically signifcant from Pclass==2;"
      "\nPclass==1 is statistically significant from Pclass==3; and Pclass==2 is not statistically significant from Pclass==3")

# Separate 'Sex' Column:
# Many people think that the best predictor is the passenger's sex.
# Do the following:
# Separate the data based on the sex of the passenger.
female = train.loc[train["Sex"]=="female"]
male = train.loc[train["Sex"]=="male"]

#TODO:
# What is the correlation of 'female' to survived?
# What is the correlation of 'male' to survived? (It is okay if it opposite of the female.)
mf = train.replace('male',1).replace('female',0)
mf.corr()
print("Female to survived correlation = 0.54. Male to survived correlation = -0.54")

# (Pick two columns for this question.) For the different columns, are they all normal distributions?
# For example, is the 'Fare' a normal distribution? Visualize the distribution. If it is not a normal distribution, transform it. Visualize it again.
# sns.distplot (e.g. sns.distplot(df_train['SalePrice']);)
plt.figure(5)
plt.title("Age and Survival")
sns.distplot(train['Age'].dropna(), fit=norm,kde=False)
print("Age versus Survival is close to a normal distribution, but it is not a perfect one.")

plt.figure(6)
plt.title("Fare and Survival")
sns.distplot(train['Fare'].dropna(), fit=norm,kde=False)
print("Fare versus survival is not a normal distribution.")


# Bivariate Visualizations:
# Do the following at least 3 times (3 visualizations):
#
# Create a bivariate visualization (e.g. scatterplot) between the 'Survived' column and another.
#
# You may separate columns - like you did with the Sex column - and visualize subsets of the data if you think that will increase your understanding.
# For example, you can do a scatterplot of 'Survived' and female.
plt.figure(1)
plt.title("Fare and Survival")
plt.scatter(train['Fare'], train['Survived'])
plt.figure(2)
plt.title("Age and Survival")
plt.scatter(train['Age'], train['Survived'])
plt.figure(3)
plt.title("Pclass and Survival")
df = train.groupby(['Pclass'])['Survived'].mean()
df.plot.bar()

# Multivariate Visualization:
# Now we need to see actual interaction between multiple variables.
# Use a mutlivariate visualization (like a Parallel Coordinates, heat map, pair plot of multiple columns)

plt.figure(4)
plt.title("Survival and other normalized columns")
#normalize
trainNorm = train.replace('female', 0).replace('male', 1)._get_numeric_data().dropna()
surv = trainNorm['Survived']
sex = trainNorm['Sex']
#different kind of norm trainNorm = (trainNorm - trainNorm.mean(axis=0))/(trainNorm.var(axis=0))
trainNorm = (trainNorm-trainNorm.min())/(trainNorm.max()-trainNorm.min())
trainNorm['Survived'] = surv
trainNorm['Sex'] = sex
parallel_coordinates(trainNorm, 'Survived')
plt.show()


# Small Report:
# In the end, write a small report (hardcode it in your python code) where you indicate the most important columns.
# Explain why you think they are the most important columns (or subset of columns).
print("\nSex is one of the most influential factors. Sex, embarked, and Pclass all have p-values <0.05 when doing chi2"
      "\nFrom the multivariate graph, sex and Pclass results from above are validated."
      "\nAdditionally, SibSp, Parch, and PClass all seem important when predicting survival according to this multivariate graph.")
