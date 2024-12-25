import pandas as pd
from scipy import stats

# Load Titanic dataset
data = pd.read_csv('train.csv')

# One Sample T-Test
print("One Sample T-Test:", stats.ttest_1samp(data['Age'].dropna(), 30))

# Two Independent Samples T-Test
males = data[data['Sex'] == 'male']['Age'].dropna()
females = data[data['Sex'] == 'female']['Age'].dropna()
print("Two Independent Samples T-Test:", stats.ttest_ind(males, females))

# Paired T-Test
fares = data['Fare'].dropna()
print("Paired T-Test:", stats.ttest_rel(fares, fares * 1.2))

# ANOVA Test
pclass_1 = data[data['Pclass'] == 1]['Fare'].dropna()
pclass_2 = data[data['Pclass'] == 2]['Fare'].dropna()
pclass_3 = data[data['Pclass'] == 3]['Fare'].dropna()
print("ANOVA Test:", stats.f_oneway(pclass_1, pclass_2, pclass_3))

# Chi-Square Test
chi2_table = pd.crosstab(data['Survived'], data['Pclass'])
print("Chi-Square Test:", stats.chi2_contingency(chi2_table)[:2])
