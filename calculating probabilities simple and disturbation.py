import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson, binom, expon

# Simple Probability
print(f"Probability of rolling a 4: {1 / 6}")

# Normal Distribution
mean, std_dev = 50, 10
samples = np.random.normal(mean, std_dev, 1000)
plt.hist(samples, bins=30, density=True, alpha=0.6, color='blue')
x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 100)
plt.plot(x, norm.pdf(x, mean, std_dev), 'r-', lw=2)
plt.title('Normal Distribution')
plt.show()

# Poisson Distribution
print(f"Probability of 3 events/hour: {poisson.pmf(3, 5)}")

# Binomial Distribution
print(f"Probability of 7 successes (10 trials, p=0.6): {binom.pmf(7, 10, 0.6)}")

# Exponential Distribution
exp_samples = np.random.exponential(scale=2, size=1000)
plt.hist(exp_samples, bins=30, density=True, alpha=0.6, color='green')
x_exp = np.linspace(0, 10, 100)
plt.plot(x_exp, expon.pdf(x_exp, scale=2), 'r-', lw=2)
plt.title('Exponential Distribution')
plt.show()
