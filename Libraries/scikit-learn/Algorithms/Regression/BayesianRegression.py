import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import Adam
import matplotlib.pyplot as plt
import seaborn as sns

# Generate some sample data
torch.manual_seed(0)
X = torch.linspace(start=0, end=10, steps=100)
true_slope = 2
true_intercept = 1
Y = true_intercept + true_slope * X + torch.randn(100)


# Define the Bayesian regression model
def model(X, Y):
    # Priors for the parameters
    slope = pyro.sample(name="slope", fn=dist.Normal(loc=0, scale=5))  # 10 yerine 5 kullanarak dağılımı daralt
    intercept = pyro.sample(name="intercept", fn=dist.Normal(loc=0, scale=5))
    sigma = pyro.sample(name="sigma", fn=dist.HalfNormal(0.5))  # Daha düşük bir sigma prior

    # Expected value of the outcome
    mu = intercept + slope * X

    # Likelihood (sampling distribution) of the observations
    with pyro.plate(name="data", size=len(X)):
        pyro.sample(name="obs", fn=dist.Normal(mu, sigma), obs=Y)


# Run Bayesian inference using SVI (Stochastic Variational Inference)
def guide(X, Y):
    # Approximate posterior distributions for the parameters
    slope_loc = pyro.param(name="slope_loc", init_tensor=torch.tensor(0.0))
    slope_scale = pyro.param(name="slope_scale", init_tensor=torch.tensor(1.0),
                             constraint=dist.constraints.positive)
    intercept_loc = pyro.param(name="intercept_loc", init_tensor=torch.tensor(0.0))
    intercept_scale = pyro.param(name="intercept_scale", init_tensor=torch.tensor(1.0),
                                 constraint=dist.constraints.positive)
    sigma_loc = pyro.param(name="sigma_loc", init_tensor=torch.tensor(1.0),
                           constraint=dist.constraints.positive)

    # Sample from the approximate posterior distributions
    slope = pyro.sample(name="slope", fn=dist.Normal(slope_loc, slope_scale))
    intercept = pyro.sample(name="intercept", fn=dist.Normal(intercept_loc,
                                                             intercept_scale))
    sigma = pyro.sample(name="sigma", fn=dist.HalfNormal(sigma_loc))


# Initialize the SVI and optimizer
optim = Adam({"lr": 0.001})  # 0.01 yerine daha düşük bir öğrenme oranı
svi = SVI(model=model, guide=guide, optim=optim, loss=Trace_ELBO())

# Run the inference loop
num_iterations = 1000
for i in range(num_iterations):
    loss = svi.step(X, Y)
    if (i + 1) % 100 == 0:
        print(f"Iteration {i + 1}/{num_iterations} - Loss: {loss}")

# Obtain posterior samples using Predictive
predictive = Predictive(model=model, guide=guide, num_samples=1000)
posterior = predictive(X, Y)

# Extract the parameter samples
slope_samples = posterior["slope"]
intercept_samples = posterior["intercept"]
sigma_samples = posterior["sigma"]

# Compute the posterior means
slope_mean = slope_samples.mean()
intercept_mean = intercept_samples.mean()
sigma_mean = sigma_samples.mean()

# Print the estimated parameters
print("Estimated Slope:", slope_mean.item())
print("Estimated Intercept:", intercept_mean.item())
print("Estimated Sigma:", sigma_mean.item())

# Create subplots
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

# Plot the posterior distribution of the slope
sns.kdeplot(slope_samples, fill=True, ax=axs[0])
axs[0].set_title("Posterior Distribution of Slope")
axs[0].set_xlabel("Slope")
axs[0].set_ylabel("Density")

# Plot the posterior distribution of the intercept
sns.kdeplot(intercept_samples, fill=True, ax=axs[1])
axs[1].set_title("Posterior Distribution of Intercept")
axs[1].set_xlabel("Intercept")
axs[1].set_ylabel("Density")

# Plot the posterior distribution of sigma
sns.kdeplot(sigma_samples, fill=True, ax=axs[2])
axs[2].set_title("Posterior Distribution of Sigma")
axs[2].set_xlabel("Sigma")
axs[2].set_ylabel("Density")

# Adjust the layout
plt.tight_layout()

# Show the plot
plt.show()
