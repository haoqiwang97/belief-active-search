import pystan
import pickle


stan_model = """
data {
    int<lower=0> D;       // space dimension
    int<lower=0> M;       // number of measurements so far
    real k;               // logistic noise parameter (scale)
    vector[2] bounds;      // hypercube bounds [lower,upper]
    int y[M];             // measurement outcomes
    vector[D] A[M];       // hyperplane directions
    vector[M] tau;        // hyperplane offsets
}
parameters {
    vector<lower=bounds[1],upper=bounds[2]>[D] W;         // the user point
}
transformed parameters {
    vector[M] z;
    for (i in 1:M)
        z[i] = dot_product(A[i], W) - tau[i];
}
model {
    // prior
    W ~ uniform(bounds[1],bounds[2]);

    // linking observations
    y ~ bernoulli_logit(k * z);
}
"""

sm = pystan.StanModel(model_code=stan_model)
with open('model.pkl', 'wb') as f:
    pickle.dump(sm, f)