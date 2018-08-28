Definition of simulated annealing algorithm for optimisation:

    import numpy as np
    
    def neighbour(xs, sigma=0.1):
        # geometric brownian step
        new_xs = xs * (1 + np.random.normal(0, sigma, size=len(xs)))
        return new_xs
    
    def optimise_by_annealing(f, n,
                              init=None,
                              scale=lambda x: x,
                              tol=1,
                              max_iter=1000,
                              init_temperature=100.0,
                              cool=0.999):
        if init == None:
            # make some initial starting values:
            xs = scale(np.ones(n))
        else:
            xs = scale(init)
        temperature = init_temperature
        for i in range(max_iter):
            loss = f(xs)
            if loss < tol:
                break
            candidate = scale(neighbour(xs))
            candidate_loss = f(candidate)
            if candidate_loss < loss:
                xs = candidate
            else:
                if np.random.uniform() < np.exp(-(candidate_loss - loss) / temperature):
                    xs = candidate
            temperature *= cool
        print("Ran annealing for {} iterations.".format(i+1))
        return xs, loss

Approximating the liquidation probability of a CDP, as a function of its collateralisation, by using a normal approximation.

    import scipy.special
    
    def cumulative_normal(x, mu=0, sigma=1.0):
        return 0.5 * (1 + scipy.special.erf((x - mu)/(np.sqrt(2) * sigma)))
    
    def liquidation_probability(ink, tab, tag, mat, sigma):
        # assumes normal distribution of returns over period
        # return to liquidation:
        hit = mat / (ink * tag / tab) - 1
        return cumulative_normal(hit, sigma=sigma)

Now we construct the optimisation objective for our problem: let \( p_i \) be the liquidation probability of CDP \( i \), which is a function of \( {\tt tab}_i \). Let \( R_i \) be the random variable representing the asset's return over some period, 

\[
p_i = \mathbb{P}(R_i < \frac{{\tt mat}_i\cdot {\tt tag}_i}{{\tt ink}_i \cdot {\tt tag}_i} - 1)
\]

The loss is then the sum of the magnitudes of the pairwise differences of the liquidation probabilities. The idea is that the optimum is achieved when all CDPs have the same probability of getting liquidated:

\[
f({\tt tabs}) = \Sigma_{i=0}^N \Sigma_{j=0}^i \| p_i - p_j \|
\]

    def make_risk_parity_objective(inks, tags, mats, sigmas):
        N = len(inks)
        lp = lambda tab, i: liquidation_probability(inks[i], tab, tags[i], mats[i], sigmas[i])
        f = lambda tabs: sum([sum([abs(lp(tabs[i], i) - lp(tabs[j], j)) for j in range(0, i)]) for i in range(0, N)])
        return f

    def do_example_with_2():
        # ETH and DGX
        gems = ["ETH", "DGX"]
        tags = [420., 38.]
        inks = [10., 100.]
        mats = [1.5, 1.1]
        sigmas = [0.2, 0.1]
        f = make_risk_parity_objective(inks, tags, mats, sigmas)
        # target debt: 3000 dai
        D = 3000
        tabs, loss = optimise_by_annealing(f, 2, scale=(lambda tabs: D * tabs / sum(tabs)), tol=.0000001, max_iter=100000)
        print("Results:")
        for i, gem in enumerate(gems):
          print("{} CDP: tab={}".format(gem, tabs[i]))
          print("(collateral ratio={}, liquidation probability={})".format(inks[i]*tags[i]/tabs[i], liquidation_probability(inks[i], tabs[i], tags[i], mats[i], sigmas[i])))

    do_example_with_2()