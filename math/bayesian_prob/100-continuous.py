from scipy import special

def posterior(x, n, p1, p2):
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not (0 <= p1 <= 1) or not (0 <= p2 <= 1):
        raise ValueError("p must be a float in the range [0, 1]")
    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")

    # Calculate the Beta parameters for the prior
    alpha = 1
    beta = 1

    # Calculate the Beta parameters for the posterior
    alpha_post = alpha + x
    beta_post = beta + n - x

    # Calculate the posterior probability
    posterior_prob = special.betainc(alpha_post, beta_post, p2) - special.betainc(alpha_post, beta_post, p1)

    return posterior_prob
