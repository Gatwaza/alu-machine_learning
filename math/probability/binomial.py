class Binomial:
    """
    Class representing a binomial distribution.
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initialize a Binomial instance.

        Args:
            data (list, optional): List of data points. Defaults to None.
            n (int, optional): Number of Bernoulli trials. Defaults to 1.
            p (float, optional): Probability of success. Defaults to 0.5.

        Raises:
            ValueError: If n is not a positive
            value or p is not a valid probability.
            TypeError: If data is not a list or
            contains less than two data points.
        """
        if n <= 0:
            raise ValueError("n must be a positive value")

        if not (0 < p < 1):
            raise ValueError("p must be greater than 0 and less than 1")

        if data is None:
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.p = sum(data) / (len(data) * n)
            self.n = round(sum(data) / self.p)

            self.p = sum(data) / (len(data) * self.n)
