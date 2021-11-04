import numpy as np
import random
import copy


class Gaussian:
    """
    Gaussian noise for actions.
    Mean 0, Sigma reducing over time.
    """
    def __init__(self, size, seed, mu, sigma, decay):
        """
        Initializes the parameters and the noise process.
        :param size: Dimensions of the environments actions
        :param seed: Random seed
        :param mu: Mean / reversion for OU process (default = 0)
        :param decay: Decay rate of variance sigma
        :param sigma: Variance / diffusion for OU process
        """
        self.size = size
        self.mu = mu * np.ones(size)
        self.decay = decay
        self.sigma = sigma
        self.theta = "Gaussian"
        self.covariance_matrix = np.eye(size) * sigma
        self.sigma_start = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """
        (Re)sets sigma.
        :return: None
        """
        self.sigma = copy.copy(self.sigma_start)

    def sample(self):
        """
        Sample mv gaussian function.
        :return: Action noise sample
        """
        action_noise = np.random.multivariate_normal(mean=self.mu, cov=self.covariance_matrix, size=self.size)
        self.covariance_matrix = self.sigma * np.eye(self.size)
        return action_noise[0]


class OUNoise:
    """
        Ornstein Uhlenbeck noise for actions.
        For reference see http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
        Stochastic gaussian-process. Process X converges to 'mean reversion level' mu by time.
        The 'mean-reversion-speed' theta induces the 'attraction' of mu on X.
        The diffusion sigma controls the randomness of the process.
    """
    def __init__(self, size, seed, mu, theta, sigma):
        """
        Initializes the parameters and the noise process.
        :param size: Dimensions of the environments actions
        :param seed: Random seed
        :param mu: Mean / reversion for OU process (default = 0)
        :param theta: Reversion speed
        :param sigma: Variance / diffusion for OU process
            0.2 works well for Pendulum-v0; 1.2 for Qube-v0?
        """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """
        Resets the noise process' internal state to mean mu.
        :return: None
        """
        self.state = copy.copy(self.mu)

    def sample(self):
        """
        Updates the noise process' internal state and returns it as noise sample.
        :return: The noise sample
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
