import numpy as np

class GaussianMixture(object):
	def __init__(self, dim, N, bounds):
		def random_gaussian(dim):
			mean = np.random.uniform(-bounds, bounds, size=dim)
			cov_chol = np.random.normal(size=(dim,dim))
			cov = np.matmul(cov_chol.T, cov_chol)
			# make sure it's not too skinny
			w, v = np.linalg.eigh(cov)
			w += 0.1
			cov = 0.1 * bounds * np.matmul(np.matmul(v, np.diag(w)), v.T)
			return mean, cov
		self.gs = [random_gaussian(dim) for _ in range(N)]

	def sample(self, N):
		k = len(self.gs)
		select = np.random.choice(range(k), size=N)
		count = (np.sum(select == i) for i in range(k))
		x = np.vstack(np.random.multivariate_normal(mu, cov, n)
				for (mu, cov), n in zip(self.gs, count))
		return x
