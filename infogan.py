import numpy as np
import tensorflow as tf

hidden_sz = 32
batch_sz = 64
critic_batches = 4

# Wasserstein GAN extra hyperparameters
wgan_alpha = 5.0e-4
wgan_c = 0.01

# InfoGAN extra hyperparameters
infogan_lambda = 0.1
cdim = 1

# parameters of our true distribution
xdim = 2
n_gaussians_mixed = 4

# generic parameters
zdim = 2

def clip_fn(x):
	return tf.clip_by_value(x, -wgan_c, wgan_c)

def critic(x, dropout_keep, reuse):
	with tf.variable_scope("critic", reuse=reuse):
		c_fc1 = tf.layers.dense(x,  hidden_sz, activation=tf.nn.relu,
			kernel_constraint=clip_fn, bias_constraint=clip_fn)
		c_dropout = tf.nn.dropout(c_fc1, dropout_keep)
		c_fc2 = tf.layers.dense(c_dropout, hidden_sz, activation=tf.nn.relu,
			kernel_constraint=clip_fn, bias_constraint=clip_fn)
		c_out = tf.layers.dense(c_fc2, 1,
			kernel_constraint=clip_fn, bias_constraint=clip_fn)
		return c_out, dropout_keep, c_fc2

def main():

	with tf.variable_scope("generator"):
		g_in_z = tf.placeholder(dtype=tf.float32, shape=[None, zdim])
		g_in_c = tf.placeholder(dtype=tf.float32, shape=[None, cdim])
		g_in = tf.concat([g_in_z, g_in_c], axis=1)
		g_fc1 = tf.layers.dense(g_in,  hidden_sz, activation=tf.nn.relu)
		g_fc2 = tf.layers.dense(g_fc1, hidden_sz, activation=tf.nn.relu)
		g_out = tf.layers.dense(g_fc2, xdim)

	c_keep = tf.placeholder(dtype=tf.float32, shape=())
	c_in_real = tf.placeholder(dtype=tf.float32, shape=[None, xdim],
		name="discriminator_real_input")
	c_real, c_keep, _ = critic(c_in_real, c_keep, reuse=False)

	c_fake, _, c_headless = critic(g_out, c_keep, reuse=True)

	with tf.variable_scope("q_c"):
		q_mean = tf.layers.dense(c_headless, cdim)
		q_std = tf.Variable(tf.ones((cdim,)))
		q = tf.distributions.Normal(loc=q_mean, scale=q_std)
		g_infogan_reward = q.log_prob(g_in_c) 

	with tf.variable_scope("loss"):
		c_reward = tf.reduce_mean(c_real) - tf.reduce_mean(c_fake) + infogan_lambda * g_infogan_reward
		g_reward = tf.reduce_mean(c_fake) + infogan_lambda * g_infogan_reward

	vars = tf.trainable_variables()

	c_vars = [v for v in vars if v.name.startswith("critic") or v.name.startswith("q_c")]
	c_adam = tf.train.RMSPropOptimizer(wgan_alpha)
	c_step = tf.Variable(0, name="c_step", trainable=False)
	c_optimize = c_adam.minimize(-c_reward, global_step=c_step, var_list=c_vars)

	g_vars = [v for v in vars if v.name.startswith("generator") or v.name.startswith("q_c")]
	g_adam = tf.train.RMSPropOptimizer(wgan_alpha)
	g_step = tf.Variable(0, name="g_step", trainable=False)
	g_optimize = g_adam.minimize(-g_reward, global_step=g_step, var_list=g_vars)

	def random_gaussian(dim):
		randn = np.random.normal
		mean = 3*randn(size=dim)
		cov_chol = randn(size=(dim,dim))
		cov = np.matmul(cov_chol.T, cov_chol)
		# make sure it's not too skinny
		w, v = np.linalg.eigh(cov)
		w += 0.1
		cov = np.matmul(np.matmul(v, np.diag(w)), v.T)
		return mean, cov

	gaussians = [random_gaussian(xdim) for _ in range(n_gaussians_mixed)]

	#tf_gaussians = [tf.distributions.Normal(loc=m, scale=np.diag(c).flat)
		#for m, c in gaussians]
	#density_in = tf.placeholder(dtype=tf.float32, shape=[None]+input_shape)
	#true_density = tf.reduce_sum([g.prob(density_in) for g in tf_gaussians]) / float(n_mix)

	def sample(N, i=None):
		if i is None:
			n = N // len(gaussians)
			assert n * len(gaussians) == N
			return np.vstack(
				np.random.multivariate_normal(mu, cov, size=n)
					for mu, cov in gaussians)
		else:
			mu, cov = gaussians[i]
			return np.random.multivariate_normal(mu, cov, size=N)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	n_batches = 4000
	for b in range(n_batches):
		# train the critic
		for _ in range(critic_batches):
			# generate real data
			r_sample = sample(batch_sz)

			# sample from generator
			g_latent_z = np.random.normal(size=(batch_sz, zdim))
			g_latent_c = np.random.normal(size=(batch_sz, cdim))
			feed = {
				c_in_real: r_sample,
				g_in_c : g_latent_c, 
				g_in_z : g_latent_z,
				c_keep: 0.5
			}
			# train the critic
			_, c_rew_value = sess.run([c_optimize, c_reward], feed_dict=feed)

		# train the generator
		g_latent_z = np.random.normal(size=(batch_sz, zdim))
		g_latent_c = np.random.normal(size=(batch_sz, cdim))
		feed = {
			g_in_c : g_latent_c, 
			g_in_z : g_latent_z,
			c_keep: 1.0
		}
		_, g_sample, g_rew_value = sess.run(
			[g_optimize, g_out, g_reward],
			feed_dict=feed)

	n_plot = 1000
	g_latent_z = np.random.normal(size=(n_plot, zdim))
	g_latent_c = np.random.normal(size=(n_plot, cdim))
	feed = {
		g_in_c : g_latent_c, 
		g_in_z : g_latent_z,
	}
	g_sample = sess.run(g_out, feed_dict=feed)
	np.set_printoptions(threshold=np.inf)
	print(np.hstack([g_sample, g_latent_c]))


main()
