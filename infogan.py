import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gaussian_mixture import GaussianMixture

hidden_sz = 64
batch_sz = 128
critic_batches = 4

# Wasserstein GAN extra hyperparameters
wgan_alpha = 5.0e-4
wgan_c = 0.01

# InfoGAN extra hyperparameters
infogan_lambda = 0.001
cdim = 1

# parameters of our true distribution
xdim = 2
n_gaussians_mixed = 4

# generic parameters
zdim = 8

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
	np.random.seed(105)

	with tf.variable_scope("generator"):
		g_in_z = tf.placeholder(dtype=tf.float32, shape=[None, zdim])
		g_in_c = tf.placeholder(dtype=tf.float32, shape=[None, cdim])
		g_in = tf.concat([g_in_z, g_in_c], axis=1)
		g_fc1 = tf.layers.dense(g_in,  hidden_sz, activation=tf.nn.relu)
		g_fc2 = tf.layers.dense(g_fc1, hidden_sz, activation=tf.nn.relu)
		#g_fc3 = tf.layers.dense(g_fc2, hidden_sz, activation=tf.nn.relu)
		g_out = tf.layers.dense(g_fc2, xdim)

	c_keep = tf.placeholder(dtype=tf.float32, shape=())
	c_in_real = tf.placeholder(dtype=tf.float32, shape=[None, xdim],
		name="discriminator_real_input")
	c_real, c_keep, _ = critic(c_in_real, c_keep, reuse=False)

	c_fake, _, c_headless = critic(g_out, c_keep, reuse=True)

	with tf.variable_scope("q_c"):
		q_mean = tf.layers.dense(c_headless, cdim)
		#q_std = tf.ones((cdim,))
		#q = tf.distributions.Normal(loc=q_mean, scale=q_std)
		#infogan_reward = tf.reduce_mean(tf.layers.flatten(q.log_prob(g_in_c)))
		infogan_reward = -infogan_lambda * tf.reduce_mean(tf.reduce_sum((g_in_c - q_mean)**2, axis=1))

	with tf.variable_scope("loss"):
		c_reward_wasserstein = tf.reduce_mean(c_real) - tf.reduce_mean(c_fake)
		c_reward = c_reward_wasserstein + infogan_reward
		g_reward_wasserstein = tf.reduce_mean(c_fake)
		g_reward = g_reward_wasserstein + infogan_reward

	vars = tf.trainable_variables()

	c_vars = [v for v in vars if v.name.startswith("critic") or v.name.startswith("q_c")]
	c_adam = tf.train.RMSPropOptimizer(wgan_alpha)
	c_step = tf.Variable(0, name="c_step", trainable=False)
	c_optimize = c_adam.minimize(-c_reward, global_step=c_step, var_list=c_vars)

	g_vars = [v for v in vars if v.name.startswith("generator")]
	g_adam = tf.train.RMSPropOptimizer(wgan_alpha)
	g_step = tf.Variable(0, name="g_step", trainable=False)
	g_optimize = g_adam.minimize(-g_reward, global_step=g_step, var_list=g_vars)

	lim = 12
	gm = GaussianMixture(xdim, n_gaussians_mixed, 0.8 * lim)

	np.random.seed(420)

	#tf_gaussians = [tf.distributions.Normal(loc=m, scale=np.diag(c).flat)
		#for m, c in gaussians]
	#density_in = tf.placeholder(dtype=tf.float32, shape=[None]+input_shape)
	#true_density = tf.reduce_sum([g.prob(density_in) for g in tf_gaussians]) / float(n_mix)

	# for printing
	rews = [c_reward, g_reward, infogan_reward]
	rew_names = ["critic", "generator", "infogan"]

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	def generator_feed(N):
		g_latent_z = np.random.normal(size=(N, zdim))
		g_latent_c = np.random.normal(size=(N, cdim))
		return {
			g_in_c : g_latent_c,
			g_in_z : g_latent_z,
		}

	def plot():
		n_plot = 1000

		plt.clf()
		plt.hold(True)

		n_grid = 100
		t = np.linspace(-lim, lim, n_grid)
		x, y = np.meshgrid(t, t)
		xy = np.concatenate([x[:,:,None], y[:,:,None]], axis=2).reshape((-1, 2))
		feed = {c_in_real: xy, c_keep: 1.0}
		cval = sess.run(c_real, feed_dict=feed).reshape(x.shape)
		plt.contourf(x, y, cval, 20, alpha=0.3, antialiased=True, cmap="bone")

		feed = generator_feed(n_plot)
		g_sample = sess.run(g_out, feed_dict=feed)
		xr, yr = gm.sample(n_plot).T
		x, y = g_sample.T
		c = feed[g_in_c]
		cutoff = 2*np.std(c)
		cnorm = 0.5 * ((c - np.mean(c) / cutoff) + 1.0)
		plt.scatter(xr, yr, c=(0.2, 0.2, 0.2), edgecolors='none')
		plt.scatter(x, y, c=c, cmap="rainbow", edgecolors='none')
		plt.xlim([-lim, lim])
		plt.ylim([-lim, lim])
		plt.show(block=False)
		plt.pause(0.001)


	n_batches = 40000
	for b in range(n_batches):
		# train the critic
		for _ in range(critic_batches):
			# generate real data
			r_sample = gm.sample(batch_sz)

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
		feed = generator_feed(batch_sz)
		feed.update({
			c_keep: 1.0
		})
		_, g_sample, g_rew_value = sess.run(
			[g_optimize, g_out, g_reward],
			feed_dict=feed)

		if b % 100 == 0:
			feed = generator_feed(batch_sz)
			feed.update({
				c_in_real: r_sample,
				c_keep: 1.0,
			})
			rew_vals = sess.run(rews, feed_dict=feed)
			print("\nrewards:")
			for name, val in zip(rew_names, rew_vals):
				print("{: <10} = {:.5f}".format(name, val))
			print()
			plot()
	
	plt.show(block=True)


main()
