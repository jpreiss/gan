import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from gaussian_mixture import GaussianMixture

hidden_sz = 64
n_mix = 4

# hyperparams on page 8 of WGAN paper
alpha = 5.0e-4
c = 0.01
m = 128
ncritic = 4

def clip_fn(x):
	return tf.clip_by_value(x, -c, c)

def critic(x, keep, reuse):
	with tf.variable_scope("critic", reuse=reuse):
		c_fc1 = tf.layers.dense(x,  hidden_sz, activation=tf.nn.relu,
			kernel_constraint=clip_fn, bias_constraint=clip_fn)
		c_dropout = tf.nn.dropout(c_fc1, keep)
		c_fc2 = tf.layers.dense(c_dropout, hidden_sz, activation=tf.nn.relu,
			kernel_constraint=clip_fn, bias_constraint=clip_fn)
		c_out = tf.layers.dense(c_fc2, 1,
			kernel_constraint=clip_fn, bias_constraint=clip_fn)
		return c_out, keep

def main():
	np.random.seed(100)

	input_shape = [2]
	latent_shape = [8]

	with tf.variable_scope("generator"):
		g_in = tf.placeholder(dtype=tf.float32, shape=[None]+latent_shape,
			name="generator_latent")
		g_fc1 = tf.layers.dense(g_in,  hidden_sz, activation=tf.nn.relu)
		g_fc2 = tf.layers.dense(g_fc1, hidden_sz, activation=tf.nn.relu)
		g_out = tf.layers.dense(g_fc2, input_shape[0])

	c_keep = tf.placeholder(dtype=tf.float32, shape=())
	c_in_real = tf.placeholder(dtype=tf.float32, shape=[None]+input_shape,
		name="discriminator_real_input")
	c_real, c_keep = critic(c_in_real, c_keep, reuse=False)

	c_fake, _ = critic(g_out, c_keep, reuse=True)

	with tf.variable_scope("loss"):
		c_reward = tf.reduce_mean(c_real) - tf.reduce_mean(c_fake)
		g_reward = tf.reduce_mean(c_fake)

	vars = tf.trainable_variables()

	c_adam = tf.train.RMSPropOptimizer(alpha)
	c_step = tf.Variable(0, name="c_step", trainable=False)
	c_optimize = c_adam.minimize(-c_reward, global_step=c_step,
		var_list=[v for v in vars if v.name.startswith("critic")])

	g_adam = tf.train.RMSPropOptimizer(alpha)
	g_step = tf.Variable(0, name="g_step", trainable=False)
	g_optimize = g_adam.minimize(-g_reward, global_step=g_step,
		var_list=[v for v in vars if v.name.startswith("generator")])

	gm = GaussianMixture(input_shape[0], n_mix, 6.0)

	#tf_gaussians = [tf.distributions.Normal(loc=m, scale=np.diag(c).flat)
		#for m, c in gaussians]
	#density_in = tf.placeholder(dtype=tf.float32, shape=[None]+input_shape)
	#true_density = tf.reduce_sum([g.prob(density_in) for g in tf_gaussians]) / float(n_mix)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	def distmap():
		plt.clf()
		plt.hold(True)

		res = 64
		lim = 10
		t = np.linspace(-lim, lim, 64)
		x, y = np.meshgrid(t,t)
		mesh = np.float32(np.c_[x.flat, y.flat])

		#dens_true = sess.run(true_density, feed_dict={density_in: mesh})
		#dsurface = sess.run(c_real, feed_dict={c_in_real: mesh, c_keep: 1.0})
		#dsurface = dsurface.reshape(x.shape)

		r_sample = gm.sample(n_mix * 10000)
		r_hist, _, _ = np.histogram2d(r_sample[:,0], r_sample[:,1],
			bins=res, range=[[-lim, lim], [-lim, lim]])

		g_latent = np.random.normal(size=[10000] + latent_shape)
		g_sample = sess.run(g_out, feed_dict={g_in: g_latent})
		g_hist, _, _ = np.histogram2d(g_sample[:,0], g_sample[:,1],
			bins=res, range=[[-lim, lim], [-lim, lim]])

		def normalize(x):
			m = np.max(x.flat)
			if m > 0:
				return x / m
			return x

		img = np.zeros((res,res,3))
		img[:,:,0] = normalize(r_hist)
		img[:,:,1] = normalize(g_hist)

		plt.imshow(img, extent=(-lim, lim, -lim, lim))

	n_batches = 40000
	for b in range(n_batches):
		# train the critic
		for _ in range(ncritic):
			# generate real data
			r_sample = gm.sample(m)

			# sample from generator
			g_latent = np.random.normal(size=[m] + latent_shape)

			# train the critic
			_, c_rew_value = sess.run([c_optimize, c_reward],
				feed_dict={c_in_real: r_sample, g_in: g_latent, c_keep: 0.5})

		# train the generator
		g_latent = np.random.normal(size=[m] + latent_shape)
		_, g_sample, g_rew_value = sess.run([g_optimize, g_out, g_reward],
			feed_dict={g_in: g_latent, c_keep: 1.0})

		if False and b % 100 == 0:
			plt.clf()
			plt.hold(True)
			lim = 10
			t = np.linspace(-lim, lim, 64)
			x, y = np.meshgrid(t,t)
			mesh = np.c_[x.flat, y.flat]
			dsurface = sess.run(c_real, feed_dict={c_in_real: mesh, c_keep: 1.0})
			dsurface = dsurface.reshape(x.shape)
			plt.imshow(dsurface, extent=(-lim, lim, -lim, lim), alpha=0.4)
			plt.scatter(r_sample[:,0], r_sample[:,1], color=(0.7, 0.1, 0.1))
			plt.scatter(g_sample[:,0], g_sample[:,1], color=(0.1, 0.6, 0.3))
			plt.show(block=False)
			plt.xlim([-lim, lim])
			plt.ylim([-lim, lim])
			plt.pause(0.001)

		if b % 100 == 0:
			distmap()
			plt.show(block=False)
			plt.pause(0.001)

	plt.show(block=True)


main()
