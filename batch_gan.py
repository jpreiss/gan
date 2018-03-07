import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

hidden_sz = 32
n_hidden = 3

input_dim = 2
latent_dim = 2
in_batch = 8

d_grad_steps = 1
batch = 64

n_mix = 4

def fc(x, dropout_keep, out_dim):
	x = tf.layers.dense(x, hidden_sz, activation=tf.nn.relu)
	for _ in range(n_hidden - 1):
		dropout = tf.nn.dropout(x, dropout_keep)
		x = tf.layers.dense(dropout, hidden_sz, activation=tf.nn.relu)
	out = tf.layers.dense(x, out_dim)
	return out

def discriminator(x, keep, reuse):
	with tf.variable_scope("discriminator", reuse=reuse):
		prev = tf.layers.dense(x,  hidden_sz, activation=tf.nn.relu)
		for _ in range(n_hidden - 1):
			dropout = tf.nn.dropout(prev, keep)
			prev = tf.layers.dense(dropout, hidden_sz, activation=tf.nn.relu)
		d_out = tf.sigmoid(tf.layers.dense(prev, 1))
		return d_out, keep 

def main():

	np.seterr(all='raise')

	with tf.variable_scope("generator"):
		g_in = tf.placeholder(
			dtype=tf.float32,
			shape=[None, latent_dim],
			name="generator_latent")
		g_out = fc(g_in, tf.constant(1.0), input_dim)
		g_out = tf.reshape(g_out, [-1, in_batch, input_dim])

	with tf.variable_scope("discriminator", reuse=False):
		d_keep = tf.placeholder(dtype=tf.float32, shape=())
		d_in_real = tf.placeholder(
			dtype=tf.float32,
			shape=[None, in_batch, input_dim],
			name="discriminator_real_input")
		d_real = tf.sigmoid(fc(d_in_real, d_keep, 1))

	with tf.variable_scope("discriminator", reuse=True):
		d_fake = tf.sigmoid(fc(g_out, d_keep, 1))

	with tf.variable_scope("loss"):
		d_loss = -tf.reduce_mean(tf.log(d_real) + tf.log(1.0 - d_fake))
		g_loss = tf.reduce_mean(tf.log(1.0 - d_fake))

	vars = tf.trainable_variables()

	d_adam = tf.train.AdamOptimizer(learning_rate=0.0001)
	#d_adam = tf.train.GradientDescentOptimizer(0.005)
	d_step = tf.Variable(0, name="d_step", trainable=False)
	d_optimize = d_adam.minimize(d_loss, global_step=d_step,
		var_list=[v for v in vars if v.name.startswith("disc")])

	g_adam = tf.train.AdamOptimizer(learning_rate=0.0001)
	#g_adam = tf.train.GradientDescentOptimizer(0.005)
	g_step = tf.Variable(0, name="g_step", trainable=False)
	g_optimize = g_adam.minimize(g_loss, global_step=g_step,
		var_list=[v for v in vars if v.name.startswith("gen")])

	def random_gaussian(dim):
		randn = np.random.normal
		mean = 3*randn(size=dim)
		#cov_chol = randn(size=(dim,dim))
		#cov = np.matmul(cov_chol.T, cov_chol)
		cov = 0.3 * np.diag(1 + 2 * np.random.uniform(size=2))
		return mean, cov

	gaussians = [random_gaussian(input_dim) for _ in range(n_mix)]

	#tf_gaussians = [tf.distributions.Normal(loc=m, scale=np.diag(c).flat)
		#for m, c in gaussians]
	#density_in = tf.placeholder(dtype=tf.float32, shape=[None]+input_shape)
	#true_density = tf.reduce_sum([g.prob(density_in) for g in tf_gaussians]) / float(n_mix)

	def sample(N):
		n = N // len(gaussians)
		assert n * len(gaussians) == N
		return np.vstack(
			np.random.multivariate_normal(mu, cov, size=n)
				for mu, cov in gaussians)

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
		#dsurface = sess.run(d_real, feed_dict={d_in_real: mesh, d_keep: 1.0})
		#dsurface = dsurface.reshape(x.shape)

		r_sample = sample(n_mix * 10000)
		r_hist, _, _ = np.histogram2d(r_sample[:,0], r_sample[:,1],
			bins=res, range=[[-lim, lim], [-lim, lim]])

		g_latent = np.random.normal(size=(10000, latent_dim))
		g_sample = sess.run(g_out, feed_dict={g_in: g_latent})
		g_sample = g_sample.reshape((-1, input_dim))
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
		# train the discriminator
		nstep = 100 if b == 0 else d_grad_steps
		for _ in range(nstep):
			# generate real data
			r_sample = sample(batch * in_batch).reshape((batch, in_batch, -1))

			# sample from generator
			g_latent = np.random.normal(size=(batch * in_batch, latent_dim))

			# train the discriminator
			_, d_loss_value = sess.run([d_optimize, d_loss],
				feed_dict={d_in_real: r_sample, g_in: g_latent, d_keep: 0.5})

		# train the generator
		g_latent = np.random.normal(size=(batch * in_batch, latent_dim))
		_, g_sample, g_loss_value = sess.run([g_optimize, g_out, g_loss],
			feed_dict={g_in: g_latent, d_keep: 1.0})

		if False and b % 100 == 0:
			plt.clf()
			plt.hold(True)
			lim = 10
			t = np.linspace(-lim, lim, 64)
			x, y = np.meshgrid(t,t)
			mesh = np.c_[x.flat, y.flat]
			dsurface = sess.run(d_real, feed_dict={d_in_real: mesh, d_keep: 1.0})
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
