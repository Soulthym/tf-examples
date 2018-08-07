from __future__ import print_function
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("tmp/data/",one_hot=True)

from tensorflow.examples.tutorials import mnist
import tensorflow as tf

print(dir(mnist))

#lr = 0.1
#nsteps = 500
#bsize = 128
#dstep = 100
#
#nhid1= 256
#nhid2 = 256
#ninput = 784
#nclass = 10
#
#X = tf.placeholder("float",[None,ninput])
#Y = tf.placeholder("float",[None,nclass])
#
#W = {
#	'h1' : tf.Variable(tf.random_normal([ninput,nhid1])),
#	'h2' : tf.Variable(tf.random_normal([nhid1,nhid2])),
#	'out': tf.Variable(tf.random_normal([nhid2,nclass]))
#	}
#
#b = {
#	'b1' : tf.Variable(tf.random_normal([nhid1])),
#	'b2' : tf.Variable(tf.random_normal([nhid2])),
#	'out': tf.Variable(tf.random_normal([nclass]))
#	}
#
#def nn(x):
#	l1 = tf.add(tf.matmul(x,W['h1']),b['b1'])
#	l2 = tf.add(tf.matmul(l1,W['h2']),b['b2'])
#	lout = tf.add(tf.matmul(l2,W['out']),b['out'])
#	return lout
#
#logits = nn(X)
#lossop = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
#	logits=logits, labels=Y))
#optimizer = tf.train.AdamOptimizer(learning_rate=lr)
#trainop = optimizer.minimize(lossop)
#
#correctpred = tf.equal(tf.argmax(logits,1),tf.argmax(Y,1))
#accuracy = tf.reduce_mean(tf.cast(correctpred,tf.float32))
#
#init = tf.global_variables_initializer()
#
#with tf.Session() as sess:
#	sess.run(init)
#	for step in range(1,nsteps+1):
#		batchx,batchy = mnist.train.next_batch(bsize)
#		sess.run(trainop, feed_dict={X: batchx, Y:batchy})
#		if step % dstep == 0 or step == 1:
#			# Calculate batch loss and accuracy
#			loss, acc = sess.run([lossop, accuracy], feed_dict={X: batchx, Y: batchy})
#			print("Step " + str(step) + ", Minibatch Loss= " + \
#				  "{:.4f}".format(loss) + ", Training Accuracy= " + \
#				  "{:.3f}".format(acc))
#
#	print("Optimization Finished!")
#
#	# Calculate accuracy for MNIST test images
#	print("Testing Accuracy:", \
#		sess.run(accuracy, feed_dict={X: mnist.test.images,
#									  Y: mnist.test.labels}))
#
