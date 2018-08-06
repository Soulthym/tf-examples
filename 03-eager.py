from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf

print("Eager")
tf.enable_eager_execution()
tfe = tf.contrib.eager

a = tf.constant(2)
print("a : {}".format(a))

b = tf.constant(3)
print("b : {}".format(b))

print("a+b = {}".format(a+b))

print("a*b = {}".format(a*b))

a = tf.constant([[2.,1.],[1.,0.]], dtype=tf.float32)
b = np.array([[3.,0.],[5.,1.]], dtype=np.float32)

print("a : {}".format(a))
print("b : {}".format(b))

print("a+b = {}".format(a+b))

print("="*80)

for i in range(a.shape[0]):
	for j in range(a.shape[1]):
		print(a[i][j])
