import tensorflow as tf
"""
constants
"""
print("="*80)
a = tf.constant(2)
b = tf.constant(3)
c = a+b

with tf.Session() as sess:
	print("a : {} || {}".format(sess.run(a),a))
	print("b : {} || {}".format(sess.run(b),b))
	print("a+b={} || {}".format(sess.run(c),c))
"""
placeholders
"""
print("="*80)

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

add = tf.add(a,b)
mul = tf.multiply(a,b)

with tf.Session() as sess:
	print("add : {}".format(sess.run(add, feed_dict={a:2, b:3})))
	print("mul : {}".format(sess.run(mul, feed_dict={a:2, b:3})))

"""
matmul
"""
print("="*80)

matrix1 = tf.constant([[3. , 3.]])
matrix2 = tf.constant([[2.],[2.]])

product = tf.matmul(matrix1,matrix2)

with tf.Session() as sess:
	res = sess.run(product)
	a = sess.run(matrix1)
	b = sess.run(matrix2)
	print("matmul : \n\ta : {}\n\tb : {}\na*b = {}".format(matrix1,matrix2,res)) 
