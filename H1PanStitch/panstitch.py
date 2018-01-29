# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 12:57:53 2018

@author: Vishwakarma
"""

import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print("\n")
# Print in session
print(sess.run(hello), "\n")
a = tf.constant(10)
b = tf.constant(32)
print("Result: ", sess.run(a + b), "\n")

sess.close()
