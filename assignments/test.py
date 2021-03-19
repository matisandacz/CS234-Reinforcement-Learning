import numpy as np
import tensorflow as tf
import matplotlib.pyplot

"""
DeepMind - Introduction to Tensorflow.

Programs can be thought of as Computational Graphs. Abstract view of traditional
representation of programs.

Computational Graphs: Do not specify the order of operation, 
but the dependency between these operations.

-> Exploits at best parallel computation. 

Ops = Nodes of graph. Specify computation.
Tensors = Edges. Data that flows. Can be thought of as multidim arrays.

Tensorflow program 2 parts
* Definition of computational graph
* Execution of a subset of the computational graph

tf.Graph:
---------

Library for definition of graph. Uses tensors/ops.

* A tensor is a description of a multidimensional array.
It does not store/init any values/space in memory.

tf.Session 
----------
Carries the actual computation.

session.run([t1,t2,...,tn]). Executes and returns value of the tensor.

-> Will execute minimal number of nodes in graph
to evaluate the tensor. 

Variables
---------

* Can be used as any other tensor in comp.graphs
* Will mantain values across graph executions, 
until a new is value is assigned.
* Can be assigned values. Assignment is also an OP.

increment_op = tf.assign(v,v+1) ; v <- v+1

* Does not hold values until execution in a
Session. 

* Must be initialized before the Session begins.

Working with Data
-----------------

* We could define the OP tf.constant, with the
dataset. Not recomended. 

PlaceHolders : Feed data to graph at runtime.
------------

* Defined at graph declaration time.
* We can use them as tensors, but will 
take the values specified in the feed dictionary
provided to the Session at execution time.
* Can take different values at every execution
of the graph.
"""

#----------------------------------------------

"""
Example 1) Univariate Linear Regression.

Lets generate a small dataset for 1D LR.
Rule y=wx+b+e, e is random noise with gaussian distribution
(zero mean and unit variance).

"""

#Generate Data.
num_samples, w, b = 20, 0.5, 2
xs = np.asarray(range(num_samples))
ys = np.asarray([x*w+b+np.random.normal() for x in range(num_samples)])
matplotlib.pyplot.plot(xs,ys)
matplotlib.pyplot.show()

#Definition of the model.

class Linear(object):
	def __init__(self):
		self.w=tf.get_variable("w",dtype=tf.float32, shape=[], initializer=tf.zeros_initializer())
		self.b=tf.get_variable("b",dtype=tf.float32, shape=[], initializer=tf.zeros_initializer())
	def __call__(self,x):
		return self.w*x+self.b

"""Automatic Differentiation
#----------------------------
Compute gradients of tensors with respect to any variable.




"""
