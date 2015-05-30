import numpy as np
import pydot
import theano
import theano.tensor as T

x = T.vector()
y = T.vector()
z = x + x
z = z + y
f = theano.function([x, y], z)
a=f(np.ones((3,)), np.ones((3,)))
theano.printing.pydotprint(z, outfile="symbolic_graph_unopt.png", var_with_name_simple=True)
theano.printing.pydotprint(f, outfile="symbolic_graph_opt.png", var_with_name_simple=True)
print(a)
print(np.ones(3,).shape)