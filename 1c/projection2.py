#-*- coding: UTF-8 -*-
# Source: http://neuralnetworksanddeeplearning.com/chap1.html
# You need only train and test sets. Simply disregard the validation set.

import numpy as np
from matplotlib.pyplot import *

figure(); hold(True)
r = 1
linestyle = ['b-','k-','m-','r-','y-']
p_values = (1,2,10)
for i,p in enumerate(p_values):
    x = np.arange(-r,r+1e-5,1/128.0)
    y = (r**p - (abs(x)**p))**(1.0/p)
    y = zip(y, -y)
    plot(x, y, linestyle[i], label=str(i))
axis('equal')
show()