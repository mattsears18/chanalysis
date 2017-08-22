#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 14:42:48 2017

@author: mattsears
"""

pt = results['partTime']

import numpy as np
import matplotlib.pyplot as plt

# the histogram of the data
n, bins, patches = plt.hist(pt, 50, normed=1, alpha=0.75)

plt.xlabel('Part Duration (s)')
plt.ylabel('Probability')
#plt.axis([40, 160, 0, 0.03])
plt.grid(True)

plt.show()

sum(pt)/len(pt)

np.median(pt)
z = [i for i in pt if i >= 50]

32/739