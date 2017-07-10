#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 14:31:56 2017

@author: Matthew H. Sears
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.spatial import ConvexHull
import time
import datetime


# CONSTANTS
X_MAX = 100       # Width of reference image
Y_MAX = 100       # Height of reference image
PERIOD = 3000     # Time period (in milliseconds) to calculate convex hull area


# Read in the data file (note that header names are important at this time)
data = pd.read_excel('DummyData.xlsx', header = 0)

# Select only the x and y coordinates
points = data.iloc[:, :2]

# Normalize coordinates based upon image size
imageSize = [X_MAX, Y_MAX]
points = points/imageSize

# Define the getStartRow function
def getStartRow(times, finishRow, period):
    startRow = False
    for row in range(finishRow, -1, -1):
        if times.iloc[finishRow] - times.iloc[row] > period:
            startRow = row + 1
            break
    return startRow
   
# Calculate time steps
times = data['totaltime']

for row in range(len(times) - 1, -1, -1):
    startRow = getStartRow(times, row, PERIOD)
    if(startRow):
        data.set_value(row, 'startRow', startRow)
        data.set_value(row, 'rowCount', row - startRow)
        data.set_value(row, 'period', times.iloc[row] - times.iloc[startRow])

# Find the first data row with enough data to meet the PERIOD
firstRow = data[data['startRow'] > 0].index.tolist()[0]
        
# Setup the plot
fig1 = plt.figure()
ax = plt.axes(xlim=(0, 1), ylim=(0, 1))
    
def update(frame):
    row = data.iloc[frame,:]
    
    if(not np.isnan(row['startRow'])):        
        plotPoints = points[int(row['startRow']):frame+1]
        plotPoints = plotPoints.as_matrix()
        
        # Calculate the convex hull
        hull = ConvexHull(plotPoints)
        
        # Save the hull and hullArea   
        data.set_value(frame, 'hull', hull)
        data.set_value(frame, 'hullArea', hull.volume)
        
        plt.clf()
        plt.axes(xlim=(0, 1), ylim=(0, 1))
        plt.title(str(row['totaltimesec'])[0:6] + ' seconds')
        plt.plot(plotPoints[:,0], plotPoints[:,1], 'o')
        
        for simplex in hull.simplices:
            plt.plot(plotPoints[simplex, 0], plotPoints[simplex, 1], 'k-')
        
    else:
        print('no plot points!')


# TODO - set range to start at first data row with hullArea, end at len(data)
animation = animation.FuncAnimation(fig1, func=update, frames=range(firstRow, 200),
                                    fargs=None, interval=200)

# Make a timestamp for unique movie filenames
ts = time.time()
dt = datetime.datetime.fromtimestamp(ts).strftime('%y%m%d.%H%M%S')

animation.save('chull_animation' + str(dt) + '.mp4', fps=5, extra_args=['-vcodec', 'libx264'])