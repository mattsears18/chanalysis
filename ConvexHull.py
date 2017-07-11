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
from pylab import rcParams


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
        
# Set the default size of the plot figure to 12" width x 5" height
rcParams['figure.figsize'] = 10, 5

# Setup the figure and subplots
plt.close("all")
fig1, (ax1, ax2) = plt.subplots(1, 2, sharey=False)

# Increase whitespace between the subplots
fig1.subplots_adjust(wspace=0.35)

# Set Axes labels of the right subplot (the line graph)
ax2.set_xlabel('Time (milliseconds)')
ax2.set_ylabel('Convex Hull Area (%)')

# Set the axis limits for the right subplot
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 100])

# Set the title of the right subplot
ax2.title.set_text('Convex Hull Area Over Time')        

# Initialize the lines for the right subplot
areaLine, = ax2.plot([], [], label = 'Convex Hull Area')
avgLine, = ax2.plot([], [], linestyle='--', label = 'Average Area')

# Initialize the average line label
avgLabel = ax2.text(0, 0, '%', horizontalalignment='left', verticalalignment='center')

# Initialize the elapsed time label in the bottom right corner of the right subplot
timeLabel = ax2.text(0, 0, '', horizontalalignment='right', verticalalignment='bottom')

# Create the legend in the top right corner of the right subplot
ax2.legend(loc=1)
    
def update(frame):
    row = data.iloc[frame,:]
    
    if(not np.isnan(row['startRow'])):       
        plotPoints = points[int(row['startRow']):frame+1]
        plotPoints = plotPoints.as_matrix()
        
        # Calculate the convex hull
        hull = ConvexHull(plotPoints)
        
        # Save the hull and hullArea   
        data.set_value(frame, 'hull', hull)
        data.set_value(frame, 'hullArea', hull.volume*100)
        
        # get the frameRange
        frameRange = 'Fixations: ' + str(firstRow) + '-' + str(frame)
        
        # Draw the points in the left subplot
        ax1.cla()
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        ax1.set_xlabel('X Coordinate (normalized)')
        ax1.set_ylabel('Y Coordinate (normalized)')
        
        # Set the subplot title to frameRange
        ax1.title.set_text('Fixation #\'s: ' + str(int(row['startRow'])) +' - ' + str(frame) + '\nPeriod: ' + str(row['period'])[0:6] + ' milliseconds')
        ax1.plot(plotPoints[:,0], plotPoints[:,1], 'o')
        
        # Draw the convex hull in the left subplot
        for simplex in hull.simplices:
            ax1.plot(plotPoints[simplex, 0], plotPoints[simplex, 1], 'k-')
        
        # Draw AREA line graph in the right subplot
        areaLine.set_data(data.loc[firstRow:frame,'totaltime'], 
                          data.loc[firstRow:frame,'hullArea'])
        
        # Update the x axis limit to include the new data
        ax2.set_xlim([data.loc[firstRow, 'totaltime'],
                      data.loc[frame, 'totaltime']])
        
        # Update and draw the flat average line
        average = np.mean(data.loc[firstRow:frame,'hullArea'])
        avgLine.set_data([0, data.loc[frame, 'totaltime']],
                         [average, average])
        
        # Update and move the average line label
        avgLabel.set_text(str(average)[0:5] + '%')
        avgLabel.set_position((data.loc[frame, 'totaltime'], average))
        
        # Update the time in the bottom right corner of the right subplot
        timeLabel.set_position((data.loc[frame, 'totaltime'], 0))        
        timeLabel.set_text(str(row['totaltimesec'])[0:6] + ' seconds')


# Animate the plots
animation = animation.FuncAnimation(fig1,
                                    func=update,
                                    frames=range(firstRow, len(data)),
                                    fargs=None,
                                    interval=200,
                                    repeat=False)

# Make a timestamp for unique movie filenames
ts = time.time()
dt = datetime.datetime.fromtimestamp(ts).strftime('%y%m%d.%H%M%S')


animation.save('chull_animation' + str(dt) + '.mp4', fps=5,
               extra_args=['-vcodec', 'libx264'])