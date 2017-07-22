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
from sklearn import preprocessing


# CONSTANTS
X_MAX = 0       # Width of reference image (set to 0 for automatic)
#X_MIN = 0       # TODO
Y_MAX = 0       # Height of reference image (set to 0 for automatic)
#Y_MIN = 0       # TODO
PERIODS = [3000]   # Time periods (in milliseconds) to calculate convex hull areas

# All other global vars
participantNums = range(1,21)
spools = range(1,11)





#########################
# Don't edit below here #
#########################





# Initialize the data variable as global
allData = 0

# Define the doCalculations function
def doCalculations(spool):
    for participantNum in participantNums:
        # Create a padded string for participantNum
        participantNumTxt = str(participantNum).zfill(2)
        
        for PERIOD in PERIODS:
            print('SPOOL: ' + str(spool))
            # Use the global data variable
            global allData
            # Create a padded string for spool
            spoolTxt = str(spool).zfill(2)
            
            # Only read data file if necessary
            if type(allData) is int:
                print('no data')
                # Data doesn't exist. Read the file.
                allData = pd.read_table("BeGaze Data/Raw Data/Participant " + participantNumTxt
                             + ".txt", delimiter = ',')
                    
                # Rename the columns
                allData.columns = ['totaltime', 'timestampHMS', 'category',
                                'index', 'x', 'y', 'aoi']
                
                # Keep only the rows for "Visual Intake"
                allData = allData.loc[allData['category'] == 'Visual Intake']
                    
                # Select only the x and y coordinates
                allPoints = allData.iloc[:, 4:6]
                
                # Define the getTimeSec function
                def getTimeSec(time):
                    return int(time)/1000
                
                # Calculate the totaltime in seconds for each row
                allData['totaltimesec'] = allData['totaltime'].apply(getTimeSec)
                
            else:
                # Data exists. Do NOT read the file.
                print('have data already')
                
            # Keep only the rows for the current spool (AOI)
            data = allData.loc[allData['aoi'] == 'Spool ' + str(spool)] 
            
            # Only proceed if there are fixations for the spool
            if(len(data.index) > 10):
        
                # Keep only the first row for each fixation
                data = data.drop_duplicates(subset = 'index', keep = 'first')
                data.reset_index(inplace = True)  
                
                
                # Normalize the points
                min_max_scaler = preprocessing.MinMaxScaler()
                
                # Normalize the coordinates
                data['x'] = min_max_scaler.fit_transform(data['x'])
                data['y'] = min_max_scaler.fit_transform(data['y'])
                
                # Get just the x and y points
                points = data.iloc[:, 5:7]
                
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
                        data.set_value(row, 'rowCount', row - startRow + 1)
                        data.set_value(row, 'period',
                                       times.iloc[row] - times.iloc[startRow])
                
                # Find the first data row with enough data to meet the PERIOD
                firstRow = data[data['startRow'] > 0].index.tolist()[0]
                        
                # Set the default size of the plot figure to 12" width x 5" height
                rcParams['figure.figsize'] = 10, 5
                
                # Setup the figure and subplots
                plt.close("all")
                fig1, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
                
                # Increase whitespace between the subplots
                fig1.subplots_adjust(wspace=0.35)
                
                # Set the figure title
                fig1.suptitle('Participant ' + participantNumTxt + ' - Spool '
                              + spoolTxt)
                
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
                avgLabel = ax2.text(0, 0, '%', horizontalalignment='left',
                                    verticalalignment='center')
                
                # Initialize the time label in the bottom right of the right subplot
                timeLabel = ax2.text(0, 0, '', horizontalalignment='right',
                                     verticalalignment='bottom')
                
                # Create the legend in the top right corner of the right subplot
                ax2.legend(loc=1)
                    
                def update(frame):            
                    row = data.iloc[frame,:]
                    
                    if(not np.isnan(row['startRow'])):
                        plotPoints = points[int(row['startRow']):frame+1]
                        
                        # Only continue if there are 3 unique points
                        if((row['rowCount'] > 2) & (len(plotPoints.drop_duplicates()) > 2)):
                            plotPoints = plotPoints.as_matrix()
                            
                            print('partic: ' + participantNumTxt
                                  + '  spool: ' + spoolTxt + '  PERIOD: '
                                  + str(PERIOD) + '  frame: ' + str(frame)
                                  + '  len:' + str(len(plotPoints))
                                  + '  enough points')
                        
                            # Enough points exist to calculate the hull
                            # Calculate the convex hull
                            hull = ConvexHull(plotPoints)
                            
                            # Save the hull and hullArea   
                            data.set_value(frame, 'hull', hull)
                            data.set_value(frame, 'hullArea', hull.volume*100)
                        
                        
                            # Draw the points in the left subplot
                            ax1.cla()
                            ax1.set_xlim([0, 1])
                            ax1.set_ylim([0, 1])
                            ax1.set_xlabel('X Coordinate (normalized)')
                            ax1.set_ylabel('Y Coordinate (normalized)')
                            
                            ax1.plot(plotPoints[:,0], plotPoints[:,1], 'o')
                            
                            # Draw the convex hull in the left subplot
                            for simplex in hull.simplices:
                                ax1.plot(plotPoints[simplex, 0],
                                         plotPoints[simplex, 1], 'k-')
                            
                        else:
                            print('partic: ' + participantNumTxt
                                  + '  spool: ' + spoolTxt + '  PERIOD: '
                                  + str(PERIOD) + '  frame: ' + str(frame)
                                  + '  len:' + str(len(plotPoints))
                                  + '  NOT ENOUGH POINTS!!!')
                            
                            # Not enough points to calculat the hull, set area to zero
                            data.set_value(frame, 'hullArea', 0)
                        
                        
                        # Draw AREA line graph in the right subplot
                        areaLine.set_data(data.loc[firstRow:frame,'totaltime'], 
                                          data.loc[firstRow:frame,'hullArea'])
                        
                        # get the frameRange
                        frameRange = 'Fixations: ' + str(firstRow) + '-' + str(frame)
                        
                        # Set the subplot title to frameRange
                        ax1.title.set_text('Fixation #\'s: '
                                           + str(int(row['startRow'])) + ' - '
                                           + str(frame) + '\nPeriod: ' +
                                           str(row['period'])[0:6] + ' milliseconds')
                        
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
                        
                        # Update the time in the bottom right corner of the right plot
                        timeLabel.set_position((data.loc[frame, 'totaltime'], 0))        
                        timeLabel.set_text(str(row['totaltimesec'])[0:6] + ' seconds')
                
                
                # Animate the plots
                anim = animation.FuncAnimation(fig1,
                                                    func=update,
                                                    frames=range(firstRow, len(data)),
                                                    fargs=None,
                                                    interval=200,
                                                    repeat=False)
                
                # Make a timestamp for unique movie filenames
                ts = time.time()
                dt = datetime.datetime.fromtimestamp(ts).strftime('%y%m%d.%H%M%S')
                
                
                anim.save('animations/' + str(PERIOD) + '_participant'
                          + participantNumTxt + '_spool' + spoolTxt + '_'
                          + str(dt) + '.mp4', fps=5,
                          extra_args=['-vcodec', 'libx264'])

# Finally, do the calculations
for spool in spools:
    doCalculations(spool)
    
#doCalculations(6)