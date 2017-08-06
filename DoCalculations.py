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


periods = [15000]  # Time periods (in ms) to calculate convex hull areas
participantNums = range(1, 21)
dwgs = range(1, 11)

partThresh = 5000  # DWG Viewing Time Threshold (ms)
partPointMin = 20  # Minimum # of points required to make a part

filePrefix = "BeGaze Data/Raw Data/Participant "
fileSuffix = ".txt"

# Convex hull areas smaller than detailThresh are considered "detailed viewing"
# while convex hull areas larger than detailThresh are considered
# "distributed viewing." This will be used to calculate a propDetail, which is
# the propoirtion of time spent in "detailed viewing"
detailThresh = 0.10 # Set threshold to 10%





#########################
# Don't edit below here #
#########################





# Initialize results
results = pd.DataFrame(columns = ['period', 'participant', 'dwg', 'part',
                                  'partAvgHullArea', 'partTime',
                                  'dwgAvgHullArea', 'dwgTime',
                                  'participantAvgHullArea', 'participantTime'])


# Remove all categories except for "Visual Intake"
def getVisualIntakes(data):
    data = data.loc[data['category'] == 'Visual Intake']
    return data


# Remove all AOI's that include spools (no '-' or 'white space')
def getSpools(data):
    data = data[data['aoi'].str.contains("Spool")]
    return data


def getFloatCoords(data):
    # Convert coordinates to floats
    data = data.astype(dtype = {'x': np.float64, 'y': np.float64})
    return data


# Remove combine rows that have the same binocular index
def getFirstIndices(data):
    data = data.drop_duplicates(subset = 'index', keep = 'first')
    return data


# Read in the raw BeGaze data files
def getData(filePrefix, fileSuffix, participantNumTxt):
    # Data doesn't exist. Read the file.
    data = pd.read_table(filePrefix + participantNumTxt + fileSuffix,
                            delimiter = ',')
    
    # Rename the columns
    data.columns = ['recordtime', 'timestamp', 'category',
                       'index', 'x', 'y', 'aoi']
        
    return data


def getCleanData(data):
    data = getVisualIntakes(data)
    data = getSpools(data)
    data = getFloatCoords(data)
    return data


def getDwgData(data, dwgNum):
    return data.loc[data['aoi'] == 'Spool '
                    + str(dwgNum)].sort_values(by = ['timestamp'])


def getScaledCoordinates(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    
    # Normalize the coordinates
    data['x'] = min_max_scaler.fit_transform(data['x'].values.reshape(-1,1))
    data['y'] = min_max_scaler.fit_transform(data['y'].values.reshape(-1,1))
    
    return data


def addDurationsCol(data):
    for i, row in data.iterrows():
        if i > 0:            
            duration = data.iloc[i,:]['recordtime'] - data.iloc[i - 1,:]['recordtime']            
        else:
            # Just use the timestamp for the very first row
            duration = data.iloc[i,:]['recordtime']
            
        data.set_value(i, 'duration', duration)
        
    return data


# Split the data into parts based upon the duration of each fixation
# If a duration exceeds partThresh (global variable defined by user) then split
def getDwgParts(data, partThresh, partPointMin):                            
    partNum = 1
                    
    # Assign part numbers
    for i, row in data.iterrows():
        if data.iloc[i,:]['duration'] > partThresh:
            data.set_value(i, 'part', 0)
            partNum += 1
        
        else:
            data.set_value(i, 'part', partNum)
        
    totalParts = partNum
    
    data = data[data['part'] > 0]
    
    # Make an array of parts
    parts = []
    
    # Append the lists of each part
    for i in range(1, totalParts + 1):
        thisPart = data[data['part'] == i]
        thisPart.reset_index(drop=True, inplace=True)
        del thisPart['part']
        if(len(thisPart) >= partPointMin):
            parts.append(thisPart)
        
    return parts


def getPartTime(data):
    minTime = data['recordtime'].min()
    
    def pt(x):
        return x['recordtime'] - minTime
    
    return data.apply(pt, axis=1)


def getStartRow(data, finishRow, period):
    startRow = False
    for row in range(finishRow, -1, -1):
        if data.iloc[finishRow]['partTime'] - data.iloc[row]['partTime'] > period:
            startRow = row + 1
            break
    return startRow


def getRowCountStartPeriod(data, period):    
    for row in range(len(data) - 1, -1, -1):
        startRow = getStartRow(data, row, period)
        if(startRow):
            data.set_value(row, 'startRow', startRow)
            data.set_value(row, 'rowCount', row - startRow + 1)
            data.set_value(row, 'period',
                           data.iloc[row]['partTime'] - data.iloc[startRow]['partTime'])
        else:
            data.set_value(row, 'startRow', np.nan)
            data.set_value(row, 'rowCount', np.nan)
            data.set_value(row, 'period', np.nan)
    return data


# Calculate the convex hulls and hullAreas
def getConvexHulls(data):
    for i, row in data.iterrows():
        if(not np.isnan(row['startRow'])):
            # Get just the points for calculating the convex hull
            points = data.iloc[int(row['startRow']):i+1][['x', 'y']]
            
            if((int(row['rowCount']) > 2) & (len(points.drop_duplicates()) > 2)):
            
                # Calculate the convex hull and save it
                hull = ConvexHull(points)
                data.set_value(i, 'hull', hull)
                data.set_value(i, 'hullArea', hull.volume*100)
            else:
                data.set_value(i, 'hullArea', 0)
                
    return data

def getPlotPoints(data, frame):
    plotPoints = []
    
    startRow = data.iloc[frame]['startRow']
    if(not np.isnan(startRow)):    
        plotPoints = data.iloc[int(startRow):frame+1][['x', 'y']]
        plotPoints.reset_index(inplace=True)
        del plotPoints['index']
    
    return plotPoints
    
    
def updatePlot(frame, data, startFrame, period, participantNumTxt,
               dwgNumTxt, partNumTxt):
    global finalTime
    global average
    
    print('period:' + str(period) + ' participant:' + participantNumTxt
          + ' dwg:' + dwgNumTxt + ' part:' + partNumTxt
          + ' frame: ' + str(frame))
    
    row = data.iloc[frame]
    
    if(row['rowCount'] > 2):
        plotPoints = getPlotPoints(data, frame)
        if(len(plotPoints) > 2):
            if(len(plotPoints.drop_duplicates()) > 2):
                plotPoints = plotPoints.as_matrix()
                # Plot the points! Draw the points in the left subplot
                ax1.cla()
                ax1.set_xlim([0, 1])
                ax1.set_ylim([0, 1])
                ax1.set_xlabel('X Coordinate (normalized)')
                ax1.set_ylabel('Y Coordinate (normalized)')
                
                ax1.plot(plotPoints[:,0], plotPoints[:,1], 'o')
                
                for simplex in row['hull'].simplices:
                    ax1.plot(plotPoints[simplex, 0],
                             plotPoints[simplex, 1], 'k-')

                # Set the subplot title to frameRange
                ax1.title.set_text('Fixation #\'s: '
                                   + str(int(row['startRow'])) + ' - '
                                   + str(frame) + '\nPeriod: ' +
                                   str(row['period'])[0:6] + ' milliseconds')
                
                # Update Right plot
                if(frame > startFrame):
                    # Update the x axis limit to include the new data
                    ax2.set_xlim(left = data.loc[startFrame, 'partTime'],
                                 right = data.loc[frame, 'partTime'])
                    
                    # Draw AREA line graph in the right subplot
                    areaLine.set_data(data.loc[startFrame:frame, 'partTime'], 
                                      data.loc[startFrame:frame, 'hullArea'])
                    
                    # Update and draw the flat average line
                    average = np.mean(data.loc[startFrame:frame,'hullArea'])
                    avgLine.set_data([0, data.loc[frame, 'partTime']],
                                     [average, average])
                    
                    # Update and move the average line label
                    avgLabel.set_text(str(average)[0:5] + '%')
                    avgLabel.set_position((data.loc[frame, 'partTime'],
                                           average))
                    
                    # Update the time in the bottom right corner of the plot
                    timeLabel.set_position((data.loc[frame, 'partTime'], 0))
                    timeLabel.set_text(str(row['partTime']/1000)[0:6]
                                        + ' seconds')
                    
                    finalTime = row['partTime']/1000
        
def getStartFrame(data):
    for i, row in data.iterrows():
        if(not np.isnan(row['startRow'])):
            return i
        

def plotAndSave(data, period, participantNumTxt, dwgNumTxt, partNumTxt):    
    global fig1
    global ax1
    global ax2
    global areaLine
    global avgLine
    global avgLabel
    global timeLabel
    global finalTime
    global average
    
    # Set the default size of the plot figure to 12" width x 5" height
    rcParams['figure.figsize'] = 10, 5
    
    # Setup the figure and subplots
    plt.close("all")
    fig1, (ax1, ax2) = plt.subplots(1, 2, sharey=False)

    # Increase whitespace between the subplots
    fig1.subplots_adjust(wspace=0.35)
    
    # Set the figure title
    fig1.suptitle('Participant ' + participantNumTxt + ' - DWG ' + dwgNumTxt
                  + ' - Part ' + partNumTxt, y=1)
    
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
    
    # Make a timestamp for unique movie filenames
    ts = time.time()
    dt = datetime.datetime.fromtimestamp(ts).strftime('%y%m%d.%H%M%S')
    
    startFrame = getStartFrame(data)
    
    finalTime = 0
    average = 0
    
    # Animate the plot
    anim = animation.FuncAnimation(fig1,
                                   func=updatePlot,
                                   frames=range(startFrame, len(data)),
                                   fargs=(data, startFrame, period,
                                          participantNumTxt, dwgNumTxt,
                                          partNumTxt),
                                   interval=200,
                                   repeat=False)
    
    anim.save('results/animations/' + str(period) + '_participant'
              + participantNumTxt + '_dwg' + dwgNumTxt + '_part' + partNumTxt
              + '_' + str(dt) + '.mp4', fps=5,
              extra_args=['-vcodec', 'libx264'])
    

def doCalculations(periods, participantNums, dwgs, partThresh, partPointMin,
                   filePrefix, fileSuffix):
    
    global results
    
    # Do everything for each period
    for period in periods:
        
        # Do everything for each participant        
        for participantNum in participantNums:            
            participantNumTxt = str(participantNum).zfill(2)

            # Get the participant data
            participantData = getData(filePrefix, fileSuffix,
                                      participantNumTxt)
            # Clean the participant data
            participantData = getCleanData(participantData)

            # Do everything for each drawing
            for dwgNum in dwgs:
                dwgNumTxt = str(dwgNum).zfill(2)
                
                dwgData = getDwgData(participantData, dwgNum)
                
                dwgData = getFirstIndices(dwgData)
                
                dwgData.reset_index(inplace = True)
                
                dwgData = getScaledCoordinates(dwgData)
                
                dwgData = addDurationsCol(dwgData)
                
                dwgParts = getDwgParts(dwgData, partThresh, partPointMin)
                
                # Do everything for each drawing part
                for i in range(0, len(dwgParts)):
                    partData = dwgParts[i]
                    
                    partNum = i+1
                    partNumTxt = str(partNum).zfill(2)
                    
                    partData['partTime'] = getPartTime(partData)
                    
                    partData = getRowCountStartPeriod(partData, period)
                    
                    # Don't calculate convex hulls unless there's enough data
                    if(partData['startRow'].nunique() > 0):
                        partData = getConvexHulls(partData)
                        
                        plotAndSave(partData, period, participantNumTxt, dwgNumTxt,
                                    partNumTxt)
                           
                        # Append this result to results
                        result = {'period': period,
                                  'participant': participantNum,
                                  'dwg': dwgNum,
                                  'part': partNum,
                                  'partAvgHullArea': average,
                                  'partTime': finalTime}
                        
                        results = results.append(result, ignore_index=True)
               
    
    # Change dtype to integers            
    results = results.astype(dtype = {'period': np.int, 'participant': np.int,
                                      'dwg': np.int, 'part': np.int})
    
    # Make a timestamp for unique movie filenames
    ts = time.time()
    dt = datetime.datetime.fromtimestamp(ts).strftime('%y%m%d.%H%M%S')
    
    # Write results to excel file
    writer = pd.ExcelWriter('results/results.xlsx',
                            engine='xlsxwriter')
    
    results.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    
    
    # Write results to excel file but save name with current date and time
    writerBackup = pd.ExcelWriter('results/results' + str(dt) + '.xlsx',
                            engine='xlsxwriter')
    
    results.to_excel(writerBackup, sheet_name='Sheet1')
    writerBackup.save()            

# Finally, call doCalculations
doCalculations(periods, participantNums, dwgs, partThresh, partPointMin,
               filePrefix, fileSuffix)














#########
######### for testing only
#########

period=15000
participantNum=1
dwgNum=2
i=2
data=partData
frame=12
startFrame=7

#########
######### for testing only
#########