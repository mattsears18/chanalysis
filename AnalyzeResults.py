#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 08:43:25 2017

@author: mattsears
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

filepath = 'results/results.xlsx'
imagePath = 'results/images/'

results = pd.read_excel(filepath)





#########################
# Don't edit below here #
#########################





# Filter Period
#results = results.loc[results['period'] == 12000]
# Filter Participants
#results = results.loc[results['participant'].isin([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])]
# Filter Drawings
#results = results.loc[results['dwg'].isin([1,2,3,4,5,6,7,8,9,10])]


### DRAWING STATS
groupedDwg = results.groupby(['period', 'participant', 'dwg'])

resultsDwg = pd.DataFrame(columns = ['period', 'participant', 'dwg',
                                     'dwgAvgHullArea', 'dwgTime', 'dwgTimeLn'])


# Calculate the sum, given data and a column label
def getSum(data, col):
    return data[col].sum()

# calculate the product of columns by row, then sum them up
def getWeightedAverage(data, value_col, weight_col):
    value_sum = 0
    weight_sum = 0
    
    for i, row in data.iterrows():
        value_sum += row[value_col] * row[weight_col]
        weight_sum += row[weight_col]
        
    average = value_sum / weight_sum
    return average


# Fill in dwgTime and dwgAvgHullArea's
for name, group in groupedDwg:
    global period
    global periodSec
    period = name[0]
    periodTxt = str(int(period))
    periodSec = int(period/1000)
    participantNum = name[1]
    dwgNum = name[2]
    
    dwgTime = getSum(group, 'viewingTime')
    dwgTimeLn = np.log(dwgTime)
    
    dwgAvgHullArea = getWeightedAverage(group, 'viewingAvgHullArea', 'viewingTime')
    
    result = {'period': period, 'participant': participantNum,
              'dwg': dwgNum, 'dwgAvgHullArea': dwgAvgHullArea,
              'dwgTime': dwgTime, 'dwgTimeLn': dwgTimeLn}
    
    for i, row in group.iterrows():      
        results.set_value(i, 'dwgTime', dwgTime)
        results.set_value(i, 'dwgAvgHullArea', dwgAvgHullArea)
    
    resultsDwg = resultsDwg.append(result, ignore_index=True)
    
    
    
### PARTICIPANT STATS
groupedParticipant = results.groupby(['period', 'participant'])

resultsParticipant = pd.DataFrame(columns = ['period', 'participant', 'dwg',
                                     'participantAvgHullArea',
                                     'participantTime'])

# Fill in participantTime and participantAvgHullArea's
for name, group in groupedParticipant:
    period = name[0]
    participantNum = name[1]
    
    participantTime = getSum(group, 'viewingTime')
    
    participantAvgHullArea = getWeightedAverage(group, 'viewingAvgHullArea',
                                                'viewingTime')
    
    result = {'period': period, 'participant': participantNum,
              'dwg': dwgNum, 'participantAvgHullArea': participantAvgHullArea,
              'participantTime': participantTime}
    
    for i, row in group.iterrows():      
        results.set_value(i, 'participantTime', participantTime)
        results.set_value(i, 'participantAvgHullArea', participantAvgHullArea)
    
    resultsParticipant = resultsParticipant.append(result, ignore_index=True)
  
"""
# Save new results
writer = pd.ExcelWriter('results/results.xlsx',
                        engine='xlsxwriter')

results.to_excel(writer, sheet_name='Sheet1')
writer.save()
"""

# Save new resultsParticipant
writer = pd.ExcelWriter('results/resultsParticipant.xlsx',
                        engine='xlsxwriter')

resultsParticipant.to_excel(writer, sheet_name='Sheet1')
writer.save()

# Save new resultsDwg
writer = pd.ExcelWriter('results/resultsDwg.xlsx',
                        engine='xlsxwriter')

resultsDwg.to_excel(writer, sheet_name='Sheet1')
writer.save()

    
def plotAndSave(x, y, pointLabel, xLabel, yLabel, title, filename):    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    plt.plot(x, y, 'o', label=pointLabel)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title, fontweight='bold')
    
    # Plot the best fit line
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)),
             linestyle='--', label='Pearson\'s Correlation')
    
    plt.legend(loc=1)
    
    # Calculate correlation and p-value
    r,p = stats.pearsonr(x, y)
    r2 = r**2
    n = len(x)
    
    rLabel = plt.text(.99, .80, 'Pearson\'s r = ' + str(r)[0:7], horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)
    r2Label = plt.text(.99, .75, 'R² = ' + str(r2)[0:7], horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)
    pLabel = plt.text(.99, .70, 'p-value = ' + str(p)[0:6], horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)
    nLabel = plt.text(.99, .65, 'n = ' + str(n), horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)
    
    plt.savefig((imagePath + filename))



plt.close('all')
"""
# Plot the Participant Results
plotAndSave(x = resultsParticipant['participantAvgHullArea'],
            y = resultsParticipant['participantTime'],
            pointLabel = 'Participant',
            xLabel = 'Average Convex Hull Area (' + str(periodSec) + ' second period)',
            yLabel = 'Total Time to Completion (s) (10 drawings)',
            title = 'Results by Participant\n(All Participants) (' + str(periodSec) + ' second Period)',
            filename = periodTxt + '_byParticipant_allDrawings.png')
"""

# Plot the Drawing Results
plotAndSave(x = resultsDwg['dwgAvgHullArea'],
            y = resultsDwg['dwgTime'],
            pointLabel = 'Participant-Drawing',
            xLabel = 'Average Convex Hull Area (' + str(periodSec) + ' second period)',
            yLabel = 'Total Time to Completion (s) (Whole Drawing)',
            title = 'Results by Drawing\n(All Drawings, All Participants) (' + str(periodSec) + ' second Period)',
            filename = periodTxt + '_byDrawing_all.png')

"""
# Plot the DrawingViewing Results
plotAndSave(x = results['viewingAvgHullArea'],
            y = results['viewingTime'],
            pointLabel = 'Drawing Viewing',
            xLabel = 'Average Convex Hull Area (' + str(periodSec) + ' second period)',
            yLabel = 'Time to Completion (s) (Drawing Viewing)',
            title ='Results by Drawing Viewing\n(All Participants, All Drawings) (' + str(periodSec) + ' second Period)',
            filename = periodTxt + '_byDrawingViewing_all.png')


# Make a plot for each participant
participantNums = results.participant.unique()

for participantNum in participantNums:
    participantNumTxt = str(participantNum).zfill(2)
    
    # Plot the Participant Results
    plotAndSave(x = resultsDwg[resultsDwg['participant'] == participantNum]['dwgAvgHullArea'],
                y = resultsDwg[resultsDwg['participant'] == participantNum]['dwgTime'],
                pointLabel = 'Drawing',
                xLabel = 'Average Convex Hull Area (' + str(periodSec) + ' second period)',
                yLabel = 'Total Time to Completion (s) (10 drawings)',
                title = 'Results by Participant\n(Participant ' + participantNumTxt + ', All Drawings) (' + str(periodSec) + ' second Period)',
                filename = periodTxt + '_byParticipant_' + participantNumTxt + '.png')


# Make a plot for each drawing
dwgNums = results.dwg.unique()

for dwgNum in dwgNums:
    dwgNumTxt = str(dwgNum).zfill(2)
        
    # Plot the Participant Results
    plotAndSave(x = resultsDwg[resultsDwg['dwg'] == dwgNum]['dwgAvgHullArea'],
                y = resultsDwg[resultsDwg['dwg'] == dwgNum]['dwgTime'],
                pointLabel = 'Participant',
                xLabel = 'Average Convex Hull Area (' + str(periodSec) + ' second period)',
                yLabel = 'Time to Completion (s)',
                title = 'Results by Drawing\n(Drawing ' + dwgNumTxt + ', All Participants) (' + str(periodSec) + ' second Period)',
                filename = periodTxt + '_byDrawing_' + dwgNumTxt + '.png')
    

plt.close('all')

"""