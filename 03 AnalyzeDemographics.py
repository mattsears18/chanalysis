#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 15:48:38 2017

@author: mattsears
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pylab import rcParams

filepath = 'results/resultsDemographics.xlsx'
imagePath = 'results/images/'

results = pd.read_excel(filepath)

# Set the default size of the plot figure to 10" width x 5" height
rcParams['figure.figsize'] = 10, 5

# Get rid of useless columns
del results['participant']
del results['pid']
del results['omarTime']
del results['cognitionHighLow']
del results['training']

def getCorr(test):
    data = results[[test[0], test[1]]]
    
    # Check for missing values and if found, delete the row
    nanCount = data.isnull().sum().sum()
    if(nanCount > 0):
        # Find the row with the missing value and remove it
        print('has NaN')
        data = data[np.isfinite(data[test[0]])]
        data = data[np.isfinite(data[test[1]])]
    
    x = data.iloc[:, 0]
    y = data.iloc[:, 1]
    
    # Calculate correlation and p-value
    r,p = stats.pearsonr(x, y)
    r2 = r**2
    n = len(x)
    return r,p,n

tests = []

for var in results:
    if 'HullArea' in var:
        for var2 in results:
            if not 'HullArea' in var2:
                tests.append((var, var2))
            
testResults = pd.DataFrame(columns = ['test', 'r2', 'r', 'p', 'n'])

for test in tests:
    r,p,n = getCorr(test)
    
    result = {'test': test, 'r2': r**2, 'r': r, 'p': p, 'n': n}
    
    testResults = testResults.append(result, ignore_index=True)
    
    print('')
    print('Test: ' + str(test))
    print('R^2 = ' + str(r**2) + ',  r = ' + str(r) + ',  p = ' + str(p) + ',  n = ' + str(n))
    






#### TESTY
#test = tests[179]
#getCorr(test)



def plotAndSave(x, y, pointLabel, xLabel, yLabel, title, filename):  
    
    data = pd.DataFrame([x.values, y.values]).transpose()
    
    # Check for missing values and if found, delete the row
    nanCount = data.isnull().sum().sum()
    if(nanCount > 0):
        # Find the row with the missing value and remove it
        print('has NaN')
        data = data[np.isfinite(data[0])]
        data = data[np.isfinite(data[1])]
    
    x = data.iloc[:, 0]
    y = data.iloc[:, 1]
    
    # Make the plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    plt.plot(x, y, 'o', label=pointLabel)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    #plt.title(title, fontweight='bold')
    
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
    
    plt.savefig((imagePath + '/Demographics/' + filename + '.png'))
    
    
x = results['compositeCognition']
y = results['participantAvgHullArea12000']



plotAndSave(x, y, 'Participant', 'Composite Spatial Cognition Score', 'Average Convex Hull Area (%) (12,000ms period)', 'title', 'Composite Cognition vs Average Convex Hull Area 12000')