
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import pylab as pl  
import random
import os

###############
# plot result #
###############
def my_plot(proTimeList, scoreList, fileNameList, folder):
    # first plot: score vs iteration
    plt.figure()
    plt.grid(True)
    for score, fileName in zip(scoreList, fileNameList):
        x = range(min(len(score), 40))
        plt.plot(x, np.array(score)[x], marker='o', linestyle='-', linewidth=2.0, label=fileName)
    plt.xlabel('Epochs')
    plt.ylabel('Average score per episode')
    plt.title(folder)
    plt.legend(loc=4)
    #plt.savefig(folder+'.pdf', format='pdf')
    plt.show()

    # second plot: score vs. training time
    plt.figure()
    plt.grid(True)
    index = 0
    for score, fileName in zip(scoreList, fileNameList):
        x = range(min(len(score), 40))
        plt.plot(np.array(proTimeList[index])[x], np.array(score)[x], marker='o', linestyle='-', linewidth=2.0, label=fileName)
        index += 1
    plt.xlabel('Time/h')
    plt.ylabel('Average score per episode')
    plt.title(folder)
    plt.legend(loc=4)
    #plt.savefig(folder+'.pdf', format='pdf')
    plt.show()

##############
# parse data #
##############
def parser(folder, word1, word2, rewardIndex, timeIndex):
    # create score and name lists
    os.chdir('C:\Users\kongl\Documents\Git\CS234Project\exp_results')
    allFiles = os.listdir(folder)
    timeList =[]
    scoreList = []
    fileNameList = []
    for fileName in allFiles:
        os.chdir('C:\Users\kongl\Documents\Git\CS234Project\exp_results' + '\\' + folder)
        file = open(fileName, 'r')
        score = []
        time = []
        for line in file:
            if word1 in line:
                splitLine = line.split()
                value = splitLine[rewardIndex]
                score.append(float(value))
                t = splitLine[timeIndex]
                time.append(t)
            if word2 in line:
                splitLine = line.split()
                value = splitLine[rewardIndex]
                score.append(float(value))
                t = splitLine[timeIndex]
                time.append(t[0:8])
        file.close()
        timeList.append(time)
        scoreList.append(score)
        fileNameList.append(os.path.splitext(fileName)[0])
    return timeList, scoreList, fileNameList

#################
# process timer #
#################
def getHour(time_str):
    h, m, s = time_str.split(':')
    return float(h) + float(m)/60 + float(s)/3600

def processTime(timeList):
    proTimeList = []
    for time in timeList:
        proTime = []
        dayIndex = []
        for i, t in enumerate(time):
            hr = getHour(t)
            if i != 0 and time[i-1] > time[i]:
                dayIndex.append(i)
            proTime.append(hr)
        for i in dayIndex:
            proTime = np.array(proTime)
            proTime[i:] += 24.0
        proTime = np.array(proTime) - proTime[0] # minus first index
        proTime = list(proTime)
        proTimeList.append(proTime)
    return proTimeList

#############
# main code #
#############
if __name__ == "__main__":
    # parameters needs to be changed based on different environment
    folder = 'IceHockey'
    word1 = 'mean_score:'
    word2 = 'Average reward:'
    rewardIndex = 4
    timeIndex = 1
    timeList, scoreList, fileNameList = parser(folder, word1, word2, rewardIndex, timeIndex)
    proTimeList = processTime(timeList)
    my_plot(proTimeList, scoreList, fileNameList, folder)


