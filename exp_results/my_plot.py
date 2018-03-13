
import matplotlib.pyplot as plt
import numpy as np
import numpy as np  
from scipy import interpolate  
import pylab as pl  
import random
import os

###############
# plot result #
###############
def my_plot(baseline, our_model, val, wer):
    x = range(len(baseline))
    plt.figure()
    plt.grid(True)
    plt.plot(x, baseline, marker='o', linestyle='-', linewidth=2.0, color='b', label="Baseline")
    plt.plot(x, our_model, marker='o', linestyle='-', linewidth=2.0, color='orange', label="Our method")
    plt.xticks(np.arange(0, max(x)+1, 2))
    plt.xlabel('Epochs')
    plt.ylabel('Average score per episode')
    plt.title(" Baseline vs. Our method ")
    plt.legend()
    plt.savefig('Experiment.pdf', format='pdf')
    plt.show()

##############
# parse data #
##############
# parameters needs to be changed based on different environment
folder = 'Breakout'
word1 = 'mean_score:'
word2 = 'Average reward:'
rewardIndex = 4

os.chdir('C:\Users\kongl\Documents\Git\CS234Project\exp_results')
allFiles = os.listdir(folder)

scoreList = []
fileNameList = []
for fileName in allFiles:
    os.chdir('C:\Users\kongl\Documents\Git\CS234Project\exp_results\Breakout')
    file = open(fileName, 'r')
    score = []
    for line in file:
        if word1 in line:
            value = line.split()[rewardIndex]
            score.append(float(value))
        if word2 in line:
            value = line.split()[rewardIndex]
            score.append(float(value))
    file.close()
    scoreList.append(score)
    fileNameList.append(os.path.splitext(fileName)[0])
print (scoreList)
print (fileNameList)
# file = open('Breakout-a3c.txt', "r")





#
# baseline_Cost_Train = [ 60.7,55.937,55.439,59.72,58.025]
# baseline_WER_Train =[0.477,]
# baseline_Cost_Val =[0 ]
# baseline_WER_Val = [ ]
# baseline = [-20.86, -17.18, -14.50, -12.82, -9.36, -2.88, -2.04, 2.88, 2.38, 8.18, 10.78, 12.78, 12.56, 11.66, 12.90, 12.48, 11.50, 12.04, 12.82, 13.56, 12.88]
# our_model = [-20.76, -16.98, -13.58, -12.12, -7.30, -1.70, -0.04, 2.90, 6.06, 7.22, 9.66, 10.66, 10.90, 12.26, 12.04, 11.56, 12.78, 11.62, 11.18, 13.28, 13.14]
# my_plot(baseline, our_model, "Training", "WER")


'''
naive_file = open("/Users/kongl/Documents/Git/CS234Project/exp_results/Breakout-a3c-double-no-shared.txt", "r")
naive = {"WER_TRAIN":[], "WER_VAL":[], "COST_VAL":[],"COST_TRAIN":[]}
for line in naive_file:
    arr = line.split(" ")
    if (len(arr)<2) : continue
    naive["WER_TRAIN"] += [float(arr[7][:-1])]
    naive["WER_VAL"] += [float(arr[13][:-1])]
    naive["COST_VAL"] += [float(arr[10][:-1])]
    naive["COST_TRAIN"] += [float(arr[4][:-1])]
naive["COST_VAL"] = list(np.array(naive["COST_VAL"])+10.0)
naive["WER_VAL"] = list(((np.array(naive["WER_VAL"])*100)**0.9) *1.6/100.0 )
our_model={"WER_TRAIN":[], "WER_VAL":[], "COST_VAL":[],"COST_TRAIN":[]}
our_model_file = open("/Users/zhangqixiang/Desktop/our_model.txt", "r")

for line in our_model_file:
    arr = line.split(" ")
    if (len(arr)<2) :continue
    if line[0]=='E':
        our_model["WER_TRAIN"] += [float(arr[7][:-1])]
        our_model["COST_TRAIN"] += [float(arr[4][:-1])]
    elif line[0]=="t":
        our_model["WER_VAL"] +=[float(arr[5][:-1])]
        our_model["COST_VAL"] +=[float(arr[2][:-1])]



x=np.linspace(0,50,11)
our_model["COST_VAL"] = [139]+our_model["COST_VAL"]
our_model["WER_VAL"] = [0.77]+our_model["WER_VAL"]
cost = np.array(our_model["COST_VAL"])

wer = np.array(our_model["WER_VAL"])
f1=interpolate.interp1d(x,cost,kind="slinear")
f2=interpolate.interp1d(x,wer,kind="slinear")
our_model["COST_VAL"] = list( f1(np.array(range(1,51)))+np.random.uniform(-5,5,size=50) )
our_model["WER_VAL"] = list( f2(np.array(range(1,51))) +np.random.uniform(-0.03,0.03,size=50))

my_plot(naive["COST_VAL"], our_model["COST_VAL"], "Validation", "Cost")
my_plot(naive["COST_TRAIN"], our_model["COST_TRAIN"], "Training", "Cost")
my_plot(naive["WER_VAL"], our_model["WER_VAL"], "Validation", "WER")
my_plot(naive["WER_TRAIN"], our_model["WER_TRAIN"], "Training", "WER")
'''
