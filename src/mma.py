import os
import numpy as np 


#MEANS AND STD FOR EACH DIMENSSION

f0=[]
f1=[]
f2=[]
f3=[]
f4=[]
f5=[]
f6=[]
Actions=[]
count=1

a_file = "/home/vamp/robomimicdir/my_scripts/BCO/action_txt_g"
for action in open(a_file):
    A = action.replace("[", "").replace("\n", "").replace("]","").split()
    a = np.array([float(x) for x in A])
    Actions.append(a)

Actions = np.array(Actions)
#min_values = np.min(Actions, axis=0)
#max_values = np.max(Actions, axis=0)
mean_values = np.mean(Actions, axis=0)
std_values = np.std(Actions, axis=0)
print(mean_values)
print(std_values)
