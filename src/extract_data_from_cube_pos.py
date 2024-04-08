import os
import h5py
import numpy as np 

hdf5_file = "/home/vamp/robomimicdir/robomimic/datasets/lift/ph/low_dim.hdf5"

f = h5py.File(hdf5_file, "r")
Ls_file = open("state_cube_left.txt", "w")
Lns_file = open("nstate_cube_left.txt", "w")
La_file = open("action_cube_left.txt", "w")

Rs_file = open("state_cube_right.txt", "w")
Rns_file = open("nstate_cube_right.txt", "w")
Ra_file = open("action_cube_right.txt", "w")

Cs_file = open("state_cube_center.txt", "w")
Cns_file = open("nstate_cube_center.txt", "w")
Ca_file = open("action_cube_center.txt", "w")
demos = list(f["data"].keys())

for i in range(len(demos)):
    states = f["data/{}/states".format(demos[i])][()]
    actions = np.array(f["data/{}/actions".format(demos[i])][()])
    for j in range(len(states)-1):
        if(states[j][0] > 1.25):
            break
        if(states[0][11] > 0.01):
            Rs_file.write("".join(str(states[j]).replace('\n',''))+"\n")
            Rns_file.write("".join(str(states[j+1]).replace('\n',''))+"\n")
            Ra_file.write("".join(str(actions[j]).replace('\n',''))+"\n")
        if(states[0][11] < -0.01):
            Ls_file.write("".join(str(states[j]).replace('\n',''))+"\n")
            Lns_file.write("".join(str(states[j+1]).replace('\n',''))+"\n")
            La_file.write("".join(str(actions[j]).replace('\n',''))+"\n")
        if(states[0][11]>-0.01 and  states[0][11]<0.01):
            Cs_file.write("".join(str(states[j]).replace('\n',''))+"\n")
            Cns_file.write("".join(str(states[j+1]).replace('\n',''))+"\n")
            Ca_file.write("".join(str(actions[j]).replace('\n',''))+"\n")

Ls_file.close()
Lns_file.close()
La_file.close()

Rs_file.close()
Rns_file.close()
Ra_file.close()

Cs_file.close()
Cns_file.close()
Ca_file.close()



