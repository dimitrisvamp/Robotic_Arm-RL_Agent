import os
import h5py
import numpy as np 

hdf5_file = "/home/vamp/robomimicdir/robomimic/datasets/lift/ph/low_dim.hdf5"

f = h5py.File(hdf5_file, "r")
s_file = open("state_txt_all", "w")
ns_file = open("nstate_txt_all", "w")
a_file = open("action_txt_all", "w")
demos = list(f["data"].keys())
for i in range(len(demos)):
    states = f["data/{}/states".format(demos[i])][()]
    actions = np.array(f["data/{}/actions".format(demos[i])][()])
    for j in range(len(states)-1):
        #if(states[j][0] > 1.25):
            #break
        s_file.write("".join(str(states[j]).replace('\n',''))+"\n")
        ns_file.write("".join(str(states[j+1]).replace('\n',''))+"\n")
        a_file.write("".join(str(actions[j]).replace('\n',''))+"\n")

s_file.close()
ns_file.close()
a_file.close()


    

        


        







