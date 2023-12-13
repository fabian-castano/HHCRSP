# %%
# import classes from folder app/classes.py

import json
import sys
import os

path_to_highs='/Users/fabiancastano/Documents/HiGHS/build/bin/highs'


DM = 8
# Esto hay que parametrizarlo con base en el mes
HMM = 216

def consolidate(input_path:str,instance_number:int=0):
    with open(input_path+f'shifts_{instance_number}.json', 'r') as f:
        shifts = json.load(f)

    with open(input_path+f'nurses_{instance_number}.json', 'r') as f:
        nurses = json.load(f)

    instance={}
    instance['nurses']=nurses
    instance['shifts']=shifts
    output_file=input_path+'instance_'+str(instance_number)+'.json'
    with open(output_file, 'w') as f:
        json.dump(instance, f)

if __name__=="__main__":
    current_path=os.getcwd()
    
    for i in range(2,172):
        try:
            consolidate(current_path+'/instances/',i)
        except:
            print(f"Error in instance {i}")
    