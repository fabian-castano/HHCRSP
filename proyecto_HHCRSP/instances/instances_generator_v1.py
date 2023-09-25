import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import holidays as pyholidays
import math
import os
import sys

from app.libs.HHCRS_multiobjective_model import check_feasibility,construct_model
from app.libs.constants import shift_type,HMM,DM

# demands dictionary has a first key representing the shift type, a second key representing the day of the week, and a third  separating
# holidays and not holidays using 1 to indicate holiday and 0 to indicate not holiday

def random_demands_generator(shift_type
                             ,output_path
                             ,year:int=2022,month:int=5,max_demand:int=5):

    demands={}
    for shift in shift_type.keys():
        morning_max_demand=max(np.random.randint(max_demand-1,max_demand+1),1)
        afternoon_max_demand=np.random.randint(1,morning_max_demand)
        for st in shift_type[shift]:
            for day in range(7):
                for holiday in [0,1]:
                    if shift=="morning":
                        if st=="M":
                            if day <5 and holiday==0:
                                demands[(st,day,holiday)]=morning_max_demand
                            else:
                                demands[(st,day,holiday)]=morning_max_demand-1
                        elif st=="COM":
                            if day <5 and holiday==0:
                                demands[(st,day,holiday)]=morning_max_demand-1
                            else:
                                demands[(st,day,holiday)]=morning_max_demand-2
                        else:
                            if day <5 and holiday==0:
                                demands[(st,day,holiday)]=0
                            else:
                                demands[(st,day,holiday)]=max(morning_max_demand-2,1)
                    if shift=="afternoon":
                        if st=="T1":
                            if day<5 and holiday==0:
                                demands[(st,day,holiday)]=afternoon_max_demand
                            else:
                                demands[(st,day,holiday)]=afternoon_max_demand
                        elif st=="T2":
                            if day<5 and holiday==0:
                                demands[(st,day,holiday)]=max(afternoon_max_demand-1,1)
                            else:
                                demands[(st,day,holiday)]=max(afternoon_max_demand-1,0)
                        else:
                            if day<5 and holiday==0:
                                demands[(st,day,holiday)]=max(morning_max_demand-1,1)  
                            else:
                                demands[(st,day,holiday)]=0                  
    list_shifts=[]
    # list of dates of january 2023
    
    days_in_month = None
    if month == 2:
        days_in_month = 28
    elif month in [4, 6, 9, 11]:
        days_in_month = 30
    else:
        days_in_month = 31
        

    dates = pd.date_range(start=f'{month}/1/{year}', end=f'{month}/{days_in_month}/{year}')
    shift_id=0
    for fecha in dates:
        for shift in shift_type:
            for jornada in shift_type[shift]:
                list_shifts.append([shift_id, 
                                    fecha.strftime("%Y-%m-%d"), 
                                    fecha.week,
                                    jornada, shift,fecha.weekday(),
                                    fecha.strftime("%Y-%m-%d") in pyholidays.CO(years=year),
                                    demands[(jornada,fecha.weekday(),int( fecha.strftime("%Y-%m-%d") in pyholidays.CO(years=2023)))]
                                    ])
                shift_id+=1
    df_shifts=pd.DataFrame(list_shifts,columns=["shift_id","shift_date","week_of_year","shift","shift_type","weekday","holiday","demand"])
    df_shifts.index=df_shifts["shift_id"]
    
    df_shifts.to_json(output_path+'shifts.json',orient='records')

def random_nurses_generator(input_file
                            ,output_path
                            ,n_nurses:int=8
                            ,year:int=2022):

       df_instance=pd.read_excel(input_file, sheet_name='8')

       # convert string to list
       df_instance["DLI"]=df_instance["DLI"].apply(lambda x: x.split(","))
       df_instance["DLI"]=df_instance["DLI"].apply(lambda x: [int(i) for i in x])
       # find position of 1's in DLI
       df_instance["DLI"]=df_instance["DLI"].apply(lambda x: np.array([i+1 for i, e in enumerate(x) if e == 1]))

       #df_instance["vacations"].apply(lambda x: x.split(",") if type(x) == str else [])
       df_instance["vacations"]=df_instance["vacations"].apply(lambda x: np.array([int(i) for i in x.split(",")]) if type(x) == str else [])#


       # select a random subset of rows
       df_instance=df_instance.sample(n_nurses)
       df_instance['name']=["person_"+str(i) for i in range(1,df_instance.shape[0]+1)]

       df_instance.rename(columns={"ID":"nurse_id",
                     "name":"nurse_name",
                     "vacations":"vacations",
                     "DLI":"dates_off",
                     "TF":"shift_preference",
                     "HA":"accumulated_hours",
                     "vacations":"vacations"
                     },inplace=True)

       #df_instance["dates_off"]=df_instance["dates_off"].apply(lambda x: x.split(",") if type(x) == str else [])
       #df_instance["dates_off"]=df_instance["dates_off"].apply(lambda x: [int(i) for i in x] if type(x) == list else [])


       dict_preference={0:"morning",1:"afternoon"}
       df_instance["shift_preference"]=df_instance["shift_preference"].apply(lambda x: dict_preference[x])
       df_instance.reset_index(inplace=True,drop=True)
       df_instance[['nurse_id', 'nurse_name',  'shift_preference',
              'morning_availability_labor_day', 'morning_availability_weekend',
              'afternoon_availability_labor_day', 'afternoon_availability_weekend',
              'dates_off', 'accumulated_hours','vacations']].to_json(output_path+'nurses.json',orient='records')
       

if __name__=="__main__":

    tot_nurses=int(sys.argv[1])
    month=int(sys.argv[2])
    year=int(sys.argv[3])
    input_file_path='/Users/user/Documents/HHCRSP/proyecto_HHCRSP/data/instances_for_resampling.xlsx'       
    random_nurses_generator(input_file_path
                            ,'/Users/user/Documents/HHCRSP/proyecto_HHCRSP/tmp/'
                            ,tot_nurses
                            ,year)
    random_demands_generator(shift_type
                             ,'/Users/user/Documents/HHCRSP/proyecto_HHCRSP/tmp/'
                             ,year
                             ,month
                             ,math.floor(tot_nurses/3))
    
    if construct_model('/Users/user/Documents/HHCRSP/proyecto_HHCRSP/tmp/'):
        #check_feasibility('/Users/user/Documents/HHCRSP/proyecto_HHCRSP/tmp/'):
        print("feasible")
        # check how many files are in the folder
        instances_total=len(os.listdir('/Users/user/Documents/HHCRSP/proyecto_HHCRSP/instance_generators/'))
        os.system(f'mv /Users/user/Documents/HHCRSP/proyecto_HHCRSP/tmp/shifts.json /Users/user/Documents/HHCRSP/proyecto_HHCRSP/instance_generators/shifts_{int(instances_total/2)}.json')
        os.system(f'mv /Users/user/Documents/HHCRSP/proyecto_HHCRSP/tmp/nurses.json /Users/user/Documents/HHCRSP/proyecto_HHCRSP/instance_generators/nurses_{int(instances_total/2)}.json')