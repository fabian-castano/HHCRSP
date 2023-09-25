# %%
# import classes from folder app/classes.py
from datetime import datetime, timedelta
from math import floor

from app.libs.classes import Shift, Nurse
from app.libs.constants import HMM, DM
import pulp as plp
from itertools import product
from app.libs.classes import Shift, Nurse
import json
import pandas as pd
import numpy as np
import sys


DM = 8
# Esto hay que parametrizarlo con base en el mes
HMM = 216

def check_feasibility(input_path:str):
    with open(input_path+'shifts.json', 'r') as f:
        shifts = json.load(f)

    with open(input_path+'nurses.json', 'r') as f:
        nurses = json.load(f)

    shifts = {shift['shift_id']: Shift(**shift) for shift in shifts}
    nurses = {nurse['nurse_id']: Nurse(**nurse) for nurse in nurses}
    
    # Additional parameters 

    model = plp.LpProblem("Nurse Scheduling", plp.LpMinimize)
    I = set(nurses.keys())
    J = set(shifts.keys())
    K = set([datetime.strptime(shifts[j].shift_date, '%Y-%m-%d').day for j in J])
    week=set([datetime.strptime(shifts[j].shift_date, '%Y-%m-%d').isocalendar()[1] for j in J])

    valid_keys = [(i, j) for (i,j) in product(I, J) if datetime.strptime(shifts[j].shift_date, '%Y-%m-%d').day not in nurses[i].vacations]


    X = plp.LpVariable.dicts("X", valid_keys, cat=plp.LpBinary)
    W = plp.LpVariable.dicts("W", [(i, k) for (i,k) in product(I, K)], cat=plp.LpBinary)
    Zmorning=plp.LpVariable.dicts("Zmorning", [(i, w) for (i,w) in product(I, week)], cat=plp.LpBinary)
    Zafternoon=plp.LpVariable.dicts("Zafternoon", [(i, w) for (i,w) in product(I, week)], cat=plp.LpBinary)

    V = plp.LpVariable.dicts("V", I, cat=plp.LpContinuous,lowBound=0)  # overtime
    Y = plp.LpVariable.dicts("Y", I, cat=plp.LpContinuous,lowBound=0) # TOTAL NUMBER OF SHIFTS ASSIGNED TO NURSE i
    MDH = plp.LpVariable("MDH", cat=plp.LpContinuous,lowBound=0) # MAXIMUM DIFFERENCE BETWEEN MAXIMUM SHIFTS AND REAL NUMBER OF SHIFTS ASSIGNED
    DOff = plp.LpVariable.dicts("DOff", I, cat=plp.LpContinuous,lowBound=0) 



    expr_PDL = plp.lpSum([(X[i, j] for (i,j) in valid_keys if datetime.strptime(shifts[j].shift_date, '%Y-%m-%d').day in nurses[i].dates_off)])
    # accounts for the number of weekends a person is assigned to a shif not in his/her preference
    expr_PWE = plp.lpSum([(X[i, j]) for (i,j) in valid_keys if
                            datetime.strptime(shifts[j].shift_date, '%Y-%m-%d').weekday() in [5, 6] and
                            (nurses[i].shift_preference != shifts[j].shift_type)])

    Obj_2 = MDH + plp.lpSum([V[i] for i in I]) * (1 / DM)


    # accounts the number of shifts assigned to a nurse
    for nurse in I:
        model+=Y[nurse] == plp.lpSum([X[nurse, shift] for shift in J if (nurse, shift) in valid_keys]), f"assigned_shifts_{nurse}"


    # accounts for the difference between shifts that should be assigned to a caregiver 
                # (considering overtime hours to be balanced) and real number of shifts assigned

    for nurse in I:
        expr = Y[nurse] - (HMM - nurses[nurse].accumulated_hours) * (1 / DM) <= MDH
        model += expr, f"balance_hours_1_{nurses[nurse].nurse_id}"

        expr = (HMM - nurses[nurse].accumulated_hours) * (1 / DM) - Y[nurse]<= MDH
        model += expr,  f"balance_hours_2_{nurses[nurse].nurse_id}"
        


    # dictionary enumerating the shifts happening in each day
    shifts_per_day = {}
    for shift in J:
        if shifts[shift].shift_date not in shifts_per_day:
            shifts_per_day[shifts[shift].shift_date] = []
        shifts_per_day[shifts[shift].shift_date].append(shift)



    for i in I:
        for spd in shifts_per_day.keys():
            model += plp.lpSum([X[i, j] for j in shifts_per_day[spd] if (i, j) in valid_keys]) <= 1, f"one_shift_per_day_{nurses[i].nurse_id}_{spd}"



    # accounts for overtime V hours required
    for i in I:
        model += Y[i] * DM <= (HMM - nurses[i].accumulated_hours) + V[i], f"overtime_{nurses[i].nurse_id}"


    # demand is satisfied
    for j in J:
        model += plp.lpSum([X[(i, j)] for i in I if (i, j) in valid_keys]) >= shifts[j].demand, f"demand_{j}"
            


    # for each weekend, a nurse works at most one day
    for i in I:
        for j1 in J:
            if (i, j1) in valid_keys and datetime.strptime(shifts[j1].shift_date,'%Y-%m-%d').weekday() == 5:
                model += X[i, j1] + plp.lpSum([X[i, j2] for j2 in J if
                                                            (i, j2) in valid_keys and datetime.strptime(shifts[j2].shift_date, '%Y-%m-%d').weekday() == 6
                                                            and datetime.strptime(shifts[j2].shift_date,'%Y-%m-%d') == datetime.strptime(shifts[j1].shift_date, '%Y-%m-%d') + timedelta(days=1)]) <= 1, f"weekend_{nurses[i].nurse_id}_{j1}"



    for i,k in product(I, K):
        model += (1 - plp.lpSum([X[i, j] for j in J if (i, j) in valid_keys and datetime.strptime(shifts[j].shift_date, '%Y-%m-%d').day == k])) ==W[i, k], f"day_is_off_{nurses[i].nurse_id}_{k}"


    days = list([datetime.strptime(shifts[j].shift_date, '%Y-%m-%d').day for j in J if datetime.strptime(shifts[j].shift_date, '%Y-%m-%d').weekday() in [5, 6]])

    total_weekend_days = len(days)

    for i in I:
        model += DOff[i] == plp.lpSum([W[i, k] for k in days]), f"weekend_off_1_{nurses[i].nurse_id}"
        model += DOff[i] >= floor(total_weekend_days / 2) + 1, f"weekend_off_2_{nurses[i].nurse_id}"


    for i,w in product(I, week):
        model += Zmorning[i, w] + Zafternoon[i, w] <= 1, f"shifts_per_week_{nurses[i].nurse_id}_{w}"


    # if a nurse works in a morning shift the varriable Z_morning is equal to 1
    for i in I:
        for j in J:
            if (i, j) in valid_keys and shifts[j].shift_type == 'morning':
                model +=  X[i, j]<=Zmorning[i, datetime.strptime(shifts[j].shift_date, '%Y-%m-%d').isocalendar()[1]], f"morning_shift_{nurses[i].nurse_id}_{j}"
            elif (i, j) in valid_keys and shifts[j].shift_type == 'afternoon':
                model +=  X[i, j]<=Zafternoon[i, datetime.strptime(shifts[j].shift_date, '%Y-%m-%d').isocalendar()[1]], f"afternoon_shift_{nurses[i].nurse_id}_{j}"


    #an afternoon shift cannot be followed by a morning shift
    for i in I:
        for j in J:
            if (i, j) in valid_keys and shifts[j].shift_type == 'morning':
                model +=  X[i, j] + plp.lpSum([X[i, j2] for j2 in J if
                                                            (i, j2) in valid_keys and datetime.strptime(shifts[j2].shift_date, '%Y-%m-%d') == datetime.strptime(shifts[j].shift_date, '%Y-%m-%d') - timedelta(days=1)
                                                            and shifts[j2].shift_type == 'afternoon']) <= 1, f"morning_afternoon_{nurses[i].nurse_id}_{j}"


    
    model+= expr_PDL+expr_PWE<=np.inf, "value_obj_1"
    model+= Obj_2<=np.inf , "value_obj_2"
    #
    model.setObjective(expr_PDL+expr_PWE)
    # solve the problem using HIGHS
    solver = plp.GUROBI_CMD(msg=0)
    model.solve(solver)

    if plp.LpStatus[model.status] != "Optimal":
        return False

    best_1=expr_PDL.value()+expr_PWE.value()
    ctr = model.constraints["value_obj_1"]
    ctr.changeRHS(best_1 )
    model.setObjective(Obj_2)
    model.solve(solver)
    worst_2=Obj_2.value()
    ctr = model.constraints["value_obj_1"]
    ctr.changeRHS(np.inf)
    ctr = model.constraints["value_obj_2"]
    ctr.changeRHS(np.inf) 
    model.setObjective(Obj_2)
    model.solve(solver)
    best_2=Obj_2.value()
    ctr = model.constraints["value_obj_2"]
    ctr.changeRHS(best_2 )

    model.setObjective(expr_PDL+expr_PWE)
    model.solve(solver)
    worst_1=expr_PDL.value()+expr_PWE.value()
    print(best_1,best_2,worst_1,worst_2)

    ctr = model.constraints["value_obj_1"]
    ctr.changeRHS(best_1)
    ctr = model.constraints["value_obj_2"]
    ctr.changeRHS(np.inf)
    model.setObjective(Obj_2)
    return best_1!=worst_1 and best_2!=worst_2
    




# %%


def construct_model(input_path:str=None,instance=None):
    with open(input_path+'shifts_'+str(instance)+'.json', 'r') as f:
        shifts = json.load(f)

    with open(input_path+'nurses_'+str(instance)+'.json', 'r') as f:
        nurses = json.load(f)

    shifts = {shift['shift_id']: Shift(**shift) for shift in shifts}
    nurses = {nurse['nurse_id']: Nurse(**nurse) for nurse in nurses}
    
    print(f" The instance {instance} has {len(nurses)} nurses and {len(shifts)} shifts")

    # Additional parameters 
    """
    model = plp.LpProblem("Nurse_Scheduling", plp.LpMinimize)
    I = set(nurses.keys())
    J = set(shifts.keys())
    K = set([datetime.strptime(shifts[j].shift_date, '%Y-%m-%d').day for j in J])
    week=set([datetime.strptime(shifts[j].shift_date, '%Y-%m-%d').isocalendar()[1] for j in J])

    valid_keys = [(i, j) for (i,j) in product(I, J) if datetime.strptime(shifts[j].shift_date, '%Y-%m-%d').day not in nurses[i].vacations]


    X = plp.LpVariable.dicts("X", valid_keys, cat=plp.LpBinary)
    W = plp.LpVariable.dicts("W", [(i, k) for (i,k) in product(I, K)], cat=plp.LpBinary)
    Zmorning=plp.LpVariable.dicts("Zmorning", [(i, w) for (i,w) in product(I, week)], cat=plp.LpBinary)
    Zafternoon=plp.LpVariable.dicts("Zafternoon", [(i, w) for (i,w) in product(I, week)], cat=plp.LpBinary)

    V = plp.LpVariable.dicts("V", I, cat=plp.LpContinuous,lowBound=0)  # overtime
    Y = plp.LpVariable.dicts("Y", I, cat=plp.LpContinuous,lowBound=0) # TOTAL NUMBER OF SHIFTS ASSIGNED TO NURSE i
    MDH = plp.LpVariable("MDH", cat=plp.LpContinuous,lowBound=0) # MAXIMUM DIFFERENCE BETWEEN MAXIMUM SHIFTS AND REAL NUMBER OF SHIFTS ASSIGNED
    DOff = plp.LpVariable.dicts("DOff", I, cat=plp.LpContinuous,lowBound=0) 



    expr_PDL = plp.lpSum([(X[i, j] for (i,j) in valid_keys if datetime.strptime(shifts[j].shift_date, '%Y-%m-%d').day in nurses[i].dates_off)])
    # accounts for the number of weekends a person is assigned to a shif not in his/her preference
    expr_PWE = plp.lpSum([(X[i, j]) for (i,j) in valid_keys if
                            datetime.strptime(shifts[j].shift_date, '%Y-%m-%d').weekday() in [5, 6] and
                            (nurses[i].shift_preference != shifts[j].shift_type)])

    Obj_2 = MDH + plp.lpSum([V[i] for i in I]) * (1 / DM)


    # accounts the number of shifts assigned to a nurse
    for nurse in I:
        model+=Y[nurse] == plp.lpSum([X[nurse, shift] for shift in J if (nurse, shift) in valid_keys]), f"assigned_shifts_{nurse}"


    # accounts for the difference between shifts that should be assigned to a caregiver 
                # (considering overtime hours to be balanced) and real number of shifts assigned

    for nurse in I:
        expr = Y[nurse] - (HMM - nurses[nurse].accumulated_hours) * (1 / DM) <= MDH
        model += expr, f"balance_hours_1_{nurses[nurse].nurse_id}"

        expr = (HMM - nurses[nurse].accumulated_hours) * (1 / DM) - Y[nurse]<= MDH
        model += expr,  f"balance_hours_2_{nurses[nurse].nurse_id}"
        


    # dictionary enumerating the shifts happening in each day
    shifts_per_day = {}
    for shift in J:
        if shifts[shift].shift_date not in shifts_per_day:
            shifts_per_day[shifts[shift].shift_date] = []
        shifts_per_day[shifts[shift].shift_date].append(shift)



    for i in I:
        for spd in shifts_per_day.keys():
            model += plp.lpSum([X[i, j] for j in shifts_per_day[spd] if (i, j) in valid_keys]) <= 1, f"one_shift_per_day_{nurses[i].nurse_id}_{spd}"



    # accounts for overtime V hours required
    for i in I:
        model += Y[i] * DM <= (HMM - nurses[i].accumulated_hours) + V[i], f"overtime_{nurses[i].nurse_id}"


    # demand is satisfied
    for j in J:
        model += plp.lpSum([X[(i, j)] for i in I if (i, j) in valid_keys]) >= shifts[j].demand, f"demand_{j}"
            


    # for each weekend, a nurse works at most one day
    for i in I:
        for j1 in J:
            if (i, j1) in valid_keys and datetime.strptime(shifts[j1].shift_date,'%Y-%m-%d').weekday() == 5:
                model += X[i, j1] + plp.lpSum([X[i, j2] for j2 in J if
                                                            (i, j2) in valid_keys and datetime.strptime(shifts[j2].shift_date, '%Y-%m-%d').weekday() == 6
                                                            and datetime.strptime(shifts[j2].shift_date,'%Y-%m-%d') == datetime.strptime(shifts[j1].shift_date, '%Y-%m-%d') + timedelta(days=1)]) <= 1, f"weekend_{nurses[i].nurse_id}_{j1}"



    for i,k in product(I, K):
        model += (1 - plp.lpSum([X[i, j] for j in J if (i, j) in valid_keys and datetime.strptime(shifts[j].shift_date, '%Y-%m-%d').day == k])) ==W[i, k], f"day_is_off_{nurses[i].nurse_id}_{k}"


    days = list([datetime.strptime(shifts[j].shift_date, '%Y-%m-%d').day for j in J if datetime.strptime(shifts[j].shift_date, '%Y-%m-%d').weekday() in [5, 6]])

    total_weekend_days = len(days)

    for i in I:
        model += DOff[i] == plp.lpSum([W[i, k] for k in days]), f"weekend_off_1_{nurses[i].nurse_id}"
        model += DOff[i] >= floor(total_weekend_days / 2) + 1, f"weekend_off_2_{nurses[i].nurse_id}"


    for i,w in product(I, week):
        model += Zmorning[i, w] + Zafternoon[i, w] <= 1, f"shifts_per_week_{nurses[i].nurse_id}_{w}"


    # if a nurse works in a morning shift the varriable Z_morning is equal to 1
    for i in I:
        for j in J:
            if (i, j) in valid_keys and shifts[j].shift_type == 'morning':
                model +=  X[i, j]<=Zmorning[i, datetime.strptime(shifts[j].shift_date, '%Y-%m-%d').isocalendar()[1]], f"morning_shift_{nurses[i].nurse_id}_{j}"
            elif (i, j) in valid_keys and shifts[j].shift_type == 'afternoon':
                model +=  X[i, j]<=Zafternoon[i, datetime.strptime(shifts[j].shift_date, '%Y-%m-%d').isocalendar()[1]], f"afternoon_shift_{nurses[i].nurse_id}_{j}"


    #an afternoon shift cannot be followed by a morning shift
    for i in I:
        for j in J:
            if (i, j) in valid_keys and shifts[j].shift_type == 'morning':
                model +=  X[i, j] + plp.lpSum([X[i, j2] for j2 in J if
                                                            (i, j2) in valid_keys and datetime.strptime(shifts[j2].shift_date, '%Y-%m-%d') == datetime.strptime(shifts[j].shift_date, '%Y-%m-%d') - timedelta(days=1)
                                                            and shifts[j2].shift_type == 'afternoon']) <= 1, f"morning_afternoon_{nurses[i].nurse_id}_{j}"





    #
    model.setObjective(expr_PDL+expr_PWE)



    # solve the problem using HIGHS
    solver = plp.GUROBI_CMD(msg=1)
    model.solve(solver)
    model+= expr_PDL+expr_PWE<=np.inf, "value_obj_1"
    model+= Obj_2<=np.inf , "value_obj_2"
    #
    model.setObjective(expr_PDL+expr_PWE)
    # solve the problem using HIGHS
    solver = plp.GUROBI_CMD(msg=0)
    model.solve(solver)
    best_1=expr_PDL.value()+expr_PWE.value()
    ctr = model.constraints["value_obj_1"]
    ctr.changeRHS(best_1 )
    model.setObjective(Obj_2)
    model.solve(solver)
    worst_2=Obj_2.value()
    ctr = model.constraints["value_obj_1"]
    ctr.changeRHS(np.inf)
    ctr = model.constraints["value_obj_2"]
    ctr.changeRHS(np.inf) 
    model.setObjective(Obj_2)
    model.solve(solver)
    best_2=Obj_2.value()
    ctr = model.constraints["value_obj_2"]
    ctr.changeRHS(best_2 )

    model.setObjective(expr_PDL+expr_PWE)
    model.solve(solver)
    worst_1=expr_PDL.value()+expr_PWE.value()

    

    print(best_1,best_2,worst_1,worst_2)

    ctr = model.constraints["value_obj_1"]
    ctr.changeRHS(best_1)
    ctr = model.constraints["value_obj_2"]
    ctr.changeRHS(np.inf)
    model.setObjective(Obj_2)

    epsilon = 1
    model.solve()
    lines=[]
    while model.status == 1:
        model.solve()
        print(expr_PDL.value()+expr_PWE.value(),Obj_2.value())
        lines.append([expr_PDL.value()+expr_PWE.value(),Obj_2.value()])
        epsilon += 1
        ctr = model.constraints["value_obj_1"]
        ctr.changeRHS(best_1 + epsilon)


        if best_1 + epsilon > worst_1:
            break
        
    model.setObjective(Obj_2)



    #Get gurobi model status code
    #set gurobi verbose to 1
    solver = plp.GUROBI_CMD(msg=1)
    model.solve(solver)


    date_range=np.sort(list(set([shifts[j].shift_date for j in range(len(shifts))])))
    nurses_shifts = {nurses[i].nurse_name: {shifts[j].shift_date:shifts[j].shift+" - "+shifts[j].shift_type for j in J if (i,j) in valid_keys and X[i, j].varValue > 0.5} for i in I}          
    nurses_morning_shifts={nurses[i].nurse_name: {shifts[j].shift_date:shifts[j].shift for j in J if (i,j) in valid_keys and X[i, j].varValue > 0.5 and shifts[j].shift_type=="morning"} for i in I}
    nurses_afternoo_shifts={nurses[i].nurse_name: {shifts[j].shift_date:shifts[j].shift for j in J if (i,j) in valid_keys and X[i, j].varValue > 0.5 and shifts[j].shift_type=="afternoon"} for i in I}

    tabulated_shifts=[]
    for nurse in nurses_shifts:
        line=[nurse]
        for date in date_range:
            if date not in nurses_shifts[nurse]:
                line.append("-")
                line.append("-")
            elif date in nurses_morning_shifts[nurse]:
                line.append(nurses_morning_shifts[nurse][date])
                line.append("-")
            else:
                line.append("-")
                line.append(nurses_afternoo_shifts[nurse][date])
        tabulated_shifts.append(line)
    tabulated_shifts=pd.DataFrame(tabulated_shifts, columns=["Nurse"]+[str(date)+str(jornada) for date,jornada in product(date_range,["-mañana","-tarde"])])
    tabulated_shifts.to_excel("tabulated_shifts.xlsx")

    df=pd.DataFrame(lines,columns=["PDL","MDH"])
    return df"""






# %%
#filepath is the same as the curren directory
#construct_model("app/libs/")