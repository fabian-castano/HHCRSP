from datetime import datetime, timedelta
from math import floor
from typing import Dict

from app.libs.classes import Shift, Nurse
from app.libs.constants import HMM, DM
import pulp as plp
from itertools import product


class MultiObjectiveSolver:
    def __int__(self, nurses: Dict[int, Nurse], shifts: Dict[str, Shift]):
        self.nurses = {}
        self.shifts = {}
        self.model = plp.LpProblem("Nurse Scheduling", plp.LpMinimize)

        # set of nurses
        self.I = set(nurses.keys())
        # set of shifts
        self.J = set(shifts.keys())
        # set of days

        self.valid_keys = self._get_valid_keys()

        self.X = None  # the nurse i ∈I is assigned at a morning shift z ∈ Z on a day k ∈ K
        self.V = None

        # auxiliary variables

        self.Y = None  # representing the number of shifts assigned during the month to nurse 𝑖𝜖𝐼
        self.MDH = None  # that estimates the difference between maximum shifts that should be assigned to a caregiver (considering overtime hours to be balanced) and real number of shifts assigned
        self.DOff = None  # denoting the number of weekend days off in the month of the nurse 𝑖𝜖𝐼.

        self._create_decision_variables()

    def _create_decision_variables(self):
        self.X = plp.LpVariable.dicts("X", self.valid_keys, cat=plp.LpBinary)
        self.W = plp.LpVariable.dicts("W", [(i, self.shifts[j].shift_date) for i in self.I for j in self.J if (i, j) in self.valid_keys], cat=plp.LpBinary)

        self.V = plp.LpVariable.dicts("V", [i for i in self.I], cat=plp.LpContinuous) # overtime

        self.Y = plp.LpVariable.dicts("Y", [i for i in self.I], cat=plp.LpContinuous)
        self.MDH = plp.LpVariable("MDH", cat=plp.LpContinuous)
        self.DOff = plp.LpVariable.dicts("DOff", [i for i in self.I], cat=plp.LpContinuous)

    def _get_valid_keys(self):
        valid_keys = [(i, j) for i in self.I for j in self.J]
        for i, j in valid_keys:
            if self.shifts[j].shift_date not in self.nurses[i].unavailability_days:
                valid_keys.remove((i, j))

        return valid_keys

    def _create_objective_minimize_penalty(self):
        expr_PDL = plp.lpSum([(self.X[i, j] for i in self.I for j in self.J if
                               self.shifts[j].shift_date in self.nurses[i].days_off_requested)])
        # accounts for the number of weekends a person is assigned to a shif not in his/her preference
        expr_PWE = plp.lpSum([(self.X[i, j]) for i in self.I for j in self.J if
                              datetime.strptime(self.shifts[j].shift_date, '%Y-%m-%d').weekday() in [5, 6] and
                              self.nurses[i].weekend_preference != self.shifts[j].type_of_shift])

        return expr_PDL + expr_PWE

    def _create_objective_minimize_overtime(self):
        return self.MDH+plp.lpSum([self.V[i] for i in self.I])*(1/DM)

    def _create_constraints(self):
        for nurse in self.I:
            self.model += self.Y[nurse] == plp.lpSum([self.X[i, j] for (i, j) in self.valid_keys if
                                                  nurse == i]), f"count_shifts_{self.nurses[nurse].nurse_id}"

        for nurse in self.I:
            self.model += self.Y[nurse] - (HMM - self.nurses[nurse].accumulated_hours) * (1 / DM) <= self.MDH, f"balance_hours_1_{self.nurses[nurse].nurse_id}"

            self.model += (HMM - self.nurses[nurse].accumulated_hours) * (1 / DM)- self.Y[nurse]  <= self.MDH, f"balance_hours_2_{self.nurses[nurse].nurse_id}"

        # for each nurse and each day, the number of shifts assigned to the nurse is less than or equal to 1
        for i in self.I:
            for j1 in self.J:
                for j2 in self.J:
                    if j1 != j2 and self.shifts[j1].shift_date == self.shifts[j2].shift_date:
                        self.model += self.X[i, j1] + self.X[i, j2] <= 1, f"one_shift_per_day_{self.nurses[i].nurse_id}_{self.shifts[j1].shift_date}"

        # accounts for overtime V hours required
        for i in self.I:
            self.model += self.Y[i]*DM<=(HMM - self.nurses[i].accumulated_hours) + self.V[i], f"overtime_{self.nurses[i].nurse_id}"

        # demand is satisfied
        for j in self.J:
            self.model += plp.lpSum([self.X[i, j] for i in self.I if (i, j) in self.valid_keys
                                     ]) >= self.shifts[j].demand, f"demand_{self.shifts[j].shift_date}"

        # for each weekend, a nurse works at most one day
        for i in self.I:
            for j1 in self.J:
                if (i,j1) in self.valid_keys and datetime.strptime(self.shifts[j1].shift_date, '%Y-%m-%d').weekday()==5:
                    self.model += self.X[i, j1] + plp.lpSum([self.X[i, j2] for j2 in self.J if (i,j2) in self.valid_keys and datetime.strptime(self.shifts[j2].shift_date, '%Y-%m-%d').weekday()==6
                                                             and datetime.strptime(self.shifts[j2].shift_date, '%Y-%m-%d')==datetime.strptime(self.shifts[j1].shift_date, '%Y-%m-%d')+timedelta(days=1)
                                                             ]) <= 1, f"weekend_{self.nurses[i].nurse_id}_{self.shifts[j1].shift_date}"

        for (i, j) in self.valid_keys:
            self.model += (1-self.X[i, j] )<= self.W[i, self.shifts[j].shift_date], f"weekend_off_{self.nurses[i].nurse_id}_{self.shifts[j].shift_date}"

        days=list([datetime.strptime(self.shifts[j].shift_date , '%Y-%m-%d').weekday() for j in self.J if datetime.strptime(self.shifts[j].shift_date , '%Y-%m-%d').weekday() in [5,6]])
        total_weekend_days=len(days)
        for i in self.I:
            self.model += self.DOff[i] == plp.lpSum([self.W[i, self.shifts[j].shift_date] for j in self.J if (i, j) in self.valid_keys]), f"weekend_off_{self.nurses[i].nurse_id}"
            self.model += self.DOff[i] >= floor(total_weekend_days/2)+1, f"weekend_off_{self.nurses[i].nurse_id}"

        # An afternoon shift cannot be followed by a morning shift in the next day, this regulation is ensured by constraint
        for i in self.I:
            for j1 in self.J:
                if (i, j1) in self.valid_keys and self.shifts[j1].type_of_shift == 'A':
                    self.model += plp.lpSum([self.X[i, j2] for j2 in self.J if (i, j2) in self.valid_keys and
                                             datetime.strptime(self.shifts[j2].shift_date, '%Y-%m-%d') ==
                                             datetime.strptime(self.shifts[j1].shift_date, '%Y-%m-%d') +
                                             timedelta(days=1) and self.shifts[j2].type_of_shift == 'M']) <= 1, f"morning_afternoon_{self.nurses[i].nurse_id}_{self.shifts[j1].shift_date}"
