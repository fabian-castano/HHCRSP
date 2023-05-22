from pulp import *
import numpy as np
import pandas as pd
import seaborn as sns


def nspterminado_calcular_modelo(modelo, version_model, I, NV, Z, B, L, K, Sat, Sun, fds, S, D, P, DM, HMM, HA, TF, DH, MPR1, MPR2, DispM, DispA, DLI ):
    model = modelo(version_model, I, NV, Z, B, L, K, Sat, Sun, fds, S, D, P, DM, HMM, HA, TF, DH, MPR1, MPR2, DispM, DispA, DLI)

    valores = {}
    for v in model.variables():
        if str(v.name) == 'MDH' and version_model == 1: #deolver FO revisar
            valores['MDH'] = v.varValue
        elif str(v.name) == 'PDL' and version_model == 2:
            valores['PDL'] = v.varValue
        elif str(v.name) == 'PTF' and version_model == 2:
            valores['PTF'] = v.varValue

    return valores


def nspterminado(funcion_modelo, I, NV, Z, B, L, K, Sat, Sun, fds, S, D, P, DM, HMM, HA, TF, DH, MPR1, MPR2, DispM, DispA, DLI):
    modelos = {
        'modelo_1': nspterminado_calcular_modelo(funcion_modelo, 1, I, NV, Z, B, L, K, Sat, Sun, fds, S, D, P, DM, HMM, HA, TF, DH, MPR1, MPR2, DispM, DispA, DLI),
        'modelo_2': nspterminado_calcular_modelo(funcion_modelo, 2, I, NV, Z, B, L, K, Sat, Sun, fds, S, D, P, DM, HMM, HA, TF, DH, MPR1, MPR2, DispM, DispA, DLI)
    }
    return modelos


def epsilon_restricciones(funcion_modelo, cant_epsilon, I, NV, Z, B, L, K, Sat, Sun, fds, S, D, P, DM, HMM, HA, TF, DH, MPR1, MPR2, DispM, DispA, DLI):
    variables_modelo_1 = funcion_modelo(1, I, NV, Z, B, L, K, Sat, Sun, fds, S, D, P, DM, HMM, HA, TF, DH, MPR1, MPR2, DispM, DispA, DLI).variablesDict()
    variables_modelo_2 = funcion_modelo(2, I, NV, Z, B, L, K, Sat, Sun, fds, S, D, P, DM, HMM, HA, TF, DH, MPR1, MPR2, DispM, DispA, DLI).variablesDict()
    min_fo1, max_fo2 = variables_modelo_1['fo_1'].varValue, variables_modelo_1['fo_2'].varValue
    max_fo1, min_fo2 = variables_modelo_2['fo_1'].varValue, variables_modelo_2['fo_2'].varValue

    print("=====================================================")
    print('Min FO1: ', np.round(min_fo1,2), 'Max FO1: ', np.round(max_fo1,2))
    print('Min FO2: ', np.round(min_fo2,2), 'Max FO2: ', np.round(max_fo2,2))
    print("=====================================================")
    epsilons = np.linspace(min_fo2, max_fo2, cant_epsilon) #epsilom es un conjunto que son los valores para los que generara el frente 
    frente_pareto = np.zeros([epsilons.size,2]) # frente_pareto es una arreglo de numpy que primero  va a estar lleno de ceros y va a tener tantos épsilon como se les mande y va a tener dos columnas en las que se guardan las funciones objetivos

    #Aqui se empieza a iterar para armar el frente
    variables_ciclo = []
    for i in range(epsilons.size):
        variables_modelo = funcion_modelo(1, I, NV, Z, B, L, K, Sat, Sun, fds, S, D, P, DM, HMM, HA, TF, DH, MPR1, MPR2, DispM, DispA, DLI, epsilons[i]).variablesDict()
        variables_ciclo.append(variables_modelo)
        frente_pareto[i,0], frente_pareto[i,1] = variables_modelo['fo_1'].varValue, variables_modelo['fo_2'].varValue
        print('Iteración: ', i, 'Epsilon: ', np.round(epsilons[i],2), 'FO1: ', np.round(frente_pareto[i,0],2), 'FO2: ', np.round(frente_pareto[i,1],2))
    variables_ciclo = pd.DataFrame(variables_ciclo)

    # return frente_pareto
    results = pd.DataFrame(frente_pareto) #guarda los resultados en una hoja de calculo un dataframe es una hoja de calculo 
    results.columns = ['MDH', 'PDL+PTF']
    display(results)
    grafica=sns.scatterplot(x=results['MDH'], y=results['PDL+PTF']) #genera el grafico de frente de pareto


    resultado_modelos = nspterminado(funcion_modelo, I, NV, Z, B, L, K, Sat, Sun, fds, S, D, P, DM, HMM, HA, TF, DH, MPR1, MPR2, DispM, DispA, DLI)
    MDH_constante = resultado_modelos['modelo_1']['MDH']
    resultado_modelos

    def calcular_suma_PDL_PTF():
        PDL = resultado_modelos['modelo_2']['PDL']
        PTF = resultado_modelos['modelo_2']['PTF']
        suma = PDL + PTF
        return suma

    suma_PDL_PTF_constante = calcular_suma_PDL_PTF()
    suma_PDL_PTF_constante

    # Gap
    GAP = pd.DataFrame()
    GAP['MDH%'] = results['MDH'].apply(lambda mdh: abs(mdh - MDH_constante) / MDH_constante)
    GAP['PDL%'] = results['PDL+PTF'].apply(lambda pdl: abs(pdl - suma_PDL_PTF_constante) / suma_PDL_PTF_constante)

    MDH_menor = GAP['MDH%'] == GAP['MDH%'].min()
    GAP_MDH_menor = GAP[MDH_menor]
    PDL_menor = GAP_MDH_menor['PDL%'] == GAP_MDH_menor['PDL%'].min()

    porcentaje_menor = GAP_MDH_menor[PDL_menor]
    punto_menor = results[MDH_menor][PDL_menor]
    variables_ciclo = variables_ciclo[MDH_menor][PDL_menor]
    variables_ciclo = variables_ciclo.iloc[0]

    display(GAP)
    display(results)
    print('Variables Camila')
    display(porcentaje_menor) # variables de Camila
    display(punto_menor) # variables de Camila
    display(variables_ciclo['MDH'].varValue, variables_ciclo['PDL'].varValue, variables_ciclo['PTF'].varValue)
    return variables_ciclo


# resultado = nspterminado()