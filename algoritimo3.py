#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 14:03:54 2018

@author: pedrocampelo
"""

import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

cot = pd.read_pickle('dataframe_acoes_arrumado')
cot=df

# Reduz dataframe para cotacoes
for column in cot:
    for delete in ['min','max','var','vol']:
        if column.startswith(delete):
            cot.drop([column],axis=1,inplace=True)
            


# Escolha as acoes para colocar no portfolio

#Lista de todas as acoes que foram baixadas            
acoes = ['PETR4.SA', 'ABEV3.SA', 'B3SA3.SA', 'BBAS3.SA', 'BBDC3.SA', 'BBSE3.SA', 
         'VALE3.SA','WEGE3.SA', 'USIM5.SA', 'TAEE11.SA', 'ITUB4.SA', 'KROT3.SA', 
         'FLRY3.SA','EGIE3.SA', 'CVCB3.SA', 'CIEL3.SA', 'BRFS3.SA', 'ITSA4.SA', 
         'LAME4.SA',]

#Sub lista como todas acoes a analisar
acoes1 = ['PETR4','ABEV3','WEGE3','BRFS3', 'ITUB4']

#Se quiser analisar todas basta fazer:
#acoes1=acoes        

ret_ln = pd.DataFrame(columns = acoes1)
for coluna in list(cot):
    for acao in acoes1:
        if acao in coluna:
            ret_ln[acao] = cot[coluna]

ret_ln = np.log(ret_ln) - np.log(ret_ln.shift(1))
ret_ln.dropna(inplace=True)

# Um ano tem 252 dias, queremos fazer pra um mes certo? 252/12
matriz_cov_mes = np.asarray(21*ret_ln.cov())
media_mes = np.asarray(21*ret_ln.mean())

# Otimizacao baseada em numeros aleatorios
n_acoes1 = len(acoes1)
n_portfolios = 100000
ret_port = []
vol_port = []
pesos_port = []
pesos_min = []
vol_min = 100000    #Aqui vc define seu nivel de risco (5%, 10%)


for um_portfolio in range(n_portfolios):
    pesos = np.random.random(n_acoes1)
    pesos /= np.sum(pesos)
    ret_temp = np.dot(pesos, media_mes)
#    ret_port.append(np.dot(pesos, media_mes))
    ret_port.append(ret_temp)
    vol_temp = np.sqrt(np.dot(pesos.T, np.dot(matriz_cov_mes, pesos)))
#    vol_port.append(np.sqrt(np.dot(pesos.T, np.dot(matriz_cov_mes, pesos))))
    vol_port.append(vol_temp)
    pesos_port.append(pesos)
#Separação da volatilidade mínima
    if vol_temp < vol_min:
        vol_min = vol_temp
        pesos_min_vol = pesos
    

portfolios = pd.DataFrame(columns=['retorno', 'volatilidade', 'pesos'])
portfolios['retorno'] = ret_port
portfolios['volatilidade'] = vol_port
portfolios['pesos'] = pesos_port

portfolios.plot.scatter(x='volatilidade', y='retorno', grid=True)
plt.xlabel('Desvio Padrão')
plt.ylabel('Retorno Esperado')
plt.title('Fronteira Eficiente')
plt.show()

#mostra qual a volatilidade minima e os pesos da carteira de minima volatilidade

print (vol_min)
print (pesos_min_vol)
