#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 15:15:48 2018

@author: pedrocampelo
"""

if __name__=='__main__':


    import matplotlib.pyplot as plt
    from matplotlib import pyplot
        
    import numpy as np
    import pandas as pd
    from datetime import datetime

        
    #df = pd.read_pickle('dataframe_acoes_arrumado')

  
    #Transformar o tempo em contínuo
    
    #Escolher período de analise
    #Dica: escolha uma dia inicial que não seja FDS/feriado
    diainicial = '02'
    mesinicial = '01'
    anoinicial = '2017'
    diafinal = '30'
    mesfinal = '11'
    anofinal = '2018'


    datainicial_aux= diainicial+'/'+mesinicial+'/'+anoinicial
    datainicial = datetime.strptime(datainicial_aux, '%d/%m/%Y')
    datafinal_aux=diafinal+'/'+mesfinal+'/'+anofinal
    datafinal = datetime.strptime(datafinal_aux, '%d/%m/%Y')
    delta =  (datafinal - datainicial).days
    
    df1 = pd.DataFrame(index=pd.date_range(datainicial, periods=delta),columns=[c for c in df])
    print(df1.head(20))
    
    #Funcão para transformar a nova matriz para a data contínua
    def dfajust(df_old, df_new):
        for i in range(len(df_old)):
            for j in range(len(df_new)):
                if (df_new.index[j]==df_old.index[i]):
                    df_new.iloc[j]=df_old.iloc[i]
    
        for j in range (len(df_new)):
            if math.isnan(df_new.iloc[j][0]):
                df_new.iloc[j]=df_new.iloc[j-1]
        return df_new
    
    df = dfajust(df,df1)
#    df1=df1[1:]
    df=df1.apply(pd.to_numeric,errors='coerce')
    del df1
    
    df.to_pickle('dataframe_acoes_tempocontinuo')


#
#    #Arrumar problema da variacao
#    
#    #tentativa 1 - nao deu certo
#    df1 = pd.DataFrame(index=pd.date_range(datainicial, periods=delta),columns=[c for c in df])
#    def dfajust1(df_old, df_new):
#        for i in range(len(df_old)):
#            for j in range(len(df_new)):
#                if (df_new.index[j]==df_old.index[i]):
#                    df_new.iloc[j]=df_old.iloc[i]
#    
#        for j in range (len(df_new)):
#            if math.isnan(df_new.iloc[j][0]):
#                var_cols = [col for col in df_new if col.startswith('var_')]
#                df_new.iloc[j]=df_new.iloc[j-1]
#                df_new.iloc[j][var_cols] = 0
#        return df_new
#    
#    df2=dfajust1(df,df1)
#        
#    
#    #tentativa2 - tb nao deu certo
#    def_aux=df[:]
#    list_cot=[]
#    list_varpc=[]
#    for column in df_aux:
#        if column.startswith('cotacao'):
#            list_cot.append(column)
#        if column.startswith('var_pc'):
#            list_varpc.append(column)
#    
#    for column in df_aux:
#        if column.startswith('cotacao'):
#            df_aux.drop([column],axis=1,inplace=True)         
#        if column.startswith('min'):
#            df_aux.drop([column],axis=1,inplace=True)
#        if column.startswith('max'):
#            df_aux.drop([column],axis=1,inplace=True)
#        if column.startswith('vol'):
#            df_aux.drop([column],axis=1,inplace=True)
#        if column.startswith('var_pc'):
#            df_aux.drop([column],axis=1,inplace=True)         
#    list_var=d_auxf.columns       
#    del_df_aux
#    
#    df1=df[0:1]
#
#    for i in list_cot:
#        for j in list_var:
#            df1[j]=df1[i]-df1[i].shift(1)
#            
#    df[0:1]=df1[0:1]
#    
    

    #Escolher variável a ser prevista, dentro da lista escolhida:
    acoes = ['PETR4.SA', 'ABEV3.SA', 'B3SA3.SA', 'BBAS3.SA', 'BBDC3.SA', 'BBSE3.SA', 'VALE3.SA',
             'WEGE3.SA', 'USIM5.SA', 'TAEE11.SA', 'ITUB4.SA', 'KROT3.SA', 'FLRY3.SA',
             'EGIE3.SA', 'CVCB3.SA', 'CIEL3.SA', 'BRFS3.SA', 'ITSA4.SA', 'LAME4.SA']
    
    y_label='ABEV3'
    y=df['cotacao_'+y_label]           #ITAU
    y.describe()

        

    #Graficos analisando a acao escolhida:
    periodo=pd.date_range(datainicial, periods=delta)
 #   periodo=periodo[1:]

    fig2, axes = plt.subplots(nrows=3, ncols=1, figsize=(7, 7))
    axes[0].set_title("Preço")
    axes[0].plot(periodo, df['cotacao_'+y_label], label='Cotação')
    axes[0].plot(periodo, df['min_'+y_label], label='Mínimo')
    axes[0].plot(periodo, df['max_'+y_label], label='Máximo')
    axes[0].legend(loc='best')
    axes[0].set_ylabel("Preço")
    
    axes[1].set_title("Variação do Preço")
    axes[1].plot(periodo, df['var_'+y_label], label='Variação (Abs)')
    axes[1].plot(periodo, df['var_pc_'+y_label], label='Variação (%)')    
    axes[1].legend(loc='best')    
    axes[1].set_ylabel("Variação")
    
    axes[2].set_title("Volume")
    axes[2].plot(periodo, df['vol_'+y_label], label='Volume')
    axes[2].legend(loc='best')
    axes[2].set_ylabel("Volume")

    fig2.tight_layout()
    plt.show()


    #Grafico comparando o preco da acao escolhida com de outros ativos
    plt.figure()
    y.plot(label=y_label)
    df['cotacao_ABEV3'].plot(label='AMBEV')
    df['cotacao_PETR4'].plot(label='Petrobrás')
    df['cotacao_ITUB4'].plot(label='ITAU')
    df['cotacao_CIEL3'].plot(label='CIELO')
    df['cotacao_BRFS3'].plot(label='BRF')
    plt.legend(loc='best')
    plt.ylabel('Preço (R$)')
    plt.title('Preço Ativos')
    plt.show()
    
   
    
    #Verificar Correlacao da Serie
    
    from pandas import DataFrame
    from pandas import concat
    
    values = DataFrame(df['cotacao_ITUB4'].values)
    dataframe = concat([values.shift(1), values], axis=1)
    dataframe.columns = ['t-1','t+1']
    result = dataframe.corr()
    print(result)



    #Para realizar previsao é preciso usar dados passados. 
    #Para isso é preciso criar lag das variáveis
    
    #Para saber quantos lags usar, usamos os plots de autocorrelacao
    
     #1) Autocorrelação PLOT   
    from pandas import Series
    from matplotlib import pyplot
    from pandas.tools.plotting import autocorrelation_plot
    from statsmodels.graphics.tsaplots import plot_acf

    autocorrelation_plot(y)
    plt.title('Gráfico de Autocorrelação do Preço do ITAU')
    plt.show()
    
    
    #2) Lag Plot
    from pandas import Series
    from matplotlib import pyplot
    import statsmodels.api as sm

    plot_acf(y, lags=50)
    pyplot.show()

    
    
    
    #Previsão de X dias:    
    forecastHorizon=30
    
    #A partir dos dois gráficos podemos decidir quantos lags usar
    #Vamos usar 3 dias de lag para rodar os modelos
    
    
    colunas=df.columns
    X = pd.DataFrame(index=pd.date_range(datainicial, periods=delta))
#    X=X[1:]
    

    for i in colunas:
        X[i+'lag1']=df[i].shift(forecastHorizon)
        
    for i in colunas:
        X[i+'lag2']=df[i].shift(forecastHorizon+1)
        
    for i in colunas:
        X[i+'lag3']=df[i].shift(forecastHorizon+2)
    
    lenx=len(X)
    X=X.dropna()
    lenx1=len(X)
    dif_len=lenx-lenx1
    y=y[dif_len:]
   
    periodo=pd.date_range(datainicial, periods=delta)
    treino= periodo[:-forecastHorizon]
    teste=periodo[-forecastHorizon:]

    
    #Dividindo em teste e treino
    train_size = int(len(X) * ((len(X)-forecastHorizon)/len(X)))
    X_treino, X_teste = X[0:train_size], X[train_size:len(X)]
    y_treino, y_teste = y[0:train_size], y[train_size:len(y)]
 
    print('Observations: %d' % (len(X)))
    print('Training Observations: %d' % (len(X_treino)))
    print('Testing Observations: %d' % (len(X_teste)))
    

                           #OLS                
    
    import statsmodels.api as sm
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import accuracy_score



    model2 = sm.OLS(y_treino,X_treino)                  #modelo
    model_fit2 = model2.fit() 
    print (model_fit2.summary())                        #sumário do modelo
    coef2=model_fit2.params
    
    R22=model_fit2.rsquared
       
    # make predictions
    y_predictions2 = model_fit2.predict(X_teste)          #previsão

    EQM2 = mean_squared_error(y_teste, y_predictions2)    #EQM
    resid2 = np.sqrt(EQM2)                                #Resíduo
    print('Test MSE, Residual: %.3f' % EQM2, resid2)
    
    
    accuracy_2 = r2_score(y_teste, y_predictions2)
    R2_2_teste = sm.OLS(y_teste,X_teste).fit().rsquared
    print ('accuracy, R2_teste: %.3f' % accuracy_2, R2_2_teste)
    
    if forecastHorizon>1:
        plt.figure()    
        pyplot.plot(y_treino, label='Treino')
        pyplot.plot(y_teste, color='black', label='Teste')
        pyplot.plot(y_predictions2, color='red', label='Previsão')
        plt.legend(loc='best')
        plt.ylabel('$')
        plt.xticks(rotation=30)
        plt.title('Previsão Preço '+y_label+' (OLS)')  
        pyplot.show()   

    
    
    
    #2)Linear Regression
  
    import numpy as np
    from sklearn.linear_model import LinearRegression
        
    reg = LinearRegression().fit(X_treino, y_treino)
    print(reg.score(X_treino, y_treino))                        #R2 fora da amostra
    print(reg.coef_)                                            #coeficientes
    coefreg=np.transpose(reg.coef_)
    
    R2reg=reg.score(X_treino, y_treino)    
    
    predictionsreg = reg.predict(X_teste)
    y_predictionsreg= pd.DataFrame(predictionsreg, index=teste)   #previsão
    
    EQMreg = mean_squared_error(y_teste, y_predictionsreg)      #EQM
    residreg = np.sqrt(EQMreg)                                #Residuo
    print('Test MSE, residuo: %.3f' % EQMreg,residreg)
        
    accuracy_reg = r2_score(y_teste, y_predictionsreg)
    R2_reg_teste = reg.score(X_teste, y_teste)  
    print ('accuracy, R2_teste: %.3f' % accuracy_reg, R2_reg_teste)
    
    if forecastHorizon>1:    
        plt.figure()    
        pyplot.plot(y_treino, label='Treino')
        pyplot.plot(y_teste, color='black', label='Teste')
        pyplot.plot(y_predictionsreg, color='red', label='Previsão')
        plt.legend(loc='best')
        plt.ylabel('$')
        plt.xticks(rotation=30)
        plt.title('Previsão Preço '+y_label+' (LR)')  
        pyplot.show()   



                                            #Lasso
    
    #1)Lasso normal
    
    from sklearn import linear_model
    
    model3 = linear_model.Lasso( alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,
    normalize=False, positive=False, precompute=False, random_state=None,
    selection='cyclic', tol=0.0001, warm_start=False)
    model_fit3=model3.fit(X_treino,y_treino)
    coef3=model3.coef_
    
    print(model_fit3.coef_)
    print(model_fit3.intercept_) 
    print(model_fit3.score(X_treino,y_treino))

    R23 = model_fit3.score(X_treino,y_treino)
    
        # make predictions
    y_predictions3 = model_fit3.predict(X_teste)
    y_predictions3= pd.DataFrame(y_predictions3, index=teste)   #previsão


    EQM3 = mean_squared_error(y_teste, y_predictions3)
    resid3 = np.sqrt(EQM3)
    print('Test MSE, residuo: %.3f' % EQM3,resid3)
    
    accuracy_3 = r2_score(y_teste, y_predictions3)
    R2_3_teste = model_fit3.score(X_teste, y_teste)  
    print ('accuracy, R2_teste: %.3f' % accuracy_3, R2_3_teste)
    

    if forecastHorizon>1:      
        plt.figure()    
        pyplot.plot(y_treino, label='Treino')
        pyplot.plot(y_teste, color='black', label='Teste')
        pyplot.plot(y_predictions3, color='red', label='Previsão')
        plt.legend(loc='best')
        plt.ylabel('$')
        plt.xticks(rotation=30)
        plt.title('Previsão Preço '+y_label+' (Lasso)')  
        pyplot.show()   
    

    
    #2) Lasso CV
    
    from sklearn.linear_model import LassoCV
    
    model4 = LassoCV(cv=365, random_state=0).fit(X_treino, y_treino)
    print(model4.coef_)
    coef4=model4.coef_
    
    R24 = model4.score(X_treino, y_treino) 


        # make predictions
    y_predictions4 = model4.predict(X_teste)
    y_predictions4= pd.DataFrame(y_predictions4, index=teste)   #previsão


    EQM4 = mean_squared_error(y_teste, y_predictions4)
    resid4 = np.sqrt(EQM4)
    print('Test MSE: %.3f' % EQM4,resid4)
    
    accuracy_4 = r2_score(y_teste, y_predictions4)
    R2_4_teste = model4.score(X_teste, y_teste)  
    print ('accuracy, R2_teste: %.3f' % accuracy_4, R2_4_teste)
    
 
    if forecastHorizon>1:    
        plt.figure()    
        pyplot.plot(y_treino, label='Treino')
        pyplot.plot(y_teste, color='black', label='Teste')
        pyplot.plot(y_predictions4, color='red', label='Previsão')
        plt.legend(loc='best')
        plt.ylabel('$')
        plt.xticks(rotation=30)
        plt.title('Previsão Preço '+y_label+' (Lasso CV)')  
        pyplot.show()   
    
        
    
    
                                        #Lars
                                        
    #1) Lars normal
     
    model5 = linear_model.Lars(n_nonzero_coefs=100)
    model5_fit=model5.fit(X_treino, y_treino)
    print(model5_fit.coef_) 
    coef5=model5_fit.coef_
    
    R25 = model5_fit.score(X_treino, y_treino) 

         # make predictions
    y_predictions5 = model5_fit.predict(X_teste)
    y_predictions5= pd.DataFrame(y_predictions5, index=teste)   #previsão


    EQM5 = mean_squared_error(y_teste, y_predictions5)
    resid5 = np.sqrt(EQM5)
    print('Test MSE: %.3f' % EQM5,resid5)
     
    accuracy_5 = r2_score(y_teste, y_predictions5)
    R2_5_teste = model5_fit.score(X_teste, y_teste)  
    print ('accuracy, R2_teste: %.3f' % accuracy_5, R2_5_teste)
       
    if forecastHorizon>1:    
        plt.figure()    
        pyplot.plot(y_treino, label='Treino')
        pyplot.plot(y_teste, color='black', label='Teste')
        pyplot.plot(y_predictions5, color='red', label='Previsão')
        plt.legend(loc='best')
        plt.ylabel('$')
        plt.xticks(rotation=30)
        plt.title('Previsão Preço '+y_label+' (Lars)')  
        pyplot.show()   

     
    
    
    
    #2) Lasso Lars 
    model6 = linear_model.LassoLars(alpha=0.01).fit(X_treino,y_treino)
    print(model6.coef_) 
    coef6=model6.coef_
    
    R26 = model6.score(X_treino, y_treino) 

    
    y_predictions6 = model6.predict(X_teste)
    y_predictions6= pd.DataFrame(y_predictions6, index=teste)   #previsão


    EQM6 = mean_squared_error(y_teste, y_predictions6)
    resid6 = np.sqrt(EQM6)
    print('Test MSE: %.3f' % EQM6,resid6)
    
    accuracy_6 = r2_score(y_teste, y_predictions6)
    R2_6_teste = model6.score(X_teste, y_teste)  
    print ('accuracy, R2_teste: %.3f' % accuracy_6, R2_6_teste)      

    if forecastHorizon>1:      
        plt.figure()    
        pyplot.plot(y_treino, label='Treino')
        pyplot.plot(y_teste, color='black', label='Teste')
        pyplot.plot(y_predictions6, color='red', label='Previsão')
        plt.legend(loc='best')
        plt.ylabel('$')
        plt.xticks(rotation=30)
        plt.title('Previsão Preço '+y_label+' (Lasso Lars)')  
        pyplot.show()   
    
    
    
    
    #3) Lasso Lars com Cross Validation 
    
    model7 = linear_model.LassoLarsCV(cv=50).fit(X_treino,y_treino)
    print(model7.coef_)
    
    coef7=model7.coef_

    
    R27 = model7.score(X_treino, y_treino) 

    
    y_predictions7 = model7.predict(X_teste)
    y_predictions7= pd.DataFrame(y_predictions7, index=teste)   #previsão

    EQM7 = mean_squared_error(y_teste, y_predictions7)
    resid7 = np.sqrt(EQM7)
    print('Test MSE: %.3f' % EQM7,resid7)
    
    accuracy_7 = r2_score(y_teste, y_predictions7)
    R2_7_teste = model7.score(X_teste, y_teste)  
    print ('accuracy, R2_teste: %.3f' % accuracy_7, R2_7_teste)

    if forecastHorizon>1:   
        plt.figure()    
        pyplot.plot(y_treino, label='Treino')
        pyplot.plot(y_teste, color='black', label='Teste')
        pyplot.plot(y_predictions7, color='red', label='Previsão')
        plt.legend(loc='best')
        plt.ylabel('$')
        plt.xticks(rotation=30)
        plt.title('Previsão Preço '+y_label+' (Lasso Lars CV)')  
        pyplot.show()   
    
    
     
                                #Ridge Regression 
    
    import matplotlib.pyplot as plt
    from sklearn.linear_model import Ridge
    
    model8 = Ridge(alpha=0.1,normalize=True)
    model8_fit=model8.fit(X_treino, y_treino)
    
    coef8=np.transpose(model8_fit.coef_)

        
    R28 = model8_fit.score(X_treino, y_treino) 

    
    y_predictions8 = model8_fit.predict(X_teste)
    y_predictions8= pd.DataFrame(y_predictions8, index=teste)   #previsão

    EQM8 = mean_squared_error(y_teste, y_predictions8)
    resid8 = np.sqrt(EQM8)
    print('Test MSE: %.3f' % EQM8,resid8)
    
    
    accuracy_8 = r2_score(y_teste, y_predictions8)
    R2_8_teste = model8_fit.score(X_teste, y_teste)  
    print ('accuracy, R2_teste: %.3f' % accuracy_8, R2_8_teste)
    
        
    if forecastHorizon>1: 
        plt.figure()    
        pyplot.plot(y_treino, label='Treino')
        pyplot.plot(y_teste, color='black', label='Teste')
        pyplot.plot(y_predictions8, color='red', label='Previsão')
        plt.legend(loc='best')
        plt.ylabel('$')
        plt.xticks(rotation=30)
        plt.title('Previsão Preço '+y_label+' (Ridge)')  
        pyplot.show()   

    

    
                                #ElasticNet 
                                
    #1) ElasticNet normal
   
    from sklearn.linear_model import ElasticNet

    
    model90 = ElasticNet().fit(X_treino,y_treino)
    print(model90.coef_) 

    R290 = model90.score(X_treino, y_treino) 
    coef90=model90.coef_
    
    y_predictions90 = model90.predict(X_teste)
    y_predictions90= pd.DataFrame(y_predictions90, index=teste)   #previsão

    EQM90 = mean_squared_error(y_teste, y_predictions90)
    resid90 = np.sqrt(EQM90)
    print('Test MSE: %.3f' % EQM90,resid90)
    
    accuracy_90 = r2_score(y_teste, y_predictions90)
    R2_90_teste = model90.score(X_teste, y_teste)  
    print ('accuracy, R2_teste: %.3f' % accuracy_90, R2_90_teste)
        
    if forecastHorizon>1:      
        plt.figure()    
        pyplot.plot(y_treino, label='Treino')
        pyplot.plot(y_teste, color='black', label='Teste')
        pyplot.plot(y_predictions90, color='red', label='Previsão')
        plt.legend(loc='best')
        plt.ylabel('$')
        plt.xticks(rotation=30)
        plt.title('Previsão Preço '+y_label+' (Elastic Net)')  
        pyplot.show()   
    
    
    #2) ElasticNetCV
    
    from sklearn.linear_model import ElasticNetCV

        

    model9 = ElasticNetCV(alphas=None, copy_X=True, cv=100, eps=0.001, fit_intercept=True,
                          l1_ratio=0.5, max_iter=1000, n_alphas=100, n_jobs=None,
                          normalize=False, positive=False, precompute='auto', random_state=0,
                          selection='cyclic', tol=0.0001, verbose=0).fit(X_treino,y_treino)
    print(model9.coef_) 

    R29 = model9.score(X_treino, y_treino) 
    coef9=model9.coef_
    
    y_predictions9 = model9.predict(X_teste)
    y_predictions9= pd.DataFrame(y_predictions9, index=teste)   #previsão

    EQM9 = mean_squared_error(y_teste, y_predictions9)
    resid9 = np.sqrt(EQM9)
    print('Test MSE: %.3f' % EQM9,resid9)
    
    accuracy_9 = r2_score(y_teste, y_predictions9)
    R2_9_teste = model9.score(X_teste, y_teste)  
    print ('accuracy, R2_teste: %.3f' % accuracy_9, R2_9_teste)
        
    if forecastHorizon>1:       
        plt.figure()    
        pyplot.plot(y_treino, label='Treino')
        pyplot.plot(y_teste, color='black', label='Teste')
        pyplot.plot(y_predictions90, color='red', label='Previsão')
        plt.legend(loc='best')
        plt.ylabel('$')
        plt.xticks(rotation=30)
        plt.title('Previsão Preço '+y_label+' (Elastic Net CV)')  
        pyplot.show()   

    
                                    #Random Forest  (MELHOR EQM)

    
    from sklearn.ensemble import RandomForestRegressor
    
    model10 = RandomForestRegressor(n_estimators = 1000, random_state = 0).fit(X_treino, y_treino)
    
    print(model10.feature_importances_)
    coef10=model10.feature_importances_
    
    
    R210 = model10.score(X_treino, y_treino) 
    
    y_predictions10 = model10.predict(X_teste)
    y_predictions10= pd.DataFrame(y_predictions10, index=teste)   #previsão

    EQM10 = mean_squared_error(y_teste, y_predictions10)
    resid10 = np.sqrt(EQM10)
    print('Test MSE: %.3f' % EQM10,resid10)
    
    
    accuracy_10 = r2_score(y_teste, y_predictions10)
    R2_10_teste = model10.score(X_teste, y_teste)  
    print ('accuracy, R2_teste: %.3f' % accuracy_10, R2_10_teste)
    
    if forecastHorizon>1: 
        plt.figure()    
        pyplot.plot(y_treino, label='Treino')
        pyplot.plot(y_teste, color='black', label='Teste')
        pyplot.plot(y_predictions10, color='red', label='Previsão')
        plt.legend(loc='best')
        plt.ylabel('$')
        plt.xticks(rotation=30)
        plt.title('Previsão Preço '+y_label+' AMBEV')  
        pyplot.show()   
    
    
    #Analisando os resultados
    
    #Aqui montamos uma planilha com os coeficientes das variaveis
    #Conseguimos ver quais sao as variaveis relevantes para os melhores modelos
    coef = pd.DataFrame(coef2, index=X.columns)
    coef.columns = ['OLS']
    coef['LinearRegression']=coefreg
    coef['Lasso']=coef3
    coef['LassoCV']=coef4
    coef['Lars']=coef5
    coef['LassoLars']=coef6
    coef['LassoLarsCV']=coef7
    coef['Ridge']=coef8
    coef['ElasticNet']=coef90  
    coef['ElasticNetCV']=coef9
    coef['RandomForest']=coef10
    
    
    #Aqui construo uma matriz com os resultados para cada modelo
    #Conferir quem teve o menor EQM/Resíduo
    
    R2_list      = [R22,R2reg, R23,R24,R25,R26,R27,R28,R290, R29,R210] 
  
    EQM_list     = [EQM2,EQMreg, EQM3,EQM4,EQM5,EQM6,EQM7,EQM8,EQM90,EQM9,EQM10]
   
    resid_list    = [resid2,residreg, resid3,resid4,resid5,resid6,resid7,
                    resid8,resid90, resid9,resid10]
   
    accuracy_list = [accuracy_2,accuracy_reg,accuracy_3,accuracy_4,accuracy_5,
                    accuracy_6,accuracy_7,accuracy_8,accuracy_90,accuracy_9,accuracy_10]
    
    R2_test_list = [R2_2_teste,R2_reg_teste, R2_3_teste,R2_4_teste,R2_5_teste,
                    R2_6_teste,R2_7_teste,R2_8_teste,R2_90_teste,R2_9_teste,R2_10_teste]
    

    
    
    index=['R2', 'EQM', 'Resíduo','Accuracy','R2 teste']   
    colunas3 = ['OLS','LinearRegression','Lasso','LassoCV','Lars','LassoLars',
                'LassoLarsCV','Ridge','ElasticNet','ElasticNetCV','RandomForest']
    previsao = pd.DataFrame([R2_list, EQM_list, resid_list, accuracy_list,
                             R2_test_list],index=index, columns=colunas3)
    print(previsao)

    if forecastHorizon==1:
        comparacao = pd.DataFrame(index=colunas3,columns=['Preço Verdadeiro'])
        for i in range(0,11):
            comparacao['Preço Verdadeiro'][i]=y_teste      
        comparacao['Preço Estimado']=0
        comparacao['Preço Estimado'][:1]=y_predictions2
        comparacao['Preço Estimado'][:2]=y_predictionsreg
        comparacao['Preço Estimado'][:3]=y_predictions3
        comparacao['Preço Estimado'][:4]=y_predictions4
        comparacao['Preço Estimado'][:5]=y_predictions5
        comparacao['Preço Estimado'][:6]=y_predictions6
        comparacao['Preço Estimado'][:7]=y_predictions7
        comparacao['Preço Estimado'][:8]=y_predictions8
        comparacao['Preço Estimado'][:9]=y_predictions9
        comparacao['Preço Estimado'][:10]=y_predictions90
        comparacao['Preço Estimado'][:11]=y_predictions10

        


    





