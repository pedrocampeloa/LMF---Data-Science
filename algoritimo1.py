import requests
import bs4 as bs
import pandas as pd
import pickle
import sys
from datetime import datetime
import math

# # Puxa o nome dos 1965 ativos que vieram do site da uol e estao no arquivo chamado ativos
# with open('ativos', 'rb') as f:
#     ativos = pickle.load(f)
# ########


# Lista de acoes que o programa vai puxar
acoes = ['PETR4.SA', 'ABEV3.SA', 'B3SA3.SA', 'BBAS3.SA', 'BBDC3.SA', 'BBSE3.SA', 'VALE3.SA',
         'WEGE3.SA', 'USIM5.SA', 'TAEE11.SA', 'ITUB4.SA', 'MGLU3.SA', 'KROT3.SA', 'FLRY3.SA',
         'EGIE3.SA', 'CVCB3.SA', 'CIEL3.SA', 'BRFS3.SA', 'ITSA4.SA', 'LAME4.SA']
#########



# Nome base das colunas, na hora o codigo ja adiciona o ticker da acao no final
columns = ['data', 'cotacao', 'min', 'max', 'variacao', 'variacao_percent', 'volume']
#########

diainicial = input('Digite o dia de ínicio de sua série: ')
mesinicial = input('Digite o mês de ínicio de sua série: ')
anoinicial = input('Digite o ano de ínicio de sua série: ')
diafinal = input('Digite o dia do fim de sua série: ')
mesfinal= input('Digite o mês do fim de sua série: ')
anofinal = input('Digite o ano do fim de sua série: ')


# URL que vai puxar o payload base. Se for mudar as datas tem que mudar aqui
url = 'http://cotacoes.economia.uol.com.br/acao/cotacoes-historicas.html'
base_payload = {'codigo': '', 'beginDay': diainicial,
           'beginMonth': mesinicial, 'beginYear': anoinicial,
           'endDay': diafinal, 'endMonth': mesfinal,
           'endYear': anofinal, 'page': 1, 'size': 200}
#########



# Funcao que recebe um payload e retorna um dataframe com os
# dados da acao passado no codigo do payload
def get_data(payload):
    global url
    global columns
    stock_columns = [x + '_' + payload['codigo'][:-3] for x in columns]
    df = pd.DataFrame(columns=stock_columns)
    print('Pegando dados da acao ' + payload['codigo'] + ' (', end='', flush=True)
    while True:
        html = requests.get(url, params=payload)
        soup = bs.BeautifulSoup(html.text, 'html5lib')
        tables = soup.findAll('table', {'id': 'tblInterday', 'class': 'tblCotacoes'})

        if len(tables) == 0:
            if payload['page'] == 1:
                print('Acao nao encontrada')
            break

        for table in tables:
            table_values = table.findChildren('tbody')
            trs = table_values[0].findAll('tr')
            for tr in trs:
                values = [td.text for td in tr.findAll('td')]
                df.loc[len(df)] = values
        print('.', end='', flush=True)
        payload['page'] += 1
    print(') DONE')
    return df

# Rodando a funcao para todas as acoes da lista acoes e guarndando os resultados em dfs
dfs = []
for i in acoes:
    base_payload['codigo'] = i
    base_payload['page'] = 1
    a = get_data(base_payload)
    dfs.append(a)



# concatenando todos os dataframes em dfs e alvando em um arquivo chamado dataframe_acoes
result = pd.concat(dfs, axis=1)
print(result.head())
result.to_pickle('dataframe_acoes')

# O arquivo foi salvo no formato pickle pois nesse formato a leitura e escrita eh mais rapida
# se achar melhor pode salvar um csv tambem

# Pra ler do arquivo pickle basta usar
df = pd.read_pickle('dataframe_acoes')

# print(df)


#Ajeitar a base de dados já importadada
#Tirar as pontos dos milhares e colocamos os pontos nas casas decimais
for column in df:
    df[column] = df[column].str.replace(".","")

for column in df:
    df[column] = df[column].str.replace(",",".")
# print(df.head())

#Colocando a data como index
lista = list(df.ix[:,0])
lista = lista[0:-1]
lista1 = []
for i in range(0,len(lista)):
    lista1.append(datetime.strptime(lista[i], '%d/%m/%Y'))

#Aqui nós tiramos o último valor, pois dia 01/01/2017 é feriado e não tinham valores
df = df[:-1]

df['index']=lista1
df=df.set_index('index')                                      #colocando a data ajustada como index
df.sort_values(by=['index'], inplace=True, ascending=True)     #ordenando do menor para o maior
print(df.head())


#Transformando em números
df = df.apply(pd.to_numeric,errors='coerce')

#Retirando as demais colunas de datas
for column in df:
    if column.startswith('data'):
        df.drop([column],axis=1,inplace=True)
print(df)



#Transformar o tempo em contínuo
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

df1 = dfajust(df,df1)
print(df1.head(15))
