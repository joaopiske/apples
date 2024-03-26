import pandas as pd
import streamlit as st
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# importar dados e fazer limpeza
df = pd.read_csv('https://raw.githubusercontent.com/joaopiske/apples/main/apple_quality.csv')
df.drop(columns=['A_id'], inplace=True)
df.drop(df.index[-1], inplace=True)

# formatar pagina
titulo = st.write('Análise de Maçãs')
tab_tab, tab_prev = st.tabs(['Tabela', 'Previsão'])

# criar def
def coluna_icone(check):
    if check == 'good':
        return '✅'
    return '❌'

# adicionar conteúdo
with tab_tab:
    df['Simbol'] = df['Quality'].apply(coluna_icone)
    st.write(df)

with tab_prev:
    X = df.iloc[:, 0:-2].values
    y = df.iloc[:, -2].values

    testes = {'Support Vector Machine': SVC(),
          'Naive Bayes': GaussianNB(),
          'K-Nearest Neighbors': KNeighborsClassifier(),
          'Random Forest': RandomForestClassifier(),
          'Decision Tree': DecisionTreeClassifier()}

    for teste in testes.values():
        teste.fit(X,y)

    with st.form('form1'):
        col1, col2, col3 = st.columns(3)
        nova_predicao = {}

        with col1:
            nova_predicao['Size'] = st.number_input('Size')
            nova_predicao['Weight'] = st.number_input('Weight')
            nova_predicao['Sweetness'] = st.number_input('Sweetness')

        with col2:
            nova_predicao['Crunchiness'] = st.number_input('Crunchiness')
            nova_predicao['Juiceness'] = st.number_input('Juiceness')
            nova_predicao['Ripeness'] = st.number_input('Ripeness')

        with col3:
            nova_predicao['Acidity'] = st.number_input('Acidity')
            botao_predicao = st.form_submit_button('Previsao')

    if botao_predicao:
        df_nova_predicao = pd.DataFrame([nova_predicao])

        col1, col2 = st.columns(2)
        predicoes = []
        with col1:
            st.write('Previsão!')
            for nome, teste in testes.items():
                resultado_predicao = teste.predict(df_nova_predicao)[0]
                icone = '❌'
                if resultado_predicao == 'good':
                    icone = '✅'
                st.write(f'Modelo: {nome} - {resultado_predicao} {icone}')

        with col2:
            st.write('Maior probabilidade')
            if predicoes.count('good') > 2:
                st.write('Good ✅')
            else:
                st.write('Bad ❌')












