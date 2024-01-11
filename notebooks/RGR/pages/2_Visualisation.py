"""
- страница 3 с визуализациями зависимостей в наборе данных
    @ визуализации Matplotlib, Seaborn
    @ минимум 4 различных вида визуализаций
"""
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

st.set_page_config(page_title = "Визуализация")
st.title('Визуализация зависимостей в наборе данных')

df = pd.read_csv("data/DataSet3_Ready.csv")

########### Круговая диаграмма ###########
if st.button('Круговая диаграмма'):
    fig, ax = plt.subplots()

    ax.pie(
        x=list(df['status'].value_counts()), 
        labels=['ReadyToMove','UnderConstraction','Unknown'],
        autopct='%1.1f%%',
        colors=['red','green']
    )

    plt.title('Круговая диаграмма коробки передач')
    
    st.pyplot(fig)

########### Гистограмма ###########
if st.button('Гистограмма цены домов'):
    fig,ax = plt.subplots()

    ax.hist(df['price'])
    
    plt.xlabel('Цена домов')
    
    plt.ylabel('Кол-во домов')
    
    plt.title('Гистограмма цены домов')
    
    st.pyplot(fig)

########### Тепловая карта корреляции ###########
if st.button('Тепловая карта корреляции'):
    fig = plt.figure()

    sns.heatmap(
        df.corr(),
        cmap="YlGnBu", 
        annot=True
    )

    plt.title('Тепловая карта корреляции\n')

    st.pyplot(fig)

########### Зависимость цены от года ###########
def correlation(df,parametr,target):
    sns.lmplot(x=parametr,y=target,data=df)
    plt.title(f'Зависимость {target} от {parametr}')

if st.button('График зависимости цены от статуса отделки дома'):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(correlation(df,'furnished_status','price'))
