""" Задание:
  - страница 1 с информацией о разработчике моделей ML:
    @ ФИО, 
    @ номер учебной группы
    @ цветная фотография, 
    @ тема РГР)
"""
# source venv/Scripts/activate
# streamlit run notebooks/RGR/Autor.py

import streamlit as stm 
  
stm.set_page_config(page_title = "Autor Info") 
stm.title("Информация об авторе")
stm.markdown('''
  ### Имя: Салабун Святослав Игоревич
  ### Группа: ФИТ-222
  ### Тема РГР: Разработка дашборда \
  для вывода моделей ML и анализа данных
''')
