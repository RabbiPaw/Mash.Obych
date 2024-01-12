"""
- страница 4, с помощью которой можно получить 
предсказание соответствующей модели ML (см. п. 1): 
    @ реализовать загрузку файла в формате *.csv
    @ сделать ввод соответствующих данных с использованием 
      интерактивных компонентов и валидации.
"""
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
def metrics(y_test, y_pred):
    return f'\n MAE: {mean_absolute_error(y_test, y_pred)}\n MSE: {mean_squared_error(y_test, y_pred)} \n RMSE: {(mean_squared_error(y_test, y_pred))**0.5}\n MAPE: {(mean_absolute_percentage_error(y_test, y_pred))**0.5} \n R^2: {r2_score(y_test, y_pred)}'
    
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score as r2, mean_squared_error as mse
import pandas as pd
import pickle

st.set_page_config(page_title = "Предсказание")
st.title('Предсказание моделей')

file = st.file_uploader('Загрузите файл в формате *.csv')

df = None if not file else pd.read_csv(file)

def test_model(x, y, model, transformer = None):
    x = x if not transformer else transformer.fit_transform(x)
    
    x_tr, x_test, y_tr, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    
    model = model.fit(x_tr, y_tr)
    
    y_pred = model.predict(x_test)

    return metrics(y_test, y_pred)

model_name = st.selectbox(
  'Выберите модель машинного обучения', 
  [None,
   'BaggingRegressor',
   'CatBoostRegressor',
   'DecisionTreeRegressor',
   'GradientBoostingRegressor',
   'LinearRegressor',
   'StackingRegressor']
)

def null(x): return x.empty if type(x) is pd.DataFrame else not x 

if model_name and not null(df):
  model = pickle.load(open(f'models/{model_name}.pickle', 'rb'))

  transformer = pickle.load(open('models/PolynomialFeatures.pickle', 'rb'))

  column = st.selectbox(
    'Выберите целевую переменную', [None] + df.columns.to_list()
  )

  if column:
    st.markdown('Обучаем модель...')
  
    x, y = df.drop(column,axis=1), df[column]

    result = test_model(x,y,model) if model_name != 'LinearRegressor' else test_model(x,y,model,transformer)

    st.markdown('Обучение завершено!')

    st.markdown(
      f'''
      ### Результаты {model_name}: 
      {result} 
      '''
    )
