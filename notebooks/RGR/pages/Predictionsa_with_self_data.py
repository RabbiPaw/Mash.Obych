import streamlit as st
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

def metrics(y_test, y_pred):
    return f'\n MAE: {round(mean_absolute_error(y_test, y_pred),3)}||MSE: {round(mean_squared_error(y_test, y_pred),3)}|| RMSE: {round((mean_squared_error(y_test, y_pred))**0.5,3)}|| MAPE: {round((mean_absolute_percentage_error(y_test, y_pred))**0.5 , 3)}'

def test_model(x, y, model, transformer = None):
    
    y_pred = model.predict(x)
    return metrics(y, y_pred)

st.set_page_config(page_title = "Предсказание по вашим данным")
st.title('Предсказание моделей по вашим данным')

price = st.number_input("Введите цену")

latitude = st.number_input("Введите координаты по широте")

longitude = st.number_input("Введите координаты по долготе")

bathrooms = st.slider("Выберите число ванных комнат",0,0,10)

status = st.radio(
    "Выберите cостояние дома",
    ["Готов к переезду", "В ремонете/перестройка"])

furnished_status = st.radio(
    "Выберите статус отделки дома",
    ["Полностью обустроен", "Частично обустроен", "Не обустроен"])

if status == "Готов к переезду":
    status = 1
else:
    status = 0
    
if furnished_status == "Полностью обустроен":
    furnished_status = 2
elif furnished_status == "Частично обустроен":
    furnished_status = 1
else:
    furnished_status = 0
        
df = pd.DataFrame({"price":[price],
                    "latitude":[latitude],
                    "longitude":[longitude],
                    "bathrooms":[bathrooms],
                    "status":[status],
                    "furnished_status":[furnished_status]})
    
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
  model = pickle.loads(open(f'models/{model_name}.pickle'))

  transformer = pickle.loads(open('models/PolynomialFeatures.pickle'))

  column = st.selectbox(
    'Выберите целевую переменную', [None] + df.columns.to_list()
  )

  if column:
    st.markdown('Данные приняты, готовим предсказание...')
    
    x, y = df.drop(column,axis=1), df[column]

    result = test_model(x,y,model) if model_name != 'LinearRegressor' else test_model(x,y,model,transformer)

    st.markdown('Готово!')

    st.markdown(
      f'''
      ### Результаты {model_name}: 
      {result} 
      '''
    )
          
