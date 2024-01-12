import streamlit as st
import pandas as pd

def test_model(x, y, model, transformer = None):
    x = x if not transformer else transformer.fit_transform(x)
    
    x_tr, x_test, y_tr, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    
    model = model.fit(x_tr, y_tr)
    
    y_pred = model.predict(x_test)

    return metrics(y_test, y_pred)

st.set_page_config(page_title = "Предсказание по вашим данным")
st.title('Предсказание моделей по вашим данным')

df = pd.DataFrame({"price": [],
                    "latitude": [],
                    "longitude": [],
                    "bathrooms": [],
                    "status": [],
                    "furnished_status": []})

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

st.button("Сброс своей базы данных", type="primary")

if st.button("Добавить данные в собственную базу данных"):
    data = [price,latitude,longitude,bathrooms,status,furnished_status]
    df = df.loc[len(df.index)] = data

if st.button("Показать данные"):
    st.write(df)
    
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
if st.button("Предсказать"):
    if model_name and not null(df):
      model = pickle.load(open(f'models/{model_name}.pickle', 'rb'))

      transformer = pickle.load(open('models/PolynomialFeatures.pickle', 'rb'))
    
      column = st.selectbox(
        'Выберите целевую переменную', [None] + df.columns.to_list()
      )
    
      if column:
        st.markdown('Идёт обучение модели...')
      
        x, y = df.drop(column,axis=1), df[column]
    
        result = test_model(x,y,model) if model_name != 'LinearRegressor' else test_model(x,y,model,transformer)
    
        st.markdown('Обучение завершено!')
    
        st.markdown(
          f'''
          ### Результаты {model_name}: 
          {result} 
          '''
        )



        
        
