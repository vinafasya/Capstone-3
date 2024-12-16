import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl
import sklearn
import xgboost
import numpy as np

df = pd.read_csv('data_california_clean.csv')
tab_1, tab_2 = st.tabs(["Price Prediction", "Model Evaluation"])

@st.cache_data()  # Menyimpan model di cache untuk performa
def load_classification_model():
    with open('model_xg.pkl', 'rb') as file:
        model = pkl.load(file)
    
    with open('scaler (1).pkl', 'rb') as scale:
        scaler = pkl.load(scale)
    return model, scaler

# Tab 1
with tab_1:
    st.title('California House Price Prediction')
    st.header('Predicting Housing Prices in California')
    st.write('This project aims to predict house prices in California as part of a Machine Learning learning journey.')

    model, scaler = load_classification_model()

    house_median_age = st.number_input('Enter the value of house median age', value=0)
    st.write('Please tick one of the ocean proximity type.')
    less_1H_OCEAN = st.checkbox("Ocean proximity is <1H ocean.", value=True)
    inland_ocean = st.checkbox("Ocean proximity is inland.", value=True)
    near_bay = st.checkbox("Ocean proximity is s near bay.", value=True)
    near_ocean = st.checkbox("Ocean proximity is near ocean.", value=True)
    st.write('')
    location_cluster = st.selectbox('Select location cluster: ',
                                    options=[0, 1, 2, 3, 4])
    normalized_rooms_per_household = st.number_input('Enter the rooms per household', value=0)
    normalized_population_per_household = st.number_input('Enter population per household', value=0)
    adjusted_median_income = st.number_input('Enter median income', value=0)
    capped_value_flag = st.selectbox('Select capped value flag: ',
                                    options=[0, 1])
    st.write('Capped Value Flag: Choose 1 if median house value is exactly 500,000, choose 0 for other nominal.')
    bedroom_to_room_ratio_scaled = st.number_input('Enter bedroom to room ratio', value=0)

    df_input = [house_median_age, less_1H_OCEAN, inland_ocean, near_bay, near_ocean,
                location_cluster, normalized_rooms_per_household, normalized_population_per_household,
                adjusted_median_income, capped_value_flag, bedroom_to_room_ratio_scaled]
    
   # Scaling data input
    scaled_input = scaler.transform([df_input])

    if st.button('Prediction'):
        if df_input:
            prediction = model.predict(scaled_input)
            st.write(f'Prediksi: {prediction}')
        else:
            st.write('Input failed')

with tab_2:
    st.title('California House Price Prediction')
    st.header('Model Evaluation Result')
    st.write('Model: XGBoost Regressor')

    df_eval = pd.DataFrame({
        'Root Mean Square Error': [49807.99899292695],
        'MAE': [34724.0443514814],
        'MAPE': [20.63013075358951]
    })

    st.table(df_eval.T)

    st.write('Correlation between features and House Price')
    correlations = df.corr()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(data=correlations, annot=True,  cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    st.table(correlations)