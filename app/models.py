import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functions import *


def Linear_Regression():
    crop_list = [
        "Arecanut",
        "Arhar/Tur",
        "Castor seed",
        "Coconut ",
        "Cotton(lint)",
        "Dry chillies",
        "Gram",
        "Jute",
        "Linseed",
        "Maize",
        "Mesta",
        "Niger seed",
        "Onion",
        "Other  Rabi pulses",
        "Potato",
        "Rapeseed &Mustard",
        "Rice",
        "Sesamum",
        "Small millets",
        "Sugarcane",
        "Sweet potato",
        "Tapioca",
        "Tobacco",
        "Turmeric",
        "Wheat",
        "Bajra",
        "Black pepper",
        "Cardamom",
        "Coriander",
        "Garlic",
        "Ginger",
        "Groundnut",
        "Horse-gram",
        "Jowar",
        "Ragi",
        "Cashewnut",
        "Banana",
        "Soyabean",
        "Barley",
        "Khesari",
        "Masoor",
        "Moong(Green Gram)",
        "Other Kharif pulses",
        "Safflower",
        "Sannhamp",
        "Sunflower",
        "Urad",
        "Peas & beans (Pulses)",
        "other oilseeds",
        "Other Cereals",
        "Cowpea(Lobia)",
        "Oilseeds total",
        "Guar seed",
        "Other Summer Pulses",
        "Moth",
    ]
    Season_list = [
        "Whole Year ",
        "Kharif     ",
        "Rabi       ",
        "Autumn     ",
        "Summer     ",
        "Winter     ",
    ]
    State_list = [
        "Assam",
        "Karnataka",
        "Kerala",
        "Meghalaya",
        "West Bengal",
        "Puducherry",
        "Goa",
        "Andhra Pradesh",
        "Tamil Nadu",
        "Odisha",
        "Bihar",
        "Gujarat",
        "Madhya Pradesh",
        "Maharashtra",
        "Mizoram",
        "Punjab",
        "Uttar Pradesh",
        "Haryana",
        "Himachal Pradesh",
        "Tripura",
        "Nagaland",
        "Chhattisgarh",
        "Uttarakhand",
        "Jharkhand",
        "Delhi",
        "Manipur",
        "Jammu and Kashmir",
        "Telangana",
        "Arunachal Pradesh",
        "Sikkim",
    ]
    crop_list.sort()
    crop = st.selectbox("Crop", crop_list)
    year = st.number_input("Year")
    season = st.selectbox("Season", Season_list)
    state = st.selectbox("State", State_list)
    area = st.number_input("Area")
    production = st.number_input("Production")
    Annual_rainfall = st.number_input("Annual rainfall")

    if st.button("Predict"):
        data = [
            crop,
            year,
            season,
            state,
            area,
            production,
            Annual_rainfall,
        ]

        dataset = pd.read_csv(r"D:\Projects\MLCOE\Notebooks\data\crop_yield.csv")
        dataset = dataset.drop(columns=["Yield", "Fertilizer", "Pesticide"])
        dataset = pd.concat([dataset, pd.DataFrame([data], columns=dataset.columns)])
        from sklearn.preprocessing import LabelEncoder

        label_encoder = LabelEncoder()

        dataset2 = dataset.copy()
        numerical_cols = dataset.select_dtypes(include=["float64", "int64"]).columns
        categorical_cols = dataset.select_dtypes(include=["object", "category"]).columns
        for col in categorical_cols:
            dataset2[col] = label_encoder.fit_transform(dataset[col])

        model0 = joblib.load(r"D:\Projects\MLCOE\Notebooks\models\linear_model.joblib")
        y_pred = model0.predict(dataset2.iloc[-1].values.reshape(1, -1))
        st.write(f"Predicted Yield: {y_pred[0]}")


def Decision_Tree():
    dataset = pd.read_csv(r"D:\Projects\MLCOE\Notebooks\data\dataset2.csv")
    gender_list = ["Male", "Female"]
    Married_list = ["Yes", "No"]
    Education_list = ["Graduate", "Not Graduate"]
    Self_Employed_list = ["Yes", "No"]
    Property_Area_list = ["Urban", "Semiurban", "Rural"]
    Dependents_list = ["0", "1", "2", "3+"]
    Credit_History_list = ["1.0", "0.0"]
    dataset = dataset.drop(columns=["Loan_ID", "Loan_Status"])

    gender = st.selectbox("Gender", gender_list)
    Married = st.selectbox("Married", Married_list)
    Dependents = st.selectbox("Dependents", Dependents_list)
    Education = st.selectbox("Education", Education_list)
    Self_Employed = st.selectbox("Self Employed", Self_Employed_list)
    ApplicantIncome = st.number_input("Applicant Income")
    CoapplicantIncome = st.number_input("Coapplicant Income")
    LoanAmount = st.number_input("Loan Amount")
    Loan_Amount_Term = st.number_input("Loan Amount Term")
    Credit_History = st.selectbox("Credit History", Credit_History_list)
    Property_Area = st.selectbox("Property Area", Property_Area_list)

    if st.button("Predict"):
        data = [
            gender,
            Married,
            Dependents,
            Education,
            Self_Employed,
            ApplicantIncome,
            CoapplicantIncome,
            LoanAmount,
            Loan_Amount_Term,
            Credit_History,
            Property_Area,
        ]

        from sklearn.preprocessing import LabelEncoder

        label_encoder = LabelEncoder()

        dataset2 = dataset.copy()
        numerical_cols = dataset.select_dtypes(include=["float64", "int64"]).columns
        categorical_cols = dataset.select_dtypes(include=["object", "category"]).columns
        for col in categorical_cols:
            dataset2[col] = label_encoder.fit_transform(dataset[col])

        model0 = joblib.load(r"D:\Projects\MLCOE\Notebooks\models\Decision_Tree.joblib")
        y_pred = model0.predict(dataset2.iloc[-1].values.reshape(1, -1))
        st.write(f"Predicted Yield: {y_pred[0]}")
