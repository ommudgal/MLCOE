import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models import Linear_Regression, Decision_Tree
from functions import *

st.title("Prediction App")
st.subheader("Provide the input values below:")

model_list = {
    "Linear Regression": Linear_Regression,
    "Decision Tree": Decision_Tree,
}
models = st.selectbox("Select the model", model_list.keys())

model_list[models]()
