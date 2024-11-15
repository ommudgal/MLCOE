import streamlit as st
import pandas as pd
import numpy as np
from functions import *

st.title("Crop Yield Prediction")

data_load_state = st.text("Loading data...")

data = load_data(r"D:\Projects\MLCOE\Notebooks\data\dataset2.csv", 10)

st.subheader("Dataset")
st.write(data)
data_load_state.text("Loading data...done!")
