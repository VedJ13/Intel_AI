# import streamlit as st
# import joblib

# # Load model and vectorizer
# model = joblib.load("model.pkl")
# vectorizer = joblib.load("vectorizer.pkl")

# st.title("Animal Health Condition Classifier")

# user_input = st.text_area("Enter clinical notes or animal symptoms:")

# if st.button("Predict"):
#     if user_input.strip() == "":
#         st.warning("Please enter some text.")
#     else:
#         vect_input = vectorizer.transform([user_input])
#         prediction = model.predict(vect_input)[0]
#         st.success(f"Predicted Condition: **{prediction}**")


import pandas as pd
from datetime import datetime

# Initialize session state
if "record_list" not in st.session_state:
    st.session_state["record_list"] = []

# After prediction:
record = {
    "DateTime": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "Notes": user_input,
    "Condition": prediction,
}
st.session_state["record_list"].append(record)

# Display structured data
df = pd.DataFrame(st.session_state["record_list"])
st.write("### Structured Records", df)

import matplotlib.pyplot as plt

# Most common conditions
if not df.empty:
    condition_counts = df['Condition'].value_counts()
    st.write("### Condition Frequency")
    st.bar_chart(condition_counts)

csv = df.to_csv(index=False).encode('utf-8')
st.download_button("Download Structured Data as CSV", csv, "structured_data.csv", "text/csv")

if condition_counts.max() > 5:
    st.warning(f"⚠️ High frequency of '{condition_counts.idxmax()}' — investigate product usage or quality in affected areas.")
