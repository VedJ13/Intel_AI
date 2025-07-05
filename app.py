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


# import pandas as pd
# from datetime import datetime

# # Initialize session state
# if "record_list" not in st.session_state:
#     st.session_state["record_list"] = []

# # After prediction:
# record = {
#     "DateTime": datetime.now().strftime("%Y-%m-%d %H:%M"),
#     "Notes": user_input,
#     "Condition": prediction,
# }
# st.session_state["record_list"].append(record)

# # Display structured data
# df = pd.DataFrame(st.session_state["record_list"])
# st.write("### Structured Records", df)

# import matplotlib.pyplot as plt

# # Most common conditions
# if not df.empty:
#     condition_counts = df['Condition'].value_counts()
#     st.write("### Condition Frequency")
#     st.bar_chart(condition_counts)

# csv = df.to_csv(index=False).encode('utf-8')
# st.download_button("Download Structured Data as CSV", csv, "structured_data.csv", "text/csv")

# if condition_counts.max() > 5:
#     st.warning(f"⚠️ High frequency of '{condition_counts.idxmax()}' — investigate product usage or quality in affected areas.")


# import streamlit as st
# import pandas as pd
# import joblib
# import matplotlib.pyplot as plt
# from datetime import datetime

# # Load the model and vectorizer
# model = joblib.load("model.pkl")
# vectorizer = joblib.load("vectorizer.pkl")

# # Create a list to store structured data
# if "records" not in st.session_state:
#     st.session_state.records = []

# st.title("🐾 Veterinary Health Assistant")
# st.markdown("**AI for Manufacturing - Veesure Animal Health**")
# st.write("Enter clinical notes or animal symptoms:")

# # User input
# user_input = st.text_area("Clinical Notes / Animal Symptoms", height=100)

# if st.button("Predict"):
#     if user_input.strip() == "":
#         st.warning("Please enter some text.")
#     else:
#         # Predict condition
#         vectorized = vectorizer.transform([user_input])
#         prediction = model.predict(vectorized)[0]

#         # Store structured record
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         st.session_state.records.append({
#             "Timestamp": timestamp,
#             "Symptoms": user_input,
#             "Predicted Condition": prediction
#         })

#         # Show prediction
#         st.success(f"🧪 Predicted Condition: **{prediction}**")

# # Show structured table
# if st.session_state.records:
#     st.subheader("📋 Structured Data Log")
#     df = pd.DataFrame(st.session_state.records)
#     st.dataframe(df)

#     # Show actionable insights: Condition frequency
#     st.subheader("📊 Most Predicted Conditions")
#     condition_counts = df["Predicted Condition"].value_counts()
    
#     fig, ax = plt.subplots()
#     condition_counts.plot(kind='bar', ax=ax, color='skyblue')
#     ax.set_xlabel("Condition")
#     ax.set_ylabel("Frequency")
#     ax.set_title("Frequency of Predicted Conditions")
#     st.pyplot(fig)


import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(page_title="Animal Symptom Classifier", layout="centered")
st.title("🐾 AI for Animal Symptom Analysis")

st.markdown("Enter animal symptoms or clinical notes to get predictions, download structured data, and view analytics.")

# File upload for multiple inputs
st.write("### Upload a Text File (.txt) with Multiple Records")
uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])

input_texts = []

if uploaded_file is not None:
    input_texts = uploaded_file.read().decode("utf-8").splitlines()

# Manual text input
st.write("### Or Enter a Single Symptom / Note")
user_input = st.text_input("Enter symptom/clinical note here:")

if user_input:
    input_texts.append(user_input)

records = []

if st.button("Predict Conditions"):
    if not input_texts:
        st.warning("Please upload a file or enter text manually.")
    else:
        for text in input_texts:
            if text.strip():
                X = vectorizer.transform([text])
                prediction = model.predict(X)[0]
                record = {
                    "Text": text,
                    "Predicted Condition": prediction,
                    "Record Type": "User Input" if text == user_input else "File Upload"
                }
                records.append(record)

        if records:
            # Convert to DataFrame
            df = pd.DataFrame(records)

            st.success("✅ Prediction Complete!")
            st.write("### Structured Data Table")
            st.dataframe(df)

            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Structured Data as CSV",
                data=csv,
                file_name="structured_data.csv",
                mime="text/csv"
            )

            # Bar chart for prediction count
            st.write("### 📊 Prediction Overview")
            chart_data = df["Predicted Condition"].value_counts()
            fig, ax = plt.subplots()
            chart_data.plot(kind="bar", color="skyblue", ax=ax)
            plt.xlabel("Predicted Condition")
            plt.ylabel("Count")
            plt.title("Distribution of Predicted Conditions")
            st.pyplot(fig)
