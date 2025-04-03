import streamlit as st
import pandas as pd
import numpy as np
import onnxruntime as ort
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    session = ort.InferenceSession("credit_card_fraud_detection_model.onnx")
    return session

def preprocess_data(df, preprocessor=None):
    columns_to_drop = ['step', 'nameOrig', 'nameDest', 'isFraud', 'isFlaggedFraud']
    df_processed = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    categorical_cols = ['type']
    numerical_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

    if preprocessor is None:
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols),
                ('num', StandardScaler(), numerical_cols)
            ])
        preprocessor.fit(df_processed)

    X_processed = preprocessor.transform(df_processed)
    return X_processed, preprocessor

st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;  /* Light background for main area */
        padding: 20px;
    }
    .stButton>button {
        background-color: #4CAF50;  /* Green button */
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
    }
    .stFileUploader {
        background-color: #ffffff;  /* Lighter uploader background */
        border: 2px dashed #cccccc;  /* Light gray dashed border */
        border-radius: 8px;
        padding: 10px;
    }
    h1, h2 {
        color: #333333;  /* Dark headers */
        font-family: 'Arial', sans-serif;
    }
    .stDataFrame {
        border: 1px solid #cccccc;  /* Lighter border for dataframe */
        border-radius: 8px;
    }
    .stExpander {
        background-color: #ffffff;  /* Light expander background */
        border: 1px solid #cccccc;
        border-radius: 8px;
    }
    .stSidebar .sidebar-content {
        background-color: #f9f9f9;  /* Light sidebar */
    }
    </style>
""", unsafe_allow_html=True)

st.title("💳 Credit Card Fraud Detection")
st.markdown("Upload a CSV file to predict fraudulent transactions with our ONNX-powered model.", unsafe_allow_html=True)

with st.sidebar:
    st.header("Instructions")
    st.write("""
        Upload a CSV file with the following columns:
        - `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`
        - `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`, `isFlaggedFraud`
        - `isFraud` is optional and ignored.
    """)
    st.write("The app will display predictions for each transaction.")

with st.container():
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", help="Upload your transaction data here")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Data Preview")
        st.dataframe(df.head(), use_container_width=True)

    X_processed, preprocessor = preprocess_data(df)

    model_session = load_model()
    input_name = model_session.get_inputs()[0].name
    output_name = model_session.get_outputs()[0].name

    X_processed = X_processed.astype(np.float32)
    predictions_prob = model_session.run([output_name], {input_name: X_processed})[0]
    predictions = (predictions_prob >= 0.5).astype(int)

    with col2:
        st.subheader("Prediction Results")
        with st.expander("View All Transactions", expanded=True):
            for i in range(len(df)):
                st.markdown(f"**Transaction {i+1}:**")
                st.write(f"  Type: {df['type'][i]}, Amount: ${df['amount'][i]:.2f}")
                st.write(f"  Predicted Fraud Probability: {predictions_prob[i][0]:.4f}")
                st.write(f"  Predicted Class: {'Fraud' if predictions[i][0] == 1 else 'Not Fraud'}")
                st.divider()

        results_df = df[['type', 'amount']].copy()
        results_df['Fraud_Probability'] = predictions_prob
        results_df['Predicted_Class'] = ['Fraud' if pred == 1 else 'Not Fraud' for pred in predictions.flatten()]
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Predictions",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv",
            use_container_width=True
        )
