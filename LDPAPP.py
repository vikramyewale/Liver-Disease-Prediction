
import streamlit as st
import pandas as pd
import joblib
import requests
from streamlit_lottie import st_lottie

st.set_page_config(page_title="My webpage", layout="wide")

def load_lottieurl(url):
    r= requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_ssjAlSigs7.json")

# Load pre-trained classification model
model = joblib.load("C:/Users/vikra/Downloads/random_forest (1).joblib")

# Add custom CSS style to change background color
st.markdown(
    """
    <style>
    body {
        background-color: #ff0000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Define function to map predicted class numbers to their corresponding names
def class_name(p):
    if p == 1:
        return 'no_disease'
    elif p == 2:
        return 'suspect_disease'
    elif p == 3:
        return 'hepatitis'
    elif p == 4:
        return 'fibrosis'
    elif p == 5:
        return 'cirrhosis'
    else:
        return "nothing"
    


with st.container():
    st.write("----")
    left_column, right_column = st.columns(2)
    with left_column:        
        st.title('Model Deployment' )
        st.header("Lever Disease Prediction Model")
    with right_column:
        st_lottie(lottie_coding, height=300 , key = "coding")

# Define Streamlit app
def app():
    #st.title("Predict Liver Disease")
    # Add UI elements to get input values for the 12 independent variables
    var1_age = st.number_input("age", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
    var2 = st.radio("gender", options=["Male", "Female"], index=0)
    gender_num = 1 if var2 == "Male" else 0
    var2_albumin = st.number_input("albumin", min_value=10.0, max_value=85.0, value=28.0, step=0.1)
    var3_alkaline_phosphatase = st.number_input("alkaline_phosphatase", min_value=5.0, max_value=500.0, value=5.0, step=0.1)
    var4_alanine_aminotransferase = st.number_input("alanine_aminotransferase", min_value=0.0, max_value=500.0, value=5.0, step=0.1)
    var5_aspartate_aminotransferase = st.number_input("aspartate_aminotransferase", min_value=5.0, max_value=400.0, value=5.0, step=0.1)
    var6_bilirubin = st.number_input("bilirubin", min_value=0.0, max_value=256.0, value=5.0, step=0.1)
    var7_cholinesterase = st.number_input("cholinesterase", min_value=0.0, max_value=17.0, value=5.0, step=0.1)
    var8_cholesterol = st.number_input("cholesterol", min_value=1.0, max_value=10.0, value=5.0, step=0.1)
    var9_creatinina = st.number_input("creatinina", min_value=10.0, max_value=1100.0, value=45.0, step=0.1)
    var10_gamma_glutamyl_transferase = st.number_input("gamma_glutamyl_transferase", min_value=2.0, max_value=700.0, value=120.0, step=0.1)
    var11_protein = st.number_input("protein", min_value=40.0, max_value=100.0, value=48.0, step=0.1)

    # Add the remaining variables here

    # Create a Pandas DataFrame with the user input values for all 12 independent variables
    user_input = pd.DataFrame({
    "var1_age": [var1_age],
    "Var2": [gender_num],
    "var2_albumin": [var2_albumin],
    "var3_alkaline_phosphatase": [var3_alkaline_phosphatase],
    "var4_alanine_aminotransferase": [var4_alanine_aminotransferase],
    "var5_aspartate_aminotransferase": [var5_aspartate_aminotransferase],
    "var6_bilirubin": [var6_bilirubin],
    "var7_cholinesterase": [var7_cholinesterase],
    "var8_cholesterol": [var8_cholesterol],
    "var9_creatinina": [var9_creatinina],
    "var10_gamma_glutamyl_transferase": [var10_gamma_glutamyl_transferase],
    "var11_protein": [var11_protein]
    # Add the remaining variables here
    })

    st.button('Predict')
    # Use the pre-trained model to make a prediction based on user input
    prediction = model.predict(user_input)

    # Map the predicted class numbers to their corresponding names using the class_name function
    class_prediction = class_name(prediction[0])

    # Display the prediction to the user
    st.write(f"The predicted class is : {class_prediction}")
    #st.bar_chart(pd.DataFrame(model.predict_proba(user_input), columns=model.classes_), height=300)

# Run the Streamlit app
if __name__ == "__main__":
    app()
