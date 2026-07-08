import os
import requests
import streamlit as st

# ── Configuration (Set these as secrets/environment variables) ──────────
# In Streamlit, you can use st.secrets["IBM_API_KEY"] or os.environ
IBM_API_KEY = os.environ.get("IBM_API_KEY")           
DEPLOYMENT_ID = os.environ.get("DEPLOYMENT_ID", "019f0900-2d1a-73af-8b39-df79761f6bd9")
SPACE_ID = os.environ.get("SPACE_ID", "b762b7f8-ba30-4d3a-89a6-b7c14f840166")
REGION_URL = os.environ.get("REGION_URL", "https://eu-de.ml.cloud.ibm.com")

IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"
SCORING_URL = f"{REGION_URL}/ml/v4/deployments/{DEPLOYMENT_ID}/predictions?version=2021-05-01"

FIELDS = [
    "finyear", "lgdstatecode", "statename", "lgddistrictcode", "districtname",
    "totalbeneficiaries", "totalmale", "totalfemale", "totaltransgender",
    "totalsc", "totalst", "totalgen", "totalobc", "totalaadhaar", "totalmpbilenumber"
]


def get_iam_token(api_key: str) -> str:
    resp = requests.post(
        IAM_TOKEN_URL,
        data={
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": api_key,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def predict_scheme(values):
    if not IBM_API_KEY:
        return "⚠️ Server misconfigured: IBM_API_KEY secret not set.", "error"

    try:
        token = get_iam_token(IBM_API_KEY)
    except Exception as e:
        return f"Error getting IAM token: {e}", "error"

    payload = {
        "input_data": [
            {
                "fields": FIELDS,
                "values": [values],
            }
        ]
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(SCORING_URL, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        prediction = result["predictions"][0]["values"][0]
        scheme_code = prediction[0]
        
        confidence = None
        if len(prediction) > 1 and isinstance(prediction[1], list):
            confidence = max(prediction[1]) * 100
            
        if confidence is not None:
            return f"Predicted Scheme: **{scheme_code}** (confidence: {confidence:.1f}%)", "success"
        return f"Predicted Scheme: **{scheme_code}**", "success"
    except Exception as e:
        return f"Error calling scoring endpoint: {e}\n\nResponse: {getattr(resp, 'text', '')}", "error"


# ── Streamlit UI Setup ──────────────────────────────────────────────────────
st.set_page_config(page_title="NSAP Scheme Eligibility Predictor", page_icon="🏛️", layout="wide")

st.title("🏛️ NSAP Scheme Eligibility Predictor")
st.markdown(
    """
    Predicts the most appropriate National Social Assistance Programme (NSAP) scheme for an applicant, 
    based on a Snap Decision Tree Classifier trained via IBM Watson AutoAI (98.9% cross-validation accuracy).
    """
)
st.divider()

# Create forms/columns for cleaner input organization
st.subheader("📋 Enter Applicant & Regional Aggregate Data")

col1, col2, col3 = st.columns(3)
with col1:
    finyear = st.text_input("Financial Year", placeholder="e.g. 2023-24")
    lgddistrictcode = st.number_input("LGD District Code", step=1, value=0)
    totalmale = st.number_input("Total Male", step=1, value=0)
    totalst = st.number_input("Total ST", step=1, value=0)
    totalaadhaar = st.number_input("Total Aadhaar-linked", step=1, value=0)

with col2:
    lgdstatecode = st.number_input("LGD State Code", step=1, value=0)
    districtname = st.text_input("District Name")
    totalfemale = st.number_input("Total Female", step=1, value=0)
    totalgen = st.number_input("Total General", step=1, value=0)
    totalmobilenumber = st.number_input("Total Mobile-linked", step=1, value=0)

with col3:
    statename = st.text_input("State Name")
    totalbeneficiaries = st.number_input("Total Beneficiaries", step=1, value=0)
    totaltransgender = st.number_input("Total Transgender", step=1, value=0)
    totalsc = st.number_input("Total SC", step=1, value=0)
    totalobc = st.number_input("Total OBC", step=1, value=0)

st.write("") # Spacer

if st.button("Predict Scheme", type="primary", use_container_width=True):
    if not finyear.strip() or not statename.strip() or not districtname.strip():
        st.warning("⚠️ Please fill in all text fields (Financial Year, State Name, District Name).")
    else:
        # Array creation in the strict order required by fields
        input_values = [
            finyear.strip(), int(lgdstatecode), statename.strip(), int(lgddistrictcode), districtname.strip(),
            int(totalbeneficiaries), int(totalmale), int(totalfemale), int(totaltransgender),
            int(totalsc), int(totalst), int(totalgen), int(totalobc),
            int(totalaadhaar), int(totalmobilenumber)
        ]
        
        with st.spinner("Calling IBM Watson AutoAI Engine..."):
            message, msg_type = predict_scheme(input_values)
            
        if msg_type == "success":
            st.success(message)
        else:
            st.error(message)
