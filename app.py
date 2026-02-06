import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
import streamlit as st
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="KRA Revenue Radar",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stMetric { background-color: #000000; padding: 15px; border-radius: 10px; border: 1px solid #ddd; }
    </style>
    """, unsafe_allow_html=True)

st.title("KRA CIT Risk Engine")


# Artifact Loading
@st.cache_resource(show_spinner="Loading model weights...")
def load_bundle():
   
    try:
        path = "deployment_artifacts/kra_cit_risk_model_v1.pkl"
        data = joblib.load(path)
        logger.info("Model artifacts loaded successfully.")
        return data
    except FileNotFoundError:
        st.error(f"Critical Error: Model file not found at {path}")
        return None

artifacts = load_bundle()

if artifacts:
    model = artifacts["model"]
    feat_cols = artifacts["feature_names"] 
    opt_threshold = artifacts.get("threshold", 0.5) 

   
    with st.sidebar:
        st.header("Taxpayer Profile")
        
        turnover = st.number_input("Gross Turnover (KES)", min_value=0.0, value=1000000.0)
        cos = st.number_input("Cost of Sales (KES)", min_value=0.0, value=750000.0)
        admin_exp = st.number_input("Admin Expenses (KES)", min_value=0.0, value=120000.0)
        
        sector_choice = st.selectbox(
            "Economic Sector",
            ["Manufacturing", "Service", "Construction", "Other"]
        )
        
        st.divider()
        analyze_btn = st.button("Run Risk Assessment", type="primary")

    #  Feature Engineering 
   
    c2t = cos / turnover if turnover > 0 else 0.0
    a2t = admin_exp / turnover if turnover > 0 else 0.0

    if analyze_btn:
        try:
          
            input_df = pd.DataFrame(0, columns=feat_cols, index=[0])

            
            mappings = {
                "num__grossturnover": turnover,
                "num__cost_of_sales": cos,
                "num__total_administrative_exp": admin_exp,
                "num__cost_to_turnover": c2t,
                "num__admin_cost_ratio": a2t
            }

            for col, val in mappings.items():
                if col in input_df.columns:
                    input_df.at[0, col] = val

           
            target_sector = f"cat__sector_{sector_choice}"
            if target_sector in input_df.columns:
                input_df.at[0, target_sector] = 1

          
            dm = xgb.DMatrix(input_df, feature_names=feat_cols)
            risk_score = float(model.predict(dm)[0])

         
            st.subheader("Analysis Summary")
            m1, m2, m3 = st.columns(3)
            
            with m1:
                st.metric("Risk Score", f"{risk_score * 100:.1f}%")
            with m2:
                st.metric("Cost To Ratio", f"{c2t:.2f}")
            with m3:
                st.metric("Admin Ratio", f"{a2t:.2f}")

            # Risk Categorization
            if risk_score >= 0.75:
                st.error("### CRITICAL: High Probability of Artificial Loss")
                st.info("**Action:** Schedule Field Audit. Focus on 'Cost of Sales' substantiation.")
            elif risk_score >= opt_threshold:
                st.warning("### ELEVATED: Flagged for Desk Review")
                st.info("**Action:** Request additional Ledger details for Administrative Expenses.")
            else:
                st.success("### LOW: Normal Compliance Profile")
                st.write("Metric falls within expected historical variance for this sector.")

        except Exception as err:
            st.error("Prediction Engine Failure")
            logger.error(f"Inference Error: {str(err)}")
            st.exception(err) 

else:
    st.warning("System offline: Model artifacts failed to load.")
