import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

# --- Page Configuration ---
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏡",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    /* Minimal White Theme Enhancements */
    .stApp {
        background-color: #FAFAFA;
        color: #333333;
        font-family: 'Inter', sans-serif;
    }
    .main .block-container {
        padding-top: 2rem;
        max-width: 800px;
    }
    h1, h2, h3 {
        color: #111827;
        font-weight: 600;
    }
    .stButton>button {
        width: 100%;
        background-color: #000000;
        color: #ffffff;
        border: none;
        padding: 0.75rem 0;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #333333;
        color: #ffffff;
    }
    .success-card {
        background-color: #ffffff;
        border-left: 5px solid #10B981;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-top: 2rem;
        text-align: center;
    }
    .price-text {
        font-size: 2.5rem;
        font-weight: 700;
        color: #10B981;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("### Project Information")
    st.markdown("**Model Used**")
    st.info("Random Forest Regressor")
    
    st.markdown("**Accuracy**")
    st.success("R² = 90.9%")
    
    st.markdown("**Dataset**")
    st.info("Ames Housing Dataset")
    
    st.markdown("---")
    st.markdown("### Built with")
    st.markdown("- Python\n- Scikit-Learn\n- Streamlit")

# --- Main Header ---
st.title("🏡 House Price Prediction")
st.markdown("Predict house prices using a Machine Learning model trained on the Ames Housing Dataset.")
st.markdown("---")

# --- Feature Dictionary ---
input_data = {}

# --- Formatting Dictionaries ---
quality_map = {"Ex": "Excellent", "Gd": "Good", "TA": "Typical/Average", "Fa": "Fair", "Po": "Poor", "None": "None"}
zoning_map = {"RL": "Residential Low Density", "RM": "Residential Medium Density", "FV": "Floating Village Residential", "RH": "Residential High Density", "C (all)": "Commercial"}
finish_map = {"Fin": "Finished", "RFn": "Rough Finished", "Unf": "Unfinished", "None": "None"}
contour_map = {"Lvl": "Near Flat/Level", "Bnk": "Banked", "HLS": "Hillside", "Low": "Depression"}
yes_no_map = {"Y": "Yes", "N": "No", "P": "Partial"}
exposure_map = {"Gd": "Good Exposure", "Av": "Average Exposure", "Mn": "Minimum Exposure", "No": "No Exposure", "None": "None"}
sale_cond_map = {"Normal": "Normal Sale", "Abnorml": "Abnormal Sale", "AdjLand": "Adjoining Land Purchase", "Alloca": "Allocation", "Family": "Sale between family", "Partial": "Home not completed when assessed"}
sale_type_map = {"WD ": "Warranty Deed", "New": "New constructed and sold", "COD": "Court Officer Deed/Estate", "ConLD": "Contract Low Down", "ConLI": "Contract Low Interest", "ConLw": "Contract Low Down payment and low interest", "CWD": "Warranty Deed - Cash", "Oth": "Other", "Con": "Contract 15% Down"}
foundation_map = {"PConc": "Poured Concrete", "CBlock": "Cinder Block", "BrkTil": "Brick & Tile", "Slab": "Slab", "Stone": "Stone", "Wood": "Wood"}
garage_type_map = {"Attchd": "Attached", "Detchd": "Detached", "BuiltIn": "Built-In", "Basment": "Basement", "CarPort": "Car Port", "2Types": "More than one type", "None": "None"}

# --- Section 1: Quick Estimate (Required) ---
st.header("Section 1: Quick Estimate (Recommended)")
st.caption("Please fill in these required features for an accurate baseline estimate.")

col1, col2 = st.columns(2)

with col1:
    input_data["Overall Qual"] = st.slider("Overall Quality (1-10)", 1, 10, 5)
    input_data["Garage Cars"] = st.number_input("Garage Cars", min_value=0, max_value=5, value=2)
    input_data["Kitchen Qual"] = st.selectbox("Kitchen Quality", ["Ex", "Gd", "TA", "Fa", "Po"], format_func=lambda x: quality_map.get(x, x))
    input_data["Bsmt Qual"] = st.selectbox("Basement Quality", ["Ex", "Gd", "TA", "Fa", "Po", "None"], format_func=lambda x: quality_map.get(x, x))
    input_data["MS Zoning"] = st.selectbox("MS Zoning", ["RL", "RM", "FV", "RH", "C (all)"], format_func=lambda x: zoning_map.get(x, x))
    input_data["Neighborhood"] = st.selectbox("Neighborhood", [
        "NAmes", "CollgCr", "OldTown", "Edwards", "Somerst", 
        "NridgHt", "Gilbert", "Sawyer", "NWAmes", "SawyerW", 
        "Mitchel", "BrkSide", "Crawfor", "IDOTRR", "Timber", 
        "NoRidge", "StoneBr", "SWISU", "ClearCr", "MeadowV", 
        "BrDale", "Blmngtn", "Veenker", "NPkVill", "Blueste"
    ])

with col2:
    input_data["Exter Qual"] = st.selectbox("Exterior Quality", ["Ex", "Gd", "TA", "Fa"], format_func=lambda x: quality_map.get(x, x))
    input_data["Full Bath"] = st.number_input("Full Bathrooms", min_value=0, max_value=4, value=2)
    input_data["Garage Finish"] = st.selectbox("Garage Finish", ["Fin", "RFn", "Unf", "None"], format_func=lambda x: finish_map.get(x, x))
    input_data["Gr Liv Area"] = st.number_input("Ground Living Area (sqft)", min_value=300, max_value=10000, value=1500)
    input_data["1st Flr SF"] = st.number_input("1st Floor Square Feet", min_value=300, max_value=5000, value=1000)
    input_data["Total Bsmt SF"] = st.number_input("Total Basement Square Feet", min_value=0, max_value=6000, value=1000)

# --- Section 2: Advanced Options (Optional) ---
st.header("Section 2: Advanced Options")
with st.expander("Expand to fine-tune optional features"):
    st.caption("Leave empty or untouched to use default values.")
    
    col3, col4 = st.columns(2)
    with col3:
        input_data["Land Contour"] = st.selectbox("Land Contour", ["Lvl", "Bnk", "HLS", "Low"], format_func=lambda x: contour_map.get(x, x))
        input_data["Fireplaces"] = st.number_input("Fireplaces", min_value=0, max_value=4, value=0)
        input_data["Central Air"] = st.selectbox("Central Air", ["Y", "N"], format_func=lambda x: yes_no_map.get(x, x))
        input_data["Bsmt Exposure"] = st.selectbox("Basement Exposure", ["Gd", "Av", "Mn", "No", "None"], format_func=lambda x: exposure_map.get(x, x))
        input_data["Sale Condition"] = st.selectbox("Sale Condition", ["Normal", "Abnorml", "AdjLand", "Alloca", "Family", "Partial"], format_func=lambda x: sale_cond_map.get(x, x))
        input_data["Exterior 1st"] = st.selectbox("Exterior 1st", ["VinylSd", "MetalSd", "Wd Sdng", "HdBoard", "BrkFace", "WdShing", "CemntBd", "Plywood", "AsbShng", "Stucco", "BrkComm", "AsphShn", "Stone", "ImStucc", "CBlock"])
        input_data["Exterior 2nd"] = st.selectbox("Exterior 2nd", ["VinylSd", "MetalSd", "Wd Sdng", "HdBoard", "Plywood", "Wd Shng", "CmentBd", "BrkFace", "Stucco", "AsbShng", "Brk Cmn", "ImStucc", "AsphShn", "Stone", "Other", "CBlock"])
        input_data["2nd Flr SF"] = st.number_input("2nd Floor Square Feet", min_value=0, max_value=3000, value=0)
        input_data["Paved Drive"] = st.selectbox("Paved Drive", ["Y", "P", "N"], format_func=lambda x: yes_no_map.get(x, x))
        
    with col4:
        input_data["Kitchen AbvGr"] = st.number_input("Kitchens Above Grade", min_value=0, max_value=3, value=1)
        input_data["Sale Type"] = st.selectbox("Sale Type", ["WD ", "New", "COD", "ConLD", "ConLI", "ConLw", "CWD", "Oth", "Con"], format_func=lambda x: sale_type_map.get(x, x))
        input_data["Foundation"] = st.selectbox("Foundation", ["PConc", "CBlock", "BrkTil", "Slab", "Stone", "Wood"], format_func=lambda x: foundation_map.get(x, x))
        input_data["BsmtFin SF 1"] = st.number_input("Basement Finished Square Feet 1", min_value=0, max_value=5000, value=0)
        input_data["Roof Matl"] = st.selectbox("Roof Material", ["CompShg", "Tar&Grv", "WdShngl", "WdShake", "Metal", "Membran", "Roll", "ClyTile"])
        input_data["Roof Style"] = st.selectbox("Roof Style", ["Gable", "Hip", "Flat", "Gambrel", "Mansard", "Shed"])
        input_data["Year Built"] = st.number_input("Year Built", min_value=1800, max_value=2026, value=2000)
        input_data["Bsmt Full Bath"] = st.number_input("Basement Full Bathrooms", min_value=0, max_value=3, value=0)
        input_data["Garage Type"] = st.selectbox("Garage Type", ["Attchd", "Detchd", "BuiltIn", "Basment", "CarPort", "2Types", "None"], format_func=lambda x: garage_type_map.get(x, x))

# --- Prediction Button & Logic ---
st.markdown("<br>", unsafe_allow_html=True)
if st.button("Predict Price"):
    with st.spinner("Calculating estimate..."):
        try:
            # Create CustomData object and format as DataFrame
            custom_data = CustomData(feature_dict=input_data)
            input_df = custom_data.get_data_as_dataframe()
            
            # Predict using the loaded pipeline
            pipeline = PredictPipeline()
            prediction = pipeline.predict(input_df)
            
            # Display Success Output
            formatted_price = f"${prediction:,.0f}"
            
            st.markdown(f"""
            <div class="success-card">
                <p style="color: #6B7280; font-size: 1.1rem; margin:0;">Estimated House Price</p>
                <p class="price-text">{formatted_price}</p>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.info("Check if the ML pipeline expects 80 features. If so, you may need to update CustomData in predict_pipeline.py to provide defaults for the unselected columns.")
