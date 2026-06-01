import os
import json
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
from google import genai
from google.genai import types

# ==========================================
# 1. PAGE INITIALIZATION & CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AI Supply Chain Optimization Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application Header & Branding
st.title("⚡ AI-Driven Real-Time Supply Chain Optimization Engine")
st.markdown("Generative Risk Assessment & Statistical Inventory Overrides")
st.markdown(
    "This system translates unstructured global disruption reports into operational metrics. ")
st.divider()

# ==========================================
# 2. INTELLIGENT LOCAL FAIL-SAFE ENGINE
# ==========================================
def intelligent_local_fallback(text_input):
    """
    Acts as an edge-computed deterministic backup engine when Google's cloud servers are overloaded.
    Uses contextual keyword heuristics to extract operational weights and prevent 503 app crashes.
    """
    text = text_input.lower()

    # 1. Infer Transit Delay Days dynamically from text
    detected_delay = 0
    words = text.split()
    for i, word in enumerate(words):
        if word in ['day', 'days', '-day'] and i > 0:
            clean_num = words[i-1].replace('-', '').strip()
            if clean_num.isdigit():
                detected_delay = int(clean_num)
                break

    if detected_delay == 0 and any(k in text for k in ['strike', 'block', 'flood', 'storm', 'delay', 'cyclone']):
        detected_delay = 3

    # 2. Infer Department Demand Shocks based on contextual impact categories
    food_shock = 1.00
    others_shock = 1.00

    if any(k in text for k in ['strike', 'block', 'protest', 'highway']):
        food_shock = 1.15
    elif any(k in text for k in ['cyclone', 'flood', 'storm', 'monsoon', 'weather']):
        food_shock = 1.60
        others_shock = 0.40
    elif any(k in text for k in ['holiday', 'festival', 'celebration', 'christmas']):
        food_shock = 1.30
        others_shock = 1.10

    # 3. Construct structured fallback payload matching Gemini's schema
    return {
        "disruption_summary": f"Local Fail-Safe Active (Cloud Congested): Inferred operational shift with an estimated +{detected_delay} day transit bottleneck.",
        "department_adjustments": {
            "Food and beverages": { "Demand_Shock_Factor": food_shock, "Lead_Time_Delay_Days": detected_delay },
            "Health and beauty": { "Demand_Shock_Factor": others_shock, "Lead_Time_Delay_Days": detected_delay },
            "Fashion accessories": { "Demand_Shock_Factor": max(0.1, others_shock * 0.5), "Lead_Time_Delay_Days": detected_delay },
            "Electronic accessories": { "Demand_Shock_Factor": others_shock, "Lead_Time_Delay_Days": detected_delay },
            "Home and lifestyle": { "Demand_Shock_Factor": others_shock, "Lead_Time_Delay_Days": detected_delay },
            "Sports and travel": { "Demand_Shock_Factor": others_shock, "Lead_Time_Delay_Days": detected_delay }
        }
    }

# ==========================================
# 3. SECURE CREDENTIALS & BASELINE LOADING
# ==========================================
if "GEMINI_API_KEY" in st.secrets:
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    try:
        client = genai.Client()
    except Exception as e:
        st.error(f"Failed to initialize the Gemini Client instance: {e}")
        st.stop()
else:
    st.error(
        "**Security Token Missing!** Please access your Streamlit Cloud Advanced Settings panel "
        "and add your key under the variable: `GEMINI_API_KEY = \"your_api_key_here\"`"
    )
    st.stop()

# Local baseline data connection
BASELINE_FILE = "supermarket_phase1_baseline.csv"

if not os.path.exists(BASELINE_FILE):
    st.warning(f"Central repository file '{BASELINE_FILE}' not found locally.")
    uploaded_fallback = st.file_uploader("Please upload your Phase 1 Baseline CSV manually:", type=["csv"])
    if uploaded_fallback is not None:
        df_baseline = pd.read_csv(uploaded_fallback)
    else:
        st.info("Waiting for data baseline file ingestion...")
        st.stop()
else:
    df_baseline = pd.read_csv(BASELINE_FILE)

# Sidebar view for structural reference
st.sidebar.header("Baseline Operations Profile")
st.sidebar.write("Normal Baseline Ingested Successfully.")
with st.sidebar.expander("View Normal Operational Profiles", expanded=False):
    st.dataframe(
        df_baseline[['Department', 'Avg Daily Sales', 'Reorder Point']],
        hide_index=True
    )

# ==========================================
# 4. INTERACTIVE RISK INGESTION DESK
# ==========================================
st.subheader("Live Event Feed & Operational Stream")
user_news_input = st.text_area(
    "Paste or type any real-world disruption alert, logistics bulletin, or geopolitical update below:",
    height=120,
    placeholder="Example: Unprecedented blizzard conditions sweep across the northern transport routes, creating 5-day delivery backups and triggering significant grocery stocking buyouts..."
)

# Action Trigger Button
if st.button("Execute Optimization Models", type="primary"):
    if not user_news_input.strip():
        st.warning("Action blocked: Please submit a structural incident profile statement first.")
    else:
        status_container = st.empty()
        
        with status_container.container():
            st.info("**Step 1:** Transferring incident data to the live cloud natural language engine...")
            
        risk_data = None
        
        try:
            # Prepare contextual operational snapshot to feed to the AI model
            baseline_context = df_baseline[['Department', 'Avg Daily Sales', 'Safety Stock', 'Reorder Point']].to_string(index=False)
            
            prompt = f"""
            You are a senior AI Supply Chain Risk Officer operating a regional supermarket network.
            Your task is to analyze an incoming real-world disruption alert and translate it into clear quantitative modifiers for our operational departments.

            ### CURRENT OPERATIONAL BASELINE STATE:
            {baseline_context}

            ### LIVE USER-SUBMITTED NEWS ALERT:
            "{user_news_input}"

            ### INSTRUCTIONS:
            Evaluate how this disruption affects each department across two specific variables:
            1. Demand_Shock_Factor (Float): A multiplier for daily sales. Use 1.00 for no change. Scale ABOVE 1.00 if panic-buying or seasonal surges spike demand (e.g., 1.40 for +40%). Scale BELOW 1.00 if store closures or conditions halt shopping (e.g., 0.30 for a 70% drop).
            2. Lead_Time_Delay_Days (Integer): The number of extra days delivery trucks will be delayed due to transit blocks or logistics shutdowns. Use 0 if there is no delay.

            ### OUTPUT FORMAT:
            You must respond ONLY with a valid JSON object matching this exact schema. Do not include markdown code formatting, backticks, or conversational text.

            {{
              "disruption_summary": "Brief 1-sentence assessment of the incident",
              "department_adjustments": {{
                "Food and beverages": {{ "Demand_Shock_Factor": 1.00, "Lead_Time_Delay_Days": 0 }},
                "Health and beauty": {{ "Demand_Shock_Factor": 1.00, "Lead_Time_Delay_Days": 0 }},
                "Fashion accessories": {{ "Demand_Shock_Factor": 1.00, "Lead_Time_Delay_Days": 0 }},
                "Electronic accessories": {{ "Demand_Shock_Factor": 1.00, "Lead_Time_Delay_Days": 0 }},
                "Home and lifestyle": {{ "Demand_Shock_Factor": 1.00, "Lead_Time_Delay_Days": 0 }},
                "Sports and travel": {{ "Demand_Shock_Factor": 1.00, "Lead_Time_Delay_Days": 0 }}
              }}
            }}
            """
            
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                ),
            )
            
            risk_data = json.loads(response.text)
            status_container.empty()
            st.success(f"**AI Assessment Summary:** {risk_data['disruption_summary']}")
            
        except Exception as err:
            # INTERCEPT CLOUD OVERLOAD / 503 SEAMLESSLY
            status_container.empty()
            st.warning("Cloud server infrastructure congested (503 High Demand). Activating Local Fail-Safe Model...")
            risk_data = intelligent_local_fallback(user_news_input)
            st.info(f"**Heuristic Assessment Summary:** {risk_data['disruption_summary']}")
            
        # ==========================================
        # 5. STATISTICAL CALCULATIONS BACKEND
        # ==========================================
        try:
            adjustments = risk_data['department_adjustments']
            st.markdown("###Statistical Distribution Overrides Matrix")
            
            NORMAL_LEAD_TIME = 2
            updated_rows = []
            
            for index, row in df_baseline.iterrows():
                dept = row['Department']
                mu = row['Avg Daily Sales']
                sigma = row['Sales Volatility']
                
                service_level = float(str(row['Target Service Level']).replace('%', '')) / 100.0
                z_score = norm.ppf(service_level)
                
                shock = adjustments.get(dept, {}).get('Demand_Shock_Factor', 1.0)
                delay = adjustments.get(dept, {}).get('Lead_Time_Delay_Days', 0)
                
                adjusted_lead_time = NORMAL_LEAD_TIME + delay
                adjusted_safety_stock = z_score * (sigma * shock) * np.sqrt(adjusted_lead_time)
                adjusted_reorder_point = ((mu * shock) * adjusted_lead_time) + adjusted_safety_stock
                
                updated_rows.append({
                    'Department': dept,
                    'Base ROP': row['Reorder Point'],
                    'Demand Shock (AI)': shock,
                    'Transit Delay (AI)': f"+{delay} Days",
                    'New Safety Stock': round(adjusted_safety_stock, 2),
                    'New Reorder Point (ROP)': round(adjusted_reorder_point, 2)
                })
                
            df_results = pd.DataFrame(updated_rows)
            max_rop_row = df_results.loc[df_results['New Reorder Point (ROP)'].idxmax()]
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Highest Risk Category", max_rop_row['Department'])
            m2.metric("Peak Priority ROP Level", f"{max_rop_row['New Reorder Point (ROP)']} Units")
            m3.metric("System Operational Status", "Optimized Override Active")
            
            st.write("")
            
            st.dataframe(
                df_results.style.highlight_max(axis=0, subset=['New Reorder Point (ROP)'], color='#ffe3e3'),
                use_container_width=True,
                hide_index=True
            )
            
            csv_export = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Updated Operational Blueprint (.CSV)",
                data=csv_export,
                file_name="supermarket_phase2_dynamic_overrides.csv",
                mime="text/csv",
                type="secondary"
            )
            
        except Exception as calculation_err:
            st.error(f"Core Math Optimization Failure: {calculation_err}")
