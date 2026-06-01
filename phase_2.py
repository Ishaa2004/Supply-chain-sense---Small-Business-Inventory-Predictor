import os
import json
import pandas as pd
import numpy as np
from scipy.stats import norm

# Import the modern Google GenAI SDK
from google import genai
from google.genai import types
from google.colab import userdata

def intelligent_local_fallback(text_input):
    """
    Acts as a local deterministic backup engine when Google's servers are overloaded.
    Uses contextual keyword heuristics to extract operational weights.
    """
    text = text_input.lower()
    
    # 1. Infer Transit Delay Days dynamically from text
    detected_delay = 0
    words = text.split()
    for i, word in enumerate(words):
        # Look for numbers preceding words like 'day' or 'days'
        if word in ['day', 'days', '-day'] and i > 0:
            clean_num = words[i-1].replace('-', '').strip()
            if clean_num.isdigit():
                detected_delay = int(clean_num)
                break
    
    # If no specific number of days is found but a bottleneck is mentioned, default to 3 days
    if detected_delay == 0 and any(k in text for k in ['strike', 'block', 'flood', 'storm', 'delay']):
        detected_delay = 3

    # 2. Infer Department Demand Shocks based on contextual impact categories
    food_shock = 1.00
    others_shock = 1.00
    
    if any(k in text for k in ['strike', 'block', 'protest', 'highway']):
        # Logistics disruption causes slight panic hoarding on food essentials
        food_shock = 1.15
    elif any(k in text for k in ['cyclone', 'flood', 'storm', 'monsoon', 'weather']):
        # Heavy natural disasters cause major grocery surges, drops in leisure spending
        food_shock = 1.50
        others_shock = 0.50
    elif any(k in text for k in ['holiday', 'festival', 'celebration', 'christmas']):
        # Festive season increases food and travel categories
        food_shock = 1.30
        others_shock = 1.10

    # 3. Construct structured JSON schema identical to Gemini's expected output
    fallback_payload = {
        "disruption_summary": f"Local Fail-Safe Mode Active: Inferred a operational disruption with an estimated +{detected_delay} day transit bottleneck.",
        "department_adjustments": {
            "Food and beverages": { "Demand_Shock_Factor": food_shock, "Lead_Time_Delay_Days": detected_delay },
            "Health and beauty": { "Demand_Shock_Factor": others_shock, "Lead_Time_Delay_Days": detected_delay },
            "Fashion accessories": { "Demand_Shock_Factor": max(0.2, others_shock * 0.8), "Lead_Time_Delay_Days": detected_delay },
            "Electronic accessories": { "Demand_Shock_Factor": others_shock, "Lead_Time_Delay_Days": detected_delay },
            "Home and lifestyle": { "Demand_Shock_Factor": others_shock, "Lead_Time_Delay_Days": detected_delay },
            "Sports and travel": { "Demand_Shock_Factor": others_shock, "Lead_Time_Delay_Days": detected_delay }
        }
    }
    return fallback_payload


def run_phase2_live_pipeline(user_live_news):
    if not os.path.exists('supermarket_phase1_baseline.csv'):
        raise FileNotFoundError("Please upload 'supermarket_phase1_baseline.csv' to Colab's file side panel first!")
        
    df_baseline = pd.read_csv('supermarket_phase1_baseline.csv')
    baseline_context = df_baseline[['Department', 'Avg Daily Sales', 'Safety Stock', 'Reorder Point']].to_string(index=False)
    
    # Standard Setup
    try:
        os.environ["GEMINI_API_KEY"] = userdata.get('GEMINI_API_KEY')
        client = genai.Client()
        use_live_ai = True
    except Exception:
        print("Secret Key missing. Dropping directly down to Local Fail-Safe Mode.")
        use_live_ai = False

    risk_data = None

    # Try calling the Live Cloud Server
    if use_live_ai:
        try:
            print("\n[Live Gen AI Layer]: Connecting to Gemini servers...")
            prompt = f"""
            You are an expert AI Supply Chain Risk Officer. Translate this text into quantitative variables.
            CURRENT OPERATIONAL BASELINE STATE:
            {baseline_context}
            LIVE USER-SUBMITTED NEWS ALERT:
            "{user_live_news}"
            Respond ONLY with a valid JSON matching this schema:
            {{
              "disruption_summary": "1-sentence assessment",
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
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            risk_data = json.loads(response.text)
            print("Cloud processing successful.")
        except Exception as api_err:
            print(f"Cloud Server Traffic Congested (Error: {api_err}).")
            print("Seamlessly switching to Local Fail-Safe Backups...")
            risk_data = intelligent_local_fallback(user_live_news)
    else:
        risk_data = intelligent_local_fallback(user_live_news)

    # Core Mathematical Pipeline Execution
    adjustments = risk_data['department_adjustments']
    print(f"Operational Assessment: {risk_data['disruption_summary']}\n")
    print("[Math Engine]: Executing parameter updates over statistical distributions...\n")
    
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
        
        # Calculations
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
    df_results.to_csv('supermarket_phase2_dynamic_overrides.csv', index=False)
    return df_results

# Run Interactive View
user_news_input = input("Enter any unscripted news line or disruption update here:\n")
if user_news_input.strip() != "":
    try:
        final_table = run_phase2_live_pipeline(user_news_input)
        print("="*75)
        print("=== INVENTORY PARAMETER OVERRIDE GENERATED SUCCESSFULLY ===")
        print("="*75)
        display(final_table)
    except Exception as e:
        print(f"Execution Crash: {e}")
