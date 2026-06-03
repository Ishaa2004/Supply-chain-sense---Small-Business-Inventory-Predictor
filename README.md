# 📦 SupplyChainSense: AI-Driven Real-Time Supply Chain Optimization Engine

SupplyChainSense is an intelligent inventory optimization application that bridges classical operations research with real-time generative AI risk contextualization. Built with **Streamlit** and powered by **Google Gemini 2.5 Flash**, the system dynamically ingests raw point-of-sale transaction logs or business summary records, automatically extracts historical daily demand distributions, and processes unscripted real-world disruption updates (such as heavy storms, logistics bottlenecks, or labor strikes) to automatically rewrite inventory triggers on the fly.

---

## ✨ Core Features

- **Dual-Mode Data Ingestion:** Automatically detects input file formats, processing either raw transaction ledgers containing individual sales records or pre-aggregated department summaries.
- **Intelligent Field Mapping:** Built-in column normalization automatically handles minor schema alignment variations like `qty`, `units_sold`, or `Product Line`.
- **Dynamic Parameter Engineering:** Utilizes statistical continuous review $(r, Q)$ model principles to calculate steady-state daily averages ($\mu$) and sales volatility matrices ($\sigma$).
- **Generative Risk Controller:** Automatically extracts unstructured contextual incident texts into precise numerical modifiers (`Demand_Shock_Factor` and `Lead_Time_Delay_Days`) using structured JSON schemas.
- **Deterministic Edge Fallback:** Features an embedded heuristic backup rules engine to calculate safety buffers locally if cloud network latency or traffic spikes occur.

---

## 🛠️ Tech Stack & Prerequisites

The application is built using open-source Python data science frameworks:
- **Streamlit** (Interactive Interface & Layout)
- **Google GenAI SDK** (Gemini 2.5 Flash API Client)
- **Pandas** & **NumPy** (Time-Series Metrics & Matrix Manipulation)
- **SciPy** (Statistical Normal Distribution Functions)

---

## 📂 Project Directory Structure

Organize your GitHub repository using this streamlined layout:

```text
📁 supply-chain-sense/
│
├── 📄 app.py                         # Core interactive Streamlit application entry point
├── 📄 requirements.txt               # System dependencies configuration
└── 📄 SuperMarket Analysis.csv       # Default historical transaction ledger for instant demo mode

**Grounding the Parameters in Operations Theory** 

1) Target Service Levels (SL) and the Z-Score Matrix
The Target Service Level represents the mathematical probability that a business will successfully fulfill customer orders without experiencing a stockout during a single replenishment cycle.
To convert this percentage into an active warehouse cushion, the code maps it onto a standard normal distribution curve using the Inverse Cumulative Distribution Function (also known as the Percent Point Function or Z-score function):

**z=ϕ^(-1) (SL)**
In the Python backend, this is handled natively via:
**z_score = norm.ppf(target_service_level)**

2) **Annual Carrying Cost Rates _I_ and Daily Holding Costs _H_**
Storing items involves overhead (rent, climate control, insurance, and spoilage). The Annual Carrying Cost Rate _(I)_ represents this physical expense as a percentage of the item's unit price. 
The engine derives the Daily Holding Cost _(H)_ per item using the following calculation: 

**_H_ = (Unit Price * _I_)/365**

**The Dynamic Recalculation Math**
When an emergency alert is passed to the engine, the system overrides the steady-state baseline vectors and computes a new distribution matrix instantly:

**〖Lead Time 〗_adjusted= 〖Lead Time 〗_(normal )+〖Delay〗_AI**

**〖SS〗_(adjusted )=z ×(σ × 〖Shock〗_AI )×√(〖Lead Time〗_adjusted  )**

**〖ROP〗_adjusted=((μ× 〖Shock〗_AI )× 〖Lead Time 〗_adjusted )+ 〖SS 〗_adjusted**
