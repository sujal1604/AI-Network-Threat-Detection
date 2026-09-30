# AI-Powered Network Threat Detection & Behavioral Anomaly Intelligence

## Web Dashboard

This project includes an interactive Streamlit webpage for demonstrating:
- Known threat classification
- Unsupervised anomaly detection
- Device behavioral analysis
- Device risk ranking
- Feature importance
- Confusion matrix
- Project report
- Optional CSV upload

## Run the webpage

Open the project folder in VS Code and run:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The browser will open at:

`http://localhost:8501`

If it does not open automatically, copy the Local URL printed in the terminal into your browser.

## Project structure

```text
AI_Network_Threat_Detection_Web/
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
├── data/
├── docs/
├── models/
├── outputs/
└── src/
```

The included data is synthetic for demonstration. For real deployment, use only authorized network telemetry and validate the models against real incidents.
