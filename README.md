# Innovative ML Project

**AI‑Driven E‑commerce Pricing Optimization**

An end‑to‑end system that ingests real retail transactions, cleans and analyzes the data, trains a state‑of‑the‑art TabNet model to predict transaction totals, and serves both a programmatic API (FastAPI) and an interactive dashboard (Streamlit). Built with modularity and production‑readiness in mind.

---

## Table of Contents

1. [Overview](#overview)  
2. [Features](#features)  
3. [Dataset](#dataset)  
4. [Getting Started](#getting-started)  
   - [Prerequisites](#prerequisites)  
   - [Installation](#installation)  
   - [Quick‑Start (Docker)](#quick-start-docker)  
   - [Quick‑Start (Local Python)](#quick-start-local-python)  
   - [Project Structure](#project-structure)  
5. [Data Preprocessing](#data-preprocessing)  
6. [Model Training](#model-training)  
7. [Evaluation](#evaluation)  
8. [API Service (FastAPI)](#api-service-fastapi)  
9. [Dashboard (Streamlit)](#dashboard-streamlit)  
10. [Containerization (Docker)](#containerization-docker)  
11. [Distribution & Submission](#distribution--submission)  
12. [Future Work](#future-work)  
13. [Contributing](#contributing)  
14. [License](#license)  
15. [Contact](#contact)  
16. [Links](#links)  

---

## Overview

Manual pricing strategies can be slow and sub‑optimal. This project demonstrates how to:

1. **Understand** customer purchase patterns via Exploratory Data Analysis (EDA).  
2. **Predict** the total price of a transaction using a pretrained TabNet model.  
3. **Serve** predictions through a FastAPI microservice.  
4. **Interact** with the model via a polished Streamlit dashboard.  
5. **Scale** and **extend** with Docker, CI/CD, monitoring, and AI‑agent enhancements.

---

## Features

- **Data Cleaning & EDA** with Jupyter Notebooks  
- **Feature Engineering**: time features, RFM summaries  
- **Self‑Supervised Pretraining** & **Fine‑Tuning** with PyTorch‑TabNet  
- **Evaluation**: hold‑out RMSE, error analysis  
- **API Service**: FastAPI endpoint (`/predict`) + Swagger UI  
- **Interactive UI**: Streamlit dashboard for single & batch predictions  
- **Modular Codebase**: clear separation of data, model, API, UI  
- **Ready for Production**: containerization, CI/CD, monitoring  

---

## Dataset

We use the **Online Retail II** dataset (UCI/Kaggle), containing ~500 K real transactions (2009–2011) from a UK online store.

Key fields:

| Column       | Type      | Description                              |
|--------------|-----------|------------------------------------------|
| Invoice      | string    | Invoice number (prefix “C” for returns)  |
| StockCode    | string    | Unique product code                      |
| Description  | string    | Product description                      |
| Quantity     | int       | Number of items                          |
| InvoiceDate  | datetime  | Invoice timestamp                        |
| UnitPrice    | float     | Price per item                           |
| CustomerID   | int       | Unique customer identifier               |
| Country      | string    | Country of purchase                      |

---

## Getting Started

### Prerequisites

- **Python 3.10+**  
- **Git**  
- **Docker & Docker‑Compose** (for containerized setup)  
- **VS Code** (optional, recommended)  
- **Windows / macOS / Linux**

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/Hamza-Waseem-Nasser/innovative-ml-project.git
cd innovative-ml-project

# 2. Create & activate virtual environment
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Quick‑Start (Docker)

```bash
# 1. Unzip package (if provided as ZIP)
unzip innovative-ml-project.zip && cd innovative-ml-project

# 2. Build and run both services
docker-compose up --build

# 3. Visit in your browser:
#    • API Swagger UI: http://localhost:8000/docs
#    • Dashboard:      http://localhost:8501
```

### Quick‑Start (Local Python)

```bash
# 1. Clone the repo
git clone https://github.com/Hamza-Waseem-Nasser/innovative-ml-project.git
cd innovative-ml-project

# 2. Create & activate venv
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 4. Start the API
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# 5. In a new terminal, start the Dashboard
streamlit run app/streamlit_app.py

# 6. Visit:
#    • API Swagger UI: http://localhost:8000/docs
#    • Dashboard:      http://localhost:8501
```

### Project Structure

```
innovative-ml-project/
├── .streamlit/              # Streamlit theming/config
├── api/                     # FastAPI service
│   ├── main.py
│   ├── schemas.py
│   └── utils.py
├── app/                     # Streamlit dashboard
│   └── streamlit_app.py
├── data/                    # Cleaned datasets
│   ├── transactions_clean.csv
│   └── rfm_summary.csv
├── docs/                    # Additional documentation/images
│   └── d/
├── models/                  # Saved ML artifacts
│   ├── tabnet_pretrainer.zip
│   ├── tabnet_regressor.zip
│   ├── enc_StockCode.pkl
│   ├── enc_Country.pkl
│   └── scaler_num.pkl
├── notebooks/               # EDA notebook + plots
│   ├── eda_online_retail.ipynb
│   └── plots/
├── src/                     # Model training & evaluation notebooks
│   ├── train_tabnet.ipynb
│   └── evaluate_tabnet.ipynb
├── requirements.txt         # Python dependencies
├── Dockerfile.api           # Dockerfile for FastAPI service
├── Dockerfile.ui            # Dockerfile for Streamlit UI
├── docker-compose.yml       # Compose file for both services
└── README.md                # This file
```

---

## Data Preprocessing

1. **Load & Merge** both sheets from `online_retail_II.xlsx`.  
2. **Clean**: drop missing `CustomerID`, remove duplicates & returns, convert types.  
3. **Feature Engineering**:  
   - `TotalPrice = Quantity × UnitPrice`  
   - Time features: `hour`, `day_of_week`, `month`  
   - RFM summary (`Recency`, `Frequency`, `Monetary`) per customer  
4. **Save**:  
   - `data/transactions_clean.csv`  
   - `data/rfm_summary.csv`  

*(See `notebooks/eda_online_retail.ipynb` for full EDA)*

---

## Model Training

We use **PyTorch‑TabNet** with self‑supervised pretraining:

1. **Pretrainer** (`TabNetPretrainer`): mask 80% features, learn embeddings.  
2. **Regressor** (`TabNetRegressor`): fine‑tune on `TotalPrice`.  
3. **Artifacts** saved under `models/`:  
   - `tabnet_pretrainer.zip`  
   - `tabnet_regressor.zip`  
   - Encoders & scaler (`.pkl` files)  

*(See `src/train_tabnet.ipynb` for code)*

---

## Evaluation

- **Hold‑out split**: 80/20, `random_state=42`  
- **Metric**: Root Mean Squared Error (RMSE) on test set  
- **Baseline comparisons**: RandomForest, LinearRegression  
- **Results**: TabNet outperforms baselines  
- **Notebook**: `src/evaluate_tabnet.ipynb`

---

## API Service (FastAPI)

The **`api/`** folder contains:

- **`schemas.py`**: Pydantic models for request/response  
- **`utils.py`**: loads encoders, scaler, model; preprocessing helper  
- **`main.py`**: defines `/predict` endpoint and Swagger UI

**Run the API**:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Open **http://127.0.0.1:8000/docs** to explore.

---

## Dashboard (Streamlit)

The **`app/streamlit_app.py`** provides:

- **Home**: KPIs (Total Sales, Avg. Order Value), RMSE, data overview, distributions  
- **Single Prediction**: form for real‑time inference  
- **Batch Prediction**: CSV upload + batch inference + error analysis  

**Run the dashboard**:

```bash
streamlit run app/streamlit_app.py
```

Open **http://localhost:8501**.

---

## Containerization (Docker)

You can containerize both services for easy deployment.

### Dockerfile for API

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY api/ models/ data/ ./
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Dockerfile for Dashboard

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ models/ data/ ./
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
```

### docker-compose.yml

```yaml
version: '3.8'
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    ports:
      - "8000:8000"
  ui:
    build:
      context: .
      dockerfile: Dockerfile.ui
    ports:
      - "8501:8501"
```

**Run both**:

```bash
docker-compose up --build
```

---

## Distribution & Submission

To package your project for submission:

1. **Clean** out your virtual environment (`.venv/`) and any `__pycache__` folders.  
2. **Verify** your `.gitignore` excludes those.  
3. **Zip** the project directory (excluding venv) from its parent folder:

   ```bash
   zip -r innovative-ml-project.zip innovative-ml-project \
     -x "innovative-ml-project/.venv/*" \
     -x "innovative-ml-project/__pycache__/*"
   ```

4. **Submit** `innovative-ml-project.zip`.  
5. **Reviewers** can unzip, then choose Docker or Local Python quick‑start steps above to run the full end‑to‑end project.

---

## Future Work

- **Security & Monitoring**: CORS, API keys, Prometheus/Grafana  
- **CI/CD**: Tests (`pytest`), GitHub Actions, auto‑deploy  
- **Customer Segmentation**: K‑Means on RFM with dashboard visuals  
- **Time‑Series Forecasting**: Prophet/ARIMA for sales projection  
- **RAG Chat Interface**: “Ask your data” with FAISS + LLM  
- **Dynamic Pricing Agent**: RL agent to recommend optimal discounts  
- **Automated Retraining**: Scheduled jobs, data validation (Great Expectations)  

---

## Contributing

1. Fork the repo  
2. Create a feature branch (`git checkout -b feature/my-feature`)  
3. Commit your changes (`git commit -m "Add my feature"`)  
4. Push to branch (`git push origin feature/my-feature`)  
5. Open a Pull Request  

Please follow the existing code style and include tests where applicable.

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## Contact

**Hamza Waseem Nasser**  
- GitHub: [@Hamza-Waseem-Nasser](https://github.com/Hamza-Waseem-Nasser)  
- Email: *your.email@example.com*

---

## Links

- **GitHub Repo**: https://github.com/Hamza-Waseem-Nasser/innovative-ml-project  
- **Live Demo**: *https://your-deployed-app.com* (if available)