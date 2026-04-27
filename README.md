# Manual K-Means with PySpark

This project implements K-Means clustering from scratch using PySpark RDDs.

## Overview
- Custom K-Means implementation (no sklearn clustering)
- Distributed computation using PySpark
- Multiple runs for stability
- Evaluation using:
  - Calinski-Harabasz Score
  - Adjusted Rand Index (ARI)

## Project Structure
```
kmeans-spark-project/
│
├── src/
│ ├── kmeans.py
│ ├── evaluation.py
│ ├── utils.py
│
├── data/
│ ├── iris.csv
│ ├── glass.csv
│ ├── parkinsons.csv
│
├── notebooks/
│   └── kmeans_experiments.ipynb
│
├── README.md
└── requirements.txt
```

## Technologies
- Python
- PySpark
- NumPy / Pandas
- Scikit-learn

## How to Run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Run notebook:
   ```
   notebooks/kmeans_experiments.ipynb
   ```
