# Customer Behaviour Analysis Using Unsupervised Learning

## Project Overview

This project applies unsupervised machine learning to segment mall customers based on their annual income and spending score. It also explores Gaussian probability-based anomaly detection to identify customers whose feature combinations are relatively uncommon.

The aim is to discover customer groups that could help a business understand different spending behaviours and support customer-targeting strategies.

## Objectives

- Explore and prepare customer data for machine learning.
- Scale numerical features using StandardScaler.
- Evaluate cluster counts using the Elbow Method and Silhouette Score.
- Segment customers using K-Means clustering.
- Summarise customer segments using average age, income, spending score, and customer count.
- Identify potential anomalies using Gaussian probability densities.
- Compare clustering results after excluding flagged customers.
- Save visualisations and analysis results.

## Dataset

The project uses the Mall Customers dataset.

**Features used for clustering and anomaly detection:**

- `Annual Income (k$)` — annual income, expressed in thousands of dollars.
- `Spending Score (1-100)` — a spending score on a scale from 1 to 100.

Age and customer ID are also used in the customer-segment summary.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Scikit-learn

## Machine Learning Methods

### 1. Feature Scaling

`StandardScaler` standardizes annual income and spending score so that the different numerical scales do not disproportionately influence K-Means.

### 2. K-Means Clustering

K-Means groups customers according to similarities in the selected features.

### 3. Cluster Evaluation

Two methods are used to help choose the number of clusters:

- **Elbow Method:** examines how inertia decreases as the number of clusters increases.
- **Silhouette Score:** measures how well-separated and cohesive the resulting clusters are.

The final model uses five clusters, based on the evaluation performed during this project.

### 4. Gaussian-Based Anomaly Detection

Gaussian probability densities are calculated for the two selected features. Their product is used as a simplified anomaly score. Customers below the fifth-percentile score threshold are flagged as potential anomalies.

This is an exploratory method, not confirmation that a customer is genuinely anomalous. It assumes feature independence when multiplying the densities.

### 5. Clustering After Excluding Flagged Customers

A second K-Means model is trained after excluding flagged customers. Its cluster sizes and centroids are compared with the original analysis.

## Visualizations

### Elbow Method
plots/elbow_method.png

### Silhouette Score Comparison
plots/silhouette_scores.png

### Customer Segments
plots/customer_segments.png

### Gaussian Anomaly Score Distribution
plots/gaussian_anomaly_scores.png

### Potential Anomalies
plots/customer_anomalies.png


## Project Structure

```text
customer-behaviour-analysis/
├── main.py
├── Mall_Customers.csv
├── README.md
├── requirements.txt
├── .gitignore
├── plots/
│   ├── elbow_method.png
│   ├── silhouette_scores.png
│   ├── customer_segments.png
│   ├── gaussian_anomaly_scores.png
│   └── customer_anomalies.png
└── outputs/
    ├── cluster_summary.csv
    ├── customer_analysis_results.csv
    └── customers_without_flagged_anomalies.csv
```

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com
cd customer_behaviour_analysis
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the script

```bash
python customer_behaviour_analysis.py
```

The script generates plots in the `plots/` folder and CSV results in the `outputs/` folder.

## Key Findings

Add your verified results here after running the script:

- Best K according to silhouette-score comparison: 5
- Final model silhouette score: 5
- Number of customers in each cluster: Cluster
                                        0    81
                                        1    39
                                        2    22
                                        3    35
                                        4    23
- Number of potential anomalies flagged: 10
- Main customer-segment observations: 190

## Limitations

- The analysis uses only annual income and spending score for clustering.
- K-Means requires a chosen number of clusters and can be affected by outliers.
- The Gaussian anomaly score uses a simplified independence assumption.
- The percentile threshold flags a selected proportion of customers; flagged records require further investigation.
- The findings describe patterns in this dataset and should not automatically be generalised to all mall customers.

## Future Improvements

- Compare the results with other clustering methods.
- Explore additional customer features.
- Evaluate cluster stability and alternative anomaly-detection methods.
- Develop a dashboard to explore customer segments interactively.

## Author

udhaya kumar (https://github.com/udhayakumar268/customer_behaviour_analysis)

MSc Artificial Intelligence | Machine Learning | Python
