import nbformat as nbf
import os

def create_summary_notebook():
    nb = nbf.v4.new_notebook()

    cells = []

    # --- Title and Header ---
    cells.append(nbf.v4.new_markdown_cell("# Fraud Detection Project: End-to-End Summary Report\n\n**Date:** December 31, 2025\n\n---"))

    # --- Executive Summary ---
    cells.append(nbf.v4.new_markdown_cell("## 1. Executive Summary\n\nThis project involved building a robust fraud detection system starting from raw transactional data. We performed extensive exploratory data analysis (EDA), engineered domain-specific features, and evaluated multiple machine learning models. The final selection, a **Tuned Random Forest**, achieved an **AUC-PR of 0.6470**, providing a highly effective balance between fraud detection and minimizing customer friction."))

    # --- Data Analysis & EDA ---
    cells.append(nbf.v4.new_markdown_cell("## 2. Data Analysis & EDA\n\nWe analyzed the dataset to identify patterns distinguishing fraudulent behavior from legitimate transactions."))

    cells.append(nbf.v4.new_markdown_cell("### 2.1 Class Distribution\nThe dataset exhibits a classic class imbalance, with fraud being the rare event."))
    cells.append(nbf.v4.new_code_cell("from IPython.display import Image, display\ndisplay(Image(filename='../report/figures/01_class_distribution.png'))"))

    cells.append(nbf.v4.new_markdown_cell("### 2.2 Behavior Insights\nFraudulent transactions are strongly correlated with a low 'Time Since Signup', suggesting automated bot-driven attacks."))
    cells.append(nbf.v4.new_code_cell("display(Image(filename='../report/figures/03_time_since_signup.png'))"))

    # --- Feature Engineering ---
    cells.append(nbf.v4.new_markdown_cell("## 3. Feature Engineering\n\nWe transformed raw data into predictive signals across four main categories:\n\n1.  **Velocity Features:** Calculated the time difference between account creation and first transaction.\n2.  **Temporal Features:** Extracted hour and day patterns to identify peak fraud times.\n3.  **Geolocation:** Mapped IP addresses to countries to identify high-risk regions.\n4.  **Network Features:** Identified shared devices and IP addresses to detect botnet activity."))

    # --- Model Performance & Comparison ---
    cells.append(nbf.v4.new_markdown_cell("## 4. Model Performance & Comparison\n\nWe evaluated multiple models using metrics specifically suited for imbalanced data (AUC-PR and F1-score)."))

    cells.append(nbf.v4.new_markdown_cell("### 4.1 Evaluation Metrics\nCompared to the Logistic Regression baseline, ensemble models showed significant performance gains."))
    cells.append(nbf.v4.new_code_cell("display(Image(filename='../report/figures/04_model_comparison_auc_pr.png'))\ndisplay(Image(filename='../report/figures/05_model_comparison_f1.png'))"))

    cells.append(nbf.v4.new_markdown_cell("### 4.2 Final Model Choice: Tuned Random Forest\nWe selected the **Tuned Random Forest** model. It optimizes the precision-recall trade-off through custom hyperparameter tuning, making it the most reliable tool for production deployment."))

    # --- Model Interpretability (SHAP) ---
    cells.append(nbf.v4.new_markdown_cell("## 5. Model Interpretability (SHAP Analysis)\n\nTo ensure the model is 'explainable,' we used SHAP values to visualize how different features impact predictions."))

    cells.append(nbf.v4.new_markdown_cell("### 5.1 Global Drivers\nSHAP summary analysis confirms that **transaction velocity** (time since signup) is the single most important factor in identifying fraud."))

    cells.append(nbf.v4.new_markdown_cell("### 5.2 Local Case Studies\nWe performed deep dives into specific predictions (True Positives, False Positives, and False Negatives) to understand the model's logic in real-world scenarios. This analysis helps us refine the model and understand when it might fail."))

    # --- Business Recommendations ---
    cells.append(nbf.v4.new_markdown_cell("## 6. Business Recommendations\n\nBased on our findings, we recommend the following strategic actions:\n\n1.  **Enforce a 'Cooling Period':** Introduce a mandatory 15-minute delay for first transactions from new accounts.\n2.  **Trigger Step-up Authentication:** Use SMS or 2FA for transactions identified with high-risk velocity scores.\n3.  **IP Reputation Scoring:** Prioritize reviews for transactions originating from regions identified as fraud hotspots."))

    nb.cells = cells

    with open('notebooks/6-project-summary-report.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

    print("Summary notebook created successfully at notebooks/6-project-summary-report.ipynb")

if __name__ == "__main__":
    create_summary_notebook()
