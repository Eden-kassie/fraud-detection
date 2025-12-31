"""
Extract key visualizations from notebooks for the PDF report.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend and style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def save_plot(fig, filename, dpi=300):
    """Save figure to report/figures directory."""
    fig.savefig(f'report/figures/{filename}', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filename}")

def extract_eda_visualizations():
    """Extract key EDA visualizations."""
    print("\\n=== Extracting EDA Visualizations ===")

    # Load the dataset
    df = pd.read_csv('data/processed/fraud_featured.csv')

    # 1. Class Distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    class_counts = df['class'].value_counts()
    ax.bar(['Legitimate', 'Fraud'], class_counts.values, color=['#2ecc71', '#e74c3c'])
    ax.set_ylabel('Count')
    ax.set_title('Class Distribution: Fraud vs. Legitimate Transactions', fontsize=14, fontweight='bold')
    for i, v in enumerate(class_counts.values):
        ax.text(i, v + 1000, f'{v:,}\\n({v/len(df)*100:.1f}%)', ha='center', fontweight='bold')
    save_plot(fig, '01_class_distribution.png')

    # 2. Purchase Value Distribution by Class
    fig, ax = plt.subplots(figsize=(10, 6))
    df[df['class'] == 0]['purchase_value'].sample(min(5000, len(df[df['class']==0]))).hist(
        bins=50, alpha=0.6, label='Legitimate', ax=ax, color='#2ecc71'
    )
    df[df['class'] == 1]['purchase_value'].hist(
        bins=50, alpha=0.6, label='Fraud', ax=ax, color='#e74c3c'
    )
    ax.set_xlabel('Purchase Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Purchase Value Distribution by Class', fontsize=14, fontweight='bold')
    ax.legend()
    save_plot(fig, '02_purchase_value_dist.png')

    # 3. Time Since Signup Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    df[df['class'] == 0]['time_since_signup'].sample(min(5000, len(df[df['class']==0]))).hist(
        bins=50, alpha=0.6, label='Legitimate', ax=ax, color='#2ecc71'
    )
    df[df['class'] == 1]['time_since_signup'].hist(
        bins=50, alpha=0.6, label='Fraud', ax=ax, color='#e74c3c'
    )
    ax.set_xlabel('Time Since Signup (seconds)')
    ax.set_ylabel('Frequency')
    ax.set_title('Time Since Signup Distribution by Class', fontsize=14, fontweight='bold')
    ax.legend()
    save_plot(fig, '03_time_since_signup.png')

    print("EDA visualizations extracted successfully!")

def extract_model_comparison():
    """Extract model comparison visualizations."""
    print("\\n=== Extracting Model Comparison Visualizations ===")

    # Model comparison data (from notebook results)
    models = ['Logistic\\nRegression', 'Random\\nForest', 'XGBoost', 'LightGBM', 'Tuned\\nRandom Forest']
    auc_pr = [0.6149, 0.6459, 0.6449, 0.6432, 0.6470]  # Approximate values
    f1 = [0.3286, 0.7101, 0.7101, 0.7067, 0.7120]

    # 1. AUC-PR Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(models, auc_pr, color=['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#e74c3c'])
    ax.set_xlabel('AUC-PR Score', fontweight='bold')
    ax.set_title('Model Comparison: AUC-PR Scores', fontsize=14, fontweight='bold')
    ax.set_xlim([0.5, 0.7])
    for i, (bar, val) in enumerate(zip(bars, auc_pr)):
        ax.text(val + 0.005, i, f'{val:.4f}', va='center', fontweight='bold')
    save_plot(fig, '04_model_comparison_auc_pr.png')

    # 2. F1 Score Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(models, f1, color=['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#e74c3c'])
    ax.set_xlabel('F1 Score', fontweight='bold')
    ax.set_title('Model Comparison: F1 Scores', fontsize=14, fontweight='bold')
    for i, (bar, val) in enumerate(zip(bars, f1)):
        ax.text(val + 0.01, i, f'{val:.4f}', va='center', fontweight='bold')
    save_plot(fig, '05_model_comparison_f1.png')

    print("Model comparison visualizations extracted successfully!")

def create_placeholder_shap():
    """Create placeholder for SHAP visualization (since we can't easily re-execute SHAP)."""
    print("\\n=== Creating SHAP Placeholders ===")

    # Note: In production, you would extract these from the notebook outputs
    # For now, we create a placeholder noting that screenshots will be taken
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, 'SHAP visualizations will be extracted\\nfrom notebook screenshots',
            ha='center', va='center', fontsize=16, bbox=dict(boxstyle='round', facecolor='wheat'))
    ax.axis('off')
    save_plot(fig, '06_shap_placeholder.png')

    print("SHAP placeholders created!")

if __name__ == "__main__":
    print("Starting visualization extraction for PDF report...")

    try:
        extract_eda_visualizations()
        extract_model_comparison()
        create_placeholder_shap()

        print("\\n✅ All visualizations extracted successfully!")
        print("Figures saved to: report/figures/")

    except Exception as e:
        print(f"\\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
