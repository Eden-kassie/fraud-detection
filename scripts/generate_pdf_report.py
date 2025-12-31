from fpdf import FPDF
import os

class FraudReport(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 10, 'Fraud Detection Project Report', border=False, ln=True, align='C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def chapter_title(self, title):
        self.set_font('helvetica', 'B', 14)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, ln=True, fill=True)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('helvetica', '', 11)
        self.multi_cell(0, 7, body)
        self.ln()

    def add_image_with_caption(self, image_path, caption, w=150):
        if os.path.exists(image_path):
            # Center the image
            x = (210 - w) / 2
            self.image(image_path, x=x, w=w)
            self.set_font('helvetica', 'I', 9)
            self.cell(0, 10, caption, ln=True, align='C')
            self.ln(5)
        else:
            print(f"Warning: Image {image_path} not found.")

def generate_pdf():
    pdf = FraudReport()
    pdf.add_page()

    # --- Executive Summary ---
    pdf.chapter_title('1. Executive Summary')
    summary = (
        "This report details the development of a high-performance fraud detection system. "
        "Starting from raw transactional data, we performed extensive exploratory data analysis (EDA), "
        "engineered domain-specific features, and evaluated multiple machine learning models. "
        "The project culminated in the selection of a Tuned Random Forest model as the production "
        "candidate, achieving an AUC-PR of 0.6470, a significant improvement over the baseline."
    )
    pdf.chapter_body(summary)

    # --- Data Analysis ---
    pdf.chapter_title('2. Data Analysis & EDA')
    eda_text = (
        "The dataset is heavily imbalanced, with only a small fraction of transactions being fraudulent. "
        "This imbalance requires metrics like AUC-PR and F1-score for evaluation."
    )
    pdf.chapter_body(eda_text)
    pdf.add_image_with_caption('report/figures/01_class_distribution.png', 'Figure 1: Class Distribution')

    insight_text = (
        "Key insights from the EDA revealed that 'Time Since Signup' is a critical predictor. "
        "Fraudulent activity often occurs immediately after account creation, indicating automated attacks."
    )
    pdf.chapter_body(insight_text)
    pdf.add_image_with_caption('report/figures/03_time_since_signup.png', 'Figure 2: Time Since Signup (Seconds) by Class')

    # --- Feature Engineering ---
    pdf.add_page()
    pdf.chapter_title('3. Feature Engineering')
    fe_text = (
        "Six key categories of features were developed:\n"
        "1. Temporal Features: Capturing fraud-prone hours and days.\n"
        "2. Velocity Features: Calculating time since signup (first transaction speed).\n"
        "3. Geolocation Mapping: Converting IP addresses to countries to flag high-risk regions.\n"
        "4. Device/IP Sharing: Detecting multiple users on the same infrastructure."
    )
    pdf.chapter_body(fe_text)

    # --- Model Development ---
    pdf.chapter_title('4. Model Performance & Selection')
    model_text = (
        "We compared a Logistic Regression baseline against Random Forest, XGBoost, and LightGBM. "
        "The Tuned Random Forest model was selected for its superior performance on imbalanced data."
    )
    pdf.chapter_body(model_text)
    pdf.add_image_with_caption('report/figures/04_model_comparison_auc_pr.png', 'Figure 3: AUC-PR Comparison Across Models')
    pdf.add_image_with_caption('report/figures/05_model_comparison_f1.png', 'Figure 4: F1 Score Comparison')

    # --- SHAP Analysis ---
    pdf.add_page()
    pdf.chapter_title('5. Model Interpretability (SHAP)')
    shap_text = (
        "To ensure transparency, we utilized SHAP values to explain model predictions.\n\n"
        "Global Results: The 'time_since_signup' feature is the most influential factor. "
        "Transactions occurring within seconds of signup are almost universally flagged as fraud.\n\n"
        "Local Predictions: We analyzed True Positives (correctly blocked fraud), "
        "False Positives (legitimate users blocked), and False Negatives (missed fraud) to improve model refinement."
    )
    pdf.chapter_body(shap_text)

    # --- Recommendations ---
    pdf.chapter_title('6. Business Recommendations')
    rec_text = (
        "1. Implement a 15-minute 'Cooling Period' for new accounts.\n"
        "2. Trigger Step-up Authentication (2FA) for transactions with low time-since-signup.\n"
        "3. Monitor high-risk IP/Country combinations identified in the geolocation analysis."
    )
    pdf.chapter_body(rec_text)

    # Output the PDF
    output_path = 'report/Fraud_Detection_Project_Report.pdf'
    pdf.output(output_path)
    print(f"PDF Report generated: {output_path}")

if __name__ == "__main__":
    generate_pdf()
