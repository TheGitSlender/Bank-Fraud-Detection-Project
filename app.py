import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Bank Fraud Detection", page_icon="🏦", layout="wide")

st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        font-size: 18px;
    }
    h1 {
        font-size: 3rem !important;
    }
    h2 {
        font-size: 2rem !important;
    }
    h3 {
        font-size: 1.5rem !important;
    }
    .stMetric label {
        font-size: 1.2rem !important;
    }
    .stMetric .metric-value {
        font-size: 2rem !important;
    }
    p, li, div {
        font-size: 1.1rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("🏦 Bank Transaction Fraud Detection System")
st.markdown("**A Machine Learning Application for Detecting Fraudulent Transactions**")


@st.cache_data
def load_data():
    df = pd.read_csv('other_data/Bank_Transaction_Fraud_Detection.csv')
    
    # Drop unnecessary columns
    columns_to_drop = ["Customer_ID", "Customer_Name", "Transaction_ID", "Merchant_ID",
                      "Customer_Contact", "Transaction_Description", "Customer_Email", 
                      "Transaction_Currency"]
    df.drop(columns=columns_to_drop, axis=1, inplace=True)
    
    # Extract date/time features
    df["year"] = pd.to_datetime(df["Transaction_Date"]).dt.year
    df["month"] = pd.to_datetime(df["Transaction_Date"]).dt.month
    df["day"] = pd.to_datetime(df["Transaction_Date"]).dt.day
    df["hour"] = pd.to_datetime(df["Transaction_Time"], format="%H:%M:%S").dt.hour
    df["minute"] = pd.to_datetime(df["Transaction_Time"], format="%H:%M:%S").dt.minute
    df["second"] = pd.to_datetime(df["Transaction_Time"], format="%H:%M:%S").dt.second
    
    df.drop(["Transaction_Time", "Transaction_Date"], axis=1, inplace=True)
    
    return df

# Preprocess and train models
@st.cache_resource
def train_models(df):
    # Encode categorical features
    df_encoded = df.copy()
    categorical = [i for i in df_encoded.columns if df_encoded[i].dtype == 'O']
    label_encoder = LabelEncoder()
    for col in categorical:
        df_encoded[col] = label_encoder.fit_transform(df_encoded[col])
    
    X = df_encoded.drop(columns=['Is_Fraud'])
    Y = df_encoded["Is_Fraud"]
    
    # Apply SMOTE and undersampling
    over = SMOTE(sampling_strategy=0.3, random_state=42)
    under = RandomUnderSampler(sampling_strategy=0.1, random_state=42)
    steps = [('under', under), ('over', over)]
    pipeline = Pipeline(steps=steps)
    X_resampled, Y_resampled = pipeline.fit_resample(X, Y)
    
    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        X_resampled, Y_resampled, test_size=0.2, stratify=Y_resampled, random_state=42
    )
    
    # Train models
    models = {}
    
    # Logistic Regression
    log_model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    log_model.fit(x_train, y_train)
    models['Logistic Regression'] = log_model
    
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42)
    rf_model.fit(x_train, y_train)
    models['Random Forest'] = rf_model
    
    # XGBoost
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    xgb_model = XGBClassifier(
        n_estimators=400,
        max_depth=9,
        learning_rate=0.03,
        subsample=0.75,
        colsample_bytree=0.75,
        colsample_bylevel=0.8,
        scale_pos_weight=scale_pos_weight * 1.2,
        min_child_weight=2,
        gamma=0.05,
        reg_alpha=0.05,
        reg_lambda=0.8,
        max_delta_step=1,
        eval_metric='aucpr',
        objective='binary:logistic',
        tree_method='hist',
        grow_policy='lossguide',
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    xgb_model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False)
    models['XGBoost'] = xgb_model
    
    return models, x_train, x_test, y_train, y_test, label_encoder, categorical

# Load data
try:
    df = load_data()
    models, x_train, x_test, y_train, y_test, label_encoder, categorical = train_models(df)
    
    tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "📈 Exploratory Analysis", "🤖 Model Performance"])
    
    # Tab 1: Data Overview
    with tab1:
        st.header("Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", f"{len(df):,}")
        with col2:
            fraud_count = df['Is_Fraud'].sum()
            st.metric("Fraudulent Transactions", f"{fraud_count:,}")
        with col3:
            fraud_rate = (fraud_count / len(df)) * 100
            st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
        with col4:
            st.metric("Features", len(df.columns))
        
        st.subheader("Sample Data")
        st.dataframe(df.head(100), use_container_width=True)
        
        st.subheader("Dataset Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Shape:**", df.shape)
            st.write("**Missing Values:**", df.isnull().sum().sum())
        with col2:
            st.write("**Duplicates:**", df.duplicated().sum())
            st.write("**Memory Usage:**", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Tab 2: Exploratory Analysis
    with tab2:
        st.header("Exploratory Data Analysis")
        
        # Fraud Distribution
        st.subheader("Fraud Distribution")
        fraud_counts = df['Is_Fraud'].value_counts()
        fig = go.Figure(data=[
            go.Bar(x=['Not Fraud', 'Fraud'], 
                   y=fraud_counts.values,
                   marker_color=['green', 'red'],
                   text=fraud_counts.values,
                   textposition='auto')
        ])
        fig.update_layout(title="Distribution of Fraudulent vs Non-Fraudulent Transactions",
                         xaxis_title="Transaction Type",
                         yaxis_title="Count",
                         height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Categorical features
        st.subheader("Fraud Analysis by Categories")
        col1, col2 = st.columns(2)
        
        with col1:
            # Gender
            gender_fraud = df.groupby(['Gender', 'Is_Fraud']).size().unstack(fill_value=0)
            fig = go.Figure(data=[
                go.Bar(name='Not Fraud', x=gender_fraud.index, y=gender_fraud[0], marker_color='green'),
                go.Bar(name='Fraud', x=gender_fraud.index, y=gender_fraud[1], marker_color='red')
            ])
            fig.update_layout(title="Fraud Distribution by Gender", barmode='group', height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Device Type
            device_fraud = df.groupby(['Device_Type', 'Is_Fraud']).size().unstack(fill_value=0)
            fig = go.Figure(data=[
                go.Bar(name='Not Fraud', x=device_fraud.index, y=device_fraud[0], marker_color='green'),
                go.Bar(name='Fraud', x=device_fraud.index, y=device_fraud[1], marker_color='red')
            ])
            fig.update_layout(title="Fraud Distribution by Device Type", barmode='group', height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        # Transaction Amount Analysis
        st.subheader("Transaction Amount Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x='Transaction_Amount', nbins=50, 
                             title="Transaction Amount Distribution",
                             color_discrete_sequence=['coral'])
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(df, x='Is_Fraud', y='Transaction_Amount',
                        title="Transaction Amount by Fraud Status",
                        color='Is_Fraud',
                        color_discrete_map={0: 'green', 1: 'red'})
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        # Age Analysis
        st.subheader("Age Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x='Age', nbins=30,
                             title="Age Distribution",
                             color_discrete_sequence=['skyblue'])
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(df, x='Is_Fraud', y='Age',
                        title="Age Distribution by Fraud Status",
                        color='Is_Fraud',
                        color_discrete_map={0: 'green', 1: 'red'})
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Model Performance
    with tab3:
        st.header("Model Performance Comparison")
        
        # Model selector
        model_name = st.selectbox("Select Model", list(models.keys()))
        model = models[model_name]
        
        y_pred = model.predict(x_test)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdYlGn', subset=['precision', 'recall', 'f1-score']), 
                        use_container_width=True)
        
        with col2:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Not Fraud', 'Fraud'],
                       yticklabels=['Not Fraud', 'Fraud'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f'Confusion Matrix - {model_name}')
            st.pyplot(fig)
        
        # Model comparison
        st.subheader("All Models Comparison")
        comparison_data = []
        for name, mdl in models.items():
            pred = mdl.predict(x_test)
            report = classification_report(y_test, pred, output_dict=True)
            comparison_data.append({
                'Model': name,
                'Accuracy': report['accuracy'],
                'Precision (Fraud)': report['1']['precision'],
                'Recall (Fraud)': report['1']['recall'],
                'F1-Score (Fraud)': report['1']['f1-score']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df.style.background_gradient(cmap='RdYlGn', subset=['Accuracy', 'Precision (Fraud)', 'Recall (Fraud)', 'F1-Score (Fraud)']),
                    use_container_width=True)
        
        # Bar chart comparison
        fig = go.Figure(data=[
            go.Bar(name='Accuracy', x=comparison_df['Model'], y=comparison_df['Accuracy']),
            go.Bar(name='Precision', x=comparison_df['Model'], y=comparison_df['Precision (Fraud)']),
            go.Bar(name='Recall', x=comparison_df['Model'], y=comparison_df['Recall (Fraud)']),
            go.Bar(name='F1-Score', x=comparison_df['Model'], y=comparison_df['F1-Score (Fraud)'])
        ])
        fig.update_layout(barmode='group', title="Model Performance Metrics Comparison", height=400)
        st.plotly_chart(fig, use_container_width=True)

except FileNotFoundError:
    st.error("❌ Data file not found! Please make sure 'other_data/Bank_Transaction_Fraud_Detection.csv' exists in your directory.")
    st.info("Upload your CSV file or update the file path in the code.")
except Exception as e:
    st.error(f"❌ An error occurred: {str(e)}")
    st.info("Please check your data file and try again.")