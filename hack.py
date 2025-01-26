"""
AI-Powered Invoice Matching System with Continuous Learning
"""

import pandas as pd
from rapidfuzz import fuzz, process
import streamlit as st
import re
import joblib
import sqlite3
import threading
import json
import datetime
import os
import time
from unidecode import unidecode
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from uuid import uuid4
import traceback

# ==================================================
#               GLOBAL CONSTANTS
# ==================================================
TOP_N = 10
FEATURE_COLS = [
    "partial_ratio", "token_set_ratio", "length_diff",
    "common_digits", "prefix_match", "suffix_match",
    "digit_ratio", "substring_match", "qr_ratio",
    "char_overlap", "seq_match"
]
MODEL_FILE = "invoice_matcher_model.pkl"

# ==================================================
#               DATABASE SETUP
# ==================================================
def get_db():
    DB_PATH = os.path.abspath('feedback.db')
    conn = sqlite3.connect(
        DB_PATH,
        detect_types=sqlite3.PARSE_DECLTYPES,
        check_same_thread=False
    )
    conn.execute('''CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        invoice_1 TEXT NOT NULL,
        invoice_2 TEXT NOT NULL,
        is_correct INTEGER NOT NULL,
        features TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    return conn

# ==================================================
#               MACHINE LEARNING MODEL
# ==================================================
class OnlineLearner:
    def __init__(self):
        self.model = SGDClassifier(loss='log_loss', warm_start=True)
        self.scaler = StandardScaler()
        self.classes = np.array([0, 1])
        self.feedback_history = []
        
    def partial_fit(self, X, y):
        if len(X) > 0:
            self.scaler.partial_fit(X)
            X_scaled = self.scaler.transform(X)
            y_np = np.array(y)
            
            if len(np.unique(self.feedback_history)) >= 2:
                class_weights = compute_class_weight(
                    'balanced',
                    classes=self.classes,
                    y=np.array(self.feedback_history)
                )
                sample_weights = np.where(y_np == 1, class_weights[1], class_weights[0])
            else:
                sample_weights = np.ones(len(y_np))
            
            self.model.partial_fit(X_scaled, y_np, classes=self.classes, sample_weight=sample_weights)
    
    def predict_proba(self, X):
        if hasattr(self.model, 'coef_'):
            X_scaled = self.scaler.transform(X)
            return self.model.predict_proba(X_scaled)
        return np.array([[0.4, 0.6]] * len(X))

def load_model():
    try:
        return joblib.load(MODEL_FILE)
    except FileNotFoundError:
        return OnlineLearner()

def save_model(model):
    joblib.dump(model, MODEL_FILE)

# ==================================================
#               CORE FUNCTIONALITY
# ==================================================
def preprocess_invoice(invoice):
    processed = unidecode(str(invoice)).upper()
    processed = re.sub(r'[^A-Z0-9]', '', processed).lstrip('0')
    return processed if processed else "MISSING"

def extract_features(inv1, inv2):
    p1 = preprocess_invoice(inv1)
    p2 = preprocess_invoice(inv2)
    
    def safe_digit_ratio(processed):
        digits = re.findall(r'\d', processed)
        return len(digits) / max(len(processed), 1)
    
    return [
        fuzz.partial_ratio(p1, p2),
        fuzz.token_set_ratio(p1, p2),
        abs(len(p1) - len(p2)),
        len(set(re.findall(r'\d', p1)) & set(re.findall(r'\d', p2))),
        int(p1[:3] == p2[:3]),
        int(p1[-3:] == p2[-3:]),
        abs(safe_digit_ratio(p1) - safe_digit_ratio(p2)),
        int(p1 in p2 or p2 in p1),
        fuzz.QRatio(p1, p2),
        len(set(p1) & set(p2)),
        int(len(set(re.findall(r'\d{3,}', p1)) & set(re.findall(r'\d{3,}', p2))) > 0)
    ]

def find_matches(df1, df2):
    blocks = {'length': {}, 'digits': {}, 'alpha': {}}
    
    for idx, row in df2.iterrows():
        processed = row['processed']
        blocks['length'].setdefault(len(processed), []).append(processed)
        blocks['digits'].setdefault(''.join(re.findall(r'\d', processed))[:5], []).append(processed)
        blocks['alpha'].setdefault(''.join(re.findall(r'[A-Z]', processed))[:3], []).append(processed)

    matches = []
    for _, row in df1.iterrows():
        processed = row["processed"]
        candidates = set()
        candidates.update(blocks['length'].get(len(processed), []))
        candidates.update(blocks['digits'].get(''.join(re.findall(r'\d', processed))[:5], []))
        candidates.update(blocks['alpha'].get(''.join(re.findall(r'[A-Z]', processed))[:3], []))
        
        if candidates:
            results = process.extract(
                processed,
                list(candidates),
                scorer=fuzz.token_set_ratio,
                limit=TOP_N,
                score_cutoff=50
            )
            
            for match, score, _ in results:
                original = df2[df2["processed"] == match]["invoice_number"].values[0]
                features = extract_features(row['invoice_number'], original)
                proba = model.predict_proba([features])[0][1]
                matches.append({
                    'invoice_1': row['invoice_number'],
                    'invoice_2': original,
                    'confidence': int(proba * 100),
                    'fuzz_score': score,
                    **dict(zip(FEATURE_COLS, features))
                })
    
    return pd.DataFrame(matches).drop_duplicates(['invoice_1', 'invoice_2']).sort_values('confidence', ascending=False)

# ==================================================
#               USER INTERFACE COMPONENTS
# ==================================================
def handle_feedback(row, is_correct):
    try:
        features = {col: float(row[col]) for col in FEATURE_COLS}
        db = get_db()
        cursor = db.cursor()
        cursor.execute(
            """INSERT INTO feedback 
            (invoice_1, invoice_2, is_correct, features)
            VALUES (?, ?, ?, ?)""",
            (row['invoice_1'], row['invoice_2'], int(is_correct), json.dumps(features))
        )
        db.commit()
        st.toast("Feedback saved successfully!", icon="✅")
    except Exception as e:
        st.error(f"Error saving feedback: {str(e)}")
    finally:
        if 'db' in locals():
            db.close()

def display_performance():
    try:
        db = get_db()
        feedback_df = pd.read_sql("SELECT * FROM feedback", db)
        
        if not feedback_df.empty:
            features_list = [
                [float(feat_dict[col]) for col in FEATURE_COLS] 
                for feat_dict in feedback_df['features'].apply(json.loads)
            ]
            X = np.array(features_list)
            y = feedback_df['is_correct'].values
            
            if hasattr(model.model, 'coef_'):
                X_scaled = model.scaler.transform(X)
                y_pred = model.model.predict(X_scaled)
                
                cols = st.columns(4)
                cols[0].metric("Accuracy", f"{accuracy_score(y, y_pred):.2%}")
                cols[1].metric("Precision", f"{precision_score(y, y_pred, zero_division=0):.2%}")
                cols[2].metric("Recall", f"{recall_score(y, y_pred, zero_division=0):.2%}")
                cols[3].metric("F1 Score", f"{f1_score(y, y_pred, zero_division=0):.2%}")
                
                daily_perf = feedback_df.set_index('timestamp').resample('D')['is_correct'].mean()
                st.line_chart(daily_perf)
            else:
                st.info("Model needs more feedback to show metrics")
        else:
            st.info("No feedback recorded yet")
    except Exception as e:
        st.error(f"Performance error: {str(e)}")

# ==================================================
#               MAIN APPLICATION
# ==================================================
def main():
    st.set_page_config(
        page_title="Invoice Matcher Pro",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS styling
    st.markdown("""
    <style>
    .navbar {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .navbar-title {
        font-size: 1.8rem;
        color: #2c3e50;
        font-weight: 600;
        margin-right: 2rem;
    }
    .stButton>button {
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

    # Navigation bar
    st.markdown("""
    <div class="navbar">
        <span class="navbar-title">🔍 Intelligent Invoice Matching</span>
        <div style="display: flex; gap: 1rem; align-items: center;">
            <div style="flex-grow: 1;">
                <input type="text" placeholder="Search invoices..." style="width: 100%; padding: 0.5rem; border-radius: 4px; border: 1px solid #ddd;">
            </div>
            <button style="padding: 0.5rem 1rem; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer;">
                📊 Generate Report
            </button>
        </div>
    </div>
    """, unsafe_allow_html=True)

    global model
    model = load_model()

    # Sidebar controls
    st.sidebar.header("System Controls")
    if st.sidebar.button("🧨 Reset All Data"):
        try:
            if os.path.exists(MODEL_FILE):
                os.remove(MODEL_FILE)
            conn = get_db()
            conn.execute("DROP TABLE IF EXISTS feedback")
            conn.commit()
            st.session_state.clear()
            st.success("System reset complete! Refresh the page.")
            time.sleep(2)
            st.rerun()
        except Exception as e:
            st.error(f"Reset failed: {str(e)}")

    # File uploaders
    uploaded_file1 = st.sidebar.file_uploader("Upload Source CSV", type=["csv"])
    uploaded_file2 = st.sidebar.file_uploader("Upload Target CSV", type=["csv"])

    if uploaded_file1 and uploaded_file2:
        try:
            # Data processing
            df1 = pd.read_csv(uploaded_file1).dropna(subset=['invoice_number'])
            df2 = pd.read_csv(uploaded_file2).dropna(subset=['invoice_number'])
            
            df1["processed"] = df1["invoice_number"].apply(preprocess_invoice)
            df2["processed"] = df2["invoice_number"].apply(preprocess_invoice)

            # Data preview section
            with st.expander("📊 Data Preview", expanded=True):
                cols = st.columns(2)
                with cols[0]:
                    st.write("**Source Data Preview**")
                    st.dataframe(df1[['invoice_number', 'processed']].head(), height=200)
                with cols[1]:
                    st.write("**Target Data Preview**")
                    st.dataframe(df2[['invoice_number', 'processed']].head(), height=200)
                
                # Zero-confidence matches toggle
                if 'matches_df' in locals():
                    st.divider()
                    hide_zero = st.checkbox("Hide zero-confidence matches", value=True)
                    preview_df = matches_df.copy()
                    if hide_zero:
                        preview_df = preview_df[preview_df['confidence'] > 0]
                    
                    st.write("**Matches Preview**")
                    st.dataframe(
                        preview_df[['invoice_1', 'invoice_2', 'confidence', 'fuzz_score']],
                        height=300,
                        column_config={
                            "confidence": st.column_config.ProgressColumn(
                                "Confidence",
                                format="%d%%",
                                min_value=0,
                                max_value=100,
                            )
                        }
                    )

            # Match processing
            matches_df = find_matches(df1, df2)
            
            if matches_df.empty:
                st.warning("No matches found. Check input patterns.")
                return
            
            # Confidence filter
            min_confidence = st.slider(
                "Minimum Confidence Threshold (%)",
                min_value=0, max_value=100, value=40, step=5
            )
            filtered = matches_df[matches_df['confidence'] >= min_confidence]

            # Display matches
            st.header("🔗 Potential Matches")
            for _, row in filtered.iterrows():
                cols = st.columns([4, 1, 1, 2])
                cols[0].markdown(f"**{row['invoice_1']}** → **{row['invoice_2']}**")
                cols[1].metric("Confidence", f"{row['confidence']}%")
                cols[2].metric("Fuzz Score", row['fuzz_score'])
                
                with cols[3]:
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("✅ Accept", key=f"accept_{uuid4()}"):
                            handle_feedback(row, 1)
                    with c2:
                        if st.button("❌ Reject", key=f"reject_{uuid4()}"):
                            handle_feedback(row, 0)

            # Performance metrics
            st.header("📈 Learning Progress")
            display_performance()

        except Exception as e:
            st.error(f"Processing error: {str(e)}")
            st.error(traceback.format_exc())

if __name__ == "__main__":
    main()