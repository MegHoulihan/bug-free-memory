
import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Column Mapper (Rules + ML)", layout="wide")

HELP_MD = (
    "**Workflow**\n\n"
    "1. Upload a mapping file (or use the demo). The file must have columns `source_column` and `target_label`.\n\n"
    "2. Paste or upload new column names to classify.\n\n"
    "3. Set a confidence threshold. Predictions below this threshold are routed to **REVIEW**.\n\n"
    "4. Download results and/or the trained model.\n\n\n"
    "**Tips**\n\n"
    "- Keep training data tidy and deduplicated for deterministic exact-match rules.\n\n"
    "- Tune the threshold based on validation accuracy and your review bandwidth.\n\n"
    "- Combine this app with a feedback loop: correct predictions → append to mappings → retrain."
)

# ---------------------- Normalization helpers ----------------------
_camel_pat1 = re.compile(r'(.)([A-Z][a-z]+)')
_camel_pat2 = re.compile(r'([a-z0-9])([A-Z])')

def normalize_colname(s: str) -> str:
    s = str(s)
    s = _camel_pat1.sub(r'\1 \2', s)
    s = _camel_pat2.sub(r'\1 \2', s)
    s = s.lower()
    s = re.sub(r'[\-\_\.\/\\]+', ' ', s)
    s = re.sub(r'[^a-z0-9 ]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# ---------------------- Business rules wrapper ----------------------

def assign_with_rules(new_columns, model, exact_map_raw, exact_map_norm, confidence_threshold=0.55, review_label="REVIEW"):
    new_norm = [normalize_colname(c) for c in new_columns]
    rows, to_model_idx = [], []

    for i, (raw, norm) in enumerate(zip(new_columns, new_norm)):
        if raw in exact_map_raw:
            rows.append({
                "source_column": raw,
                "source_norm": norm,
                "assigned_target": exact_map_raw[raw],
                "assignment_source": "rule_raw",
                "confidence": np.nan,
                "top3_suggestions": None
            })
        elif norm in exact_map_norm:
            rows.append({
                "source_column": raw,
                "source_norm": norm,
                "assigned_target": exact_map_norm[norm],
                "assignment_source": "rule_norm",
                "confidence": np.nan,
                "top3_suggestions": None
            })
        else:
            rows.append(None)
            to_model_idx.append(i)

    if to_model_idx:
        texts = [new_norm[i] for i in to_model_idx]
        pred = model.predict(texts)
        has_proba = hasattr(model, 'predict_proba')
        if has_proba:
            proba = model.predict_proba(texts)
            # Try to find classes in a robust way
            classes = None
            if hasattr(model, 'named_steps') and 'clf' in model.named_steps and hasattr(model.named_steps['clf'], 'classes_'):
                classes = model.named_steps['clf'].classes_
            elif hasattr(model, 'classes_'):
                classes = model.classes_
        else:
            proba, classes = None, None

        for j, i in enumerate(to_model_idx):
            raw = new_columns[i]
            norm = new_norm[i]
            p = pred[j]

            if has_proba and proba is not None:
                row_proba = proba[j]
                max_conf = float(row_proba.max())
                import numpy as _np
                topk = _np.argsort(-row_proba)[:3]
                top3 = [(str(classes[k]), float(row_proba[k])) for k in topk]
            else:
                max_conf = np.nan
                top3 = None

            if has_proba and max_conf < confidence_threshold:
                assigned, src = (review_label if review_label else None), 'review'
            else:
                assigned, src = p, 'model'

            rows[i] = {
                "source_column": raw,
                "source_norm": norm,
                "assigned_target": assigned,
                "assignment_source": src,
                "confidence": max_conf,
                "top3_suggestions": top3
            }

    df = pd.DataFrame(rows)
    # Order rows for readability
    order = pd.CategoricalDtype(["rule_raw","rule_norm","model","review"], ordered=True)
    df['assignment_source'] = df['assignment_source'].astype(order)
    df = df.sort_values(["assignment_source","confidence"], ascending=[True, False], na_position='last')
    return df

# ---------------------- Model training ----------------------

def train_model(mdf: pd.DataFrame) -> Pipeline:
    mdf = mdf.copy()
    mdf['source_norm'] = mdf['source_column'].map(normalize_colname)
    X = mdf['source_norm']
    y = mdf['target_label']
    model = Pipeline(steps=[
        ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=1)),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced'))
    ])
    model.fit(X, y)
    return model

# ---------------------- Demo data ----------------------

def demo_mapping() -> pd.DataFrame:
    return pd.DataFrame({
        'source_column': [
            'acct_id','account_id','accountid','account_identifier',
            'first_name','fname','given_name',
            'last_name','lname','surname',
            'email','email_address','e_mail',
            'phone','phone_number','mobile',
            'zip','zipcode','postal_code',
            'created_at','create_date','createdon'
        ],
        'target_label': [
            'Account ID','Account ID','Account ID','Account ID',
            'First Name','First Name','First Name',
            'Last Name','Last Name','Last Name',
            'Email','Email','Email',
            'Phone','Phone','Phone',
            'Postal Code','Postal Code','Postal Code',
            'Created Date','Created Date','Created Date'
        ]
    })

# ---------------------- UI ----------------------
st.title("🧭 Column Name → Target Label Mapper (Rules + ML)")

with st.sidebar:
    st.header("Settings")
    use_demo = st.checkbox("Use demo mapping", value=True,
                           help="Check to try the app with built-in demo mappings. Uncheck to upload your own.")
    threshold = st.slider("Confidence threshold for REVIEW routing", 0.0, 1.0, 0.55, 0.01)
    review_label = st.text_input("Review label (blank for empty)", "REVIEW")

    st.markdown("---")
    st.caption("Upload mapping file (CSV/XLSX with columns: source_column, target_label)")
    uploaded_map = st.file_uploader("Mapping file", type=["csv","xlsx"], key="mapping")

    st.caption("Upload new columns (TXT/CSV with one column name per line or in the first column)")
    uploaded_new = st.file_uploader("New columns file (optional)", type=["txt","csv"], key="newcols")

    st.caption("Or paste new columns below (one per line)")
    pasted = st.text_area("Pasted new columns", """AccountIdentifier
emailAddress
MobilePhone
CreatedOn
zip_code
lastName
acct_id
Account_Id
unknown_field_xyz""",
                          height=180)

    run = st.button("▶️ Run Mapping", type="primary")

# Load mapping
mapping_df = None
warning_msgs = []
if use_demo:
    mapping_df = demo_mapping()
else:
    if uploaded_map is not None:
        name = uploaded_map.name.lower()
        try:
            if name.endswith('.csv'):
                mapping_df = pd.read_csv(uploaded_map)
            else:
                mapping_df = pd.read_excel(uploaded_map, engine='openpyxl')
        except Exception as e:
            st.error(f"Failed to read mapping file: {e}")
    else:
        st.info("Upload a mapping file in the sidebar or switch on 'Use demo mapping'.")

# Validate and normalize mapping
if mapping_df is not None:
    mapping_df = mapping_df.rename(columns={c: c.lower() for c in mapping_df.columns})
    if not {'source_column','target_label'}.issubset(mapping_df.columns):
        st.error("Mapping file must contain columns: source_column, target_label")
        mapping_df = None
    else:
        mapping_df = mapping_df.dropna(subset=['source_column','target_label']).copy()
        mapping_df['source_column'] = mapping_df['source_column'].astype(str)
        mapping_df['target_label'] = mapping_df['target_label'].astype(str)
        mapping_df['source_norm'] = mapping_df['source_column'].map(normalize_colname)

        # Build rule dictionaries
        exact_map_raw = dict(zip(mapping_df['source_column'], mapping_df['target_label']))
        exact_map_norm = dict(zip(mapping_df['source_norm'],   mapping_df['target_label']))

        # Duplicate checks
        dupes_raw = mapping_df.duplicated(subset=['source_column'], keep=False)
        dupes_norm = mapping_df.duplicated(subset=['source_norm'], keep=False)
        if dupes_raw.any() or dupes_norm.any():
            warning_msgs.append("Duplicate source names detected in mapping; rule overrides may be ambiguous.")

# Show mapping preview
if mapping_df is not None:
    st.subheader("Mapping preview")
    st.dataframe(mapping_df.head(20), use_container_width=True)
    if warning_msgs:
        for m in warning_msgs:
            st.warning(m)

# Run
if run and mapping_df is not None:
    # Train model
    with st.spinner("Training model..."):
        model = train_model(mapping_df)

    # Get new columns from upload and/or paste
    new_cols = []
    if uploaded_new is not None:
        try:
            if uploaded_new.name.lower().endswith('.txt'):
                content = uploaded_new.read().decode('utf-8', errors='ignore')
                new_cols.extend([line.strip() for line in content.splitlines() if line.strip()])
            else:
                df_in = pd.read_csv(uploaded_new)
                first_col = df_in.columns[0]
                new_cols.extend(df_in[first_col].astype(str).tolist())
        except Exception as e:
            st.error(f"Could not read new columns file: {e}")
    pasted_cols = [line.strip() for line in pasted.splitlines() if line.strip()]
    new_cols.extend(pasted_cols)

    if not new_cols:
        st.warning("Provide new columns via upload or paste.")
    else:
        with st.spinner("Predicting..."):
            hybrid_df = assign_with_rules(
                new_columns=new_cols,
                model=model,
                exact_map_raw=exact_map_raw,
                exact_map_norm=exact_map_norm,
                confidence_threshold=float(threshold),
                review_label=(review_label if review_label else None)
            )

        st.subheader("Hybrid results")
        st.dataframe(hybrid_df, use_container_width=True)

        review_queue = hybrid_df[hybrid_df['assignment_source'] == 'review'].copy()
        final_assignments = hybrid_df[hybrid_df['assignment_source'].isin(['rule_raw','rule_norm','model'])].copy()

        colA, colB, colC = st.columns(3)
        with colA:
            st.download_button(
                "Download hybrid_results.csv",
                data=hybrid_df.to_csv(index=False),
                file_name="hybrid_results.csv",
                mime="text/csv"
            )
        with colB:
            st.download_button(
                "Download review_queue.csv",
                data=review_queue.to_csv(index=False),
                file_name="review_queue.csv",
                mime="text/csv"
            )
        with colC:
            st.download_button(
                "Download final_assignments.csv",
                data=final_assignments.to_csv(index=False),
                file_name="final_assignments.csv",
                mime="text/csv"
            )

        # Offer model artifact download
        with st.expander("Export trained model (joblib)"):
            buf = io.BytesIO()
            joblib.dump({
                'model': model,
                'trained_with_labels': sorted(mapping_df['target_label'].unique().tolist()),
                'note': 'Include normalize_colname logic in your inference environment.'
            }, buf)
            buf.seek(0)
            st.download_button(
                "Download column_name_labeler.joblib",
                data=buf,
                file_name="column_name_labeler.joblib",
                mime="application/octet-stream"
            )

# Footer help
with st.expander("Help & Notes"):
    st.markdown(HELP_MD)
