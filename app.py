import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Placement Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
            
    /* Hide the sidebar header (app text) */
    [data-testid="stSidebarNav"] {
        display: none;
    }
            
    /* Global text */
    html, body, [class*="css"]  {
        color: #EAEAEA;
        background-color: #0E1117;
    }

    /* Section titles */
    .section-title {
        font-size: 20px;
        font-weight: 600;
        color: #FFFFFF;   /* FIXED: bright text */
        margin: 1.5rem 0 0.8rem;
        border-left: 4px solid #00C2A8;
        padding-left: 10px;
    }

    /* Metric cards */
    .metric-card {
        background: #1A1F2B;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border: 1px solid #2A2F3A;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    }

    .metric-label { 
        font-size: 13px; 
        color: #A0A7B5; 
        margin-bottom: 4px; 
    }

    .metric-value { 
        font-size: 28px; 
        font-weight: 600; 
        color: #FFFFFF; 
    }

    .metric-sub { 
        font-size: 12px; 
        color: #8892A6; 
        margin-top: 2px; 
    }

    /* Prediction boxes */
    .predict-box {
        background: #132A26;
        border: 1px solid #00C2A8;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1rem;
        color: #EAEAEA;
    }

    .predict-box-red {
        background: #2A1A1A;
        border: 1px solid #FF6B6B;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1rem;
        color: #EAEAEA;
    }

</style>
""", unsafe_allow_html=True)

# ── Load data & model ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/Placement_Data_Full_Class.csv")
    df.drop(columns=["sl_no"], inplace=True)
    return df

@st.cache_resource
def load_model():
    model        = joblib.load("app/model.pkl")
    scaler       = joblib.load("app/scaler.pkl")
    feature_names= joblib.load("app/feature_names.pkl")
    le_dict      = joblib.load("app/label_encoders.pkl")
    with open("app/model_metadata.json") as f:
        meta = json.load(f)
    return model, scaler, feature_names, le_dict, meta

df                               = load_data()
model, scaler, feature_names, le_dict, meta = load_model()

placed_df    = df[df["status"] == "Placed"]
not_placed_df= df[df["status"] == "Not Placed"]
placed_count = len(placed_df)
total        = len(df)
placement_rate = placed_count / total * 100

PLACED_COLOR   = "#6BAF92"
UNPLACED_COLOR = "#E07B7B"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Placement Analytics")
    st.markdown("**Campus Recruitment System**")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📊 Dashboard", "🤖 Predict Placement", "📈 Detailed Analysis"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown(f"**Dataset:** benroshan (Kaggle)")
    st.markdown(f"**Students:** {total}")
    st.markdown(f"**Best Model:** {meta['best_model']}")
    st.markdown(f"**Accuracy:** {meta['accuracy']*100:.2f}%")
    st.markdown(f"**ROC-AUC:** {meta['roc_auc']:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.title("📊 Placement Analytics Dashboard")
    st.markdown("Campus recruitment insights — Academic & Employability factors")
    st.markdown("---")

    # ── KPI row ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Total Students</div>
            <div class="metric-value">{total}</div>
            <div class="metric-sub">in dataset</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Placed</div>
            <div class="metric-value" style="color:#6BAF92">{placed_count}</div>
            <div class="metric-sub">{placement_rate:.1f}% placement rate</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        avg_salary = placed_df["salary"].mean()
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Avg Salary (Placed)</div>
            <div class="metric-value">₹{avg_salary/100000:.2f}L</div>
            <div class="metric-sub">per annum</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        workex_rate = df[df["workex"]=="Yes"]["status"].eq("Placed").mean()*100
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Work Exp → Placed</div>
            <div class="metric-value">{workex_rate:.1f}%</div>
            <div class="metric-sub">vs {df[df["workex"]=="No"]["status"].eq("Placed").mean()*100:.1f}% without</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Row 1: Placement dist + Gender ──
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-title">Placement Distribution</p>', unsafe_allow_html=True)
        counts = df["status"].value_counts()
        fig = px.pie(
            names=counts.index, values=counts.values,
            color=counts.index,
            color_discrete_map={"Placed": PLACED_COLOR, "Not Placed": UNPLACED_COLOR},
            hole=0.45
        )
        fig.update_traces(textinfo="percent+label", textfont_size=14)
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-title">Placement by Gender</p>', unsafe_allow_html=True)
        gender_ct = pd.crosstab(df["gender"], df["status"])
        gender_ct_pct = gender_ct.div(gender_ct.sum(axis=1), axis=0) * 100
        fig = go.Figure()
        for status, color in [("Placed", PLACED_COLOR), ("Not Placed", UNPLACED_COLOR)]:
            if status in gender_ct_pct.columns:
                fig.add_trace(go.Bar(
                    name=status, x=gender_ct_pct.index,
                    y=gender_ct_pct[status].round(1),
                    marker_color=color, text=gender_ct_pct[status].round(1).astype(str)+"%",
                    textposition="outside"
                ))
        fig.update_layout(
            barmode="group", height=300,
            margin=dict(t=10, b=10, l=10, r=10),
            yaxis_title="Percentage (%)", xaxis_title="",
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: Scores comparison ──
    st.markdown('<p class="section-title">Academic Scores — Placed vs Not Placed</p>', unsafe_allow_html=True)
    score_cols   = ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"]
    score_labels = ["10th %", "12th %", "Degree %", "E-Test %", "MBA %"]

    fig = go.Figure()
    placed_means   = [placed_df[c].mean() for c in score_cols]
    unplaced_means = [not_placed_df[c].mean() for c in score_cols]

    fig.add_trace(go.Bar(
        name="Placed", x=score_labels, y=[round(v,2) for v in placed_means],
        marker_color=PLACED_COLOR, text=[f"{v:.1f}%" for v in placed_means],
        textposition="outside"
    ))
    fig.add_trace(go.Bar(
        name="Not Placed", x=score_labels, y=[round(v,2) for v in unplaced_means],
        marker_color=UNPLACED_COLOR, text=[f"{v:.1f}%" for v in unplaced_means],
        textposition="outside"
    ))
    fig.update_layout(
        barmode="group", height=350,
        margin=dict(t=30, b=10, l=10, r=10),
        yaxis_title="Average Score (%)",
        legend=dict(orientation="h", y=1.05)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Work experience + Degree type ──
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-title">Work Experience Impact</p>', unsafe_allow_html=True)
        we_ct = pd.crosstab(df["workex"], df["status"])
        we_pct = we_ct.div(we_ct.sum(axis=1), axis=0) * 100
        fig = go.Figure()
        for status, color in [("Placed", PLACED_COLOR), ("Not Placed", UNPLACED_COLOR)]:
            if status in we_pct.columns:
                fig.add_trace(go.Bar(
                    name=status, x=we_pct.index, y=we_pct[status].round(1),
                    marker_color=color,
                    text=we_pct[status].round(1).astype(str)+"%",
                    textposition="outside"
                ))
        fig.update_layout(
            barmode="group", height=300,
            margin=dict(t=10, b=10, l=10, r=10),
            yaxis_title="%", xaxis_title="Work Experience",
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-title">UG Degree Type vs Placement</p>', unsafe_allow_html=True)
        deg_ct = pd.crosstab(df["degree_t"], df["status"])
        deg_pct = deg_ct.div(deg_ct.sum(axis=1), axis=0) * 100
        fig = go.Figure()
        for status, color in [("Placed", PLACED_COLOR), ("Not Placed", UNPLACED_COLOR)]:
            if status in deg_pct.columns:
                fig.add_trace(go.Bar(
                    name=status, x=deg_pct.index, y=deg_pct[status].round(1),
                    marker_color=color,
                    text=deg_pct[status].round(1).astype(str)+"%",
                    textposition="outside"
                ))
        fig.update_layout(
            barmode="group", height=300,
            margin=dict(t=10, b=10, l=10, r=10),
            yaxis_title="%", xaxis_title="Degree Type",
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Predict Placement":
    st.title("🤖 Placement Predictor")
    st.markdown("Fill in student details below to predict placement chances.")
    st.markdown("---")

    col_form, col_result = st.columns([1.1, 0.9])

    with col_form:
        st.markdown("#### Student Profile")

        c1, c2 = st.columns(2)
        with c1:
            gender = st.selectbox("Gender", ["M", "F"],
                                  format_func=lambda x: "Male" if x=="M" else "Female")
            ssc_b  = st.selectbox("10th Board", ["Central", "Others"])
            hsc_b  = st.selectbox("12th Board", ["Central", "Others"])
        with c2:
            workex = st.selectbox("Work Experience", ["No", "Yes"])
            hsc_s  = st.selectbox("12th Stream", ["Science", "Commerce", "Arts"])
            degree_t = st.selectbox("UG Degree Type",
                                    ["Sci&Tech", "Comm&Mgmt", "Others"])

        specialisation = st.selectbox("MBA Specialisation", ["Mkt&HR", "Mkt&Fin"])

        st.markdown("#### Academic Scores")
        c1, c2 = st.columns(2)
        with c1:
            ssc_p    = st.slider("10th Percentage (%)",    40.0, 100.0, 67.0, 0.5)
            hsc_p    = st.slider("12th Percentage (%)",    40.0, 100.0, 66.0, 0.5)
            degree_p = st.slider("Degree Percentage (%)",  40.0, 100.0, 66.0, 0.5)
        with c2:
            etest_p  = st.slider("E-Test Percentage (%)",  40.0, 100.0, 72.0, 0.5)
            mba_p    = st.slider("MBA Percentage (%)",     40.0, 100.0, 62.0, 0.5)

        predict_btn = st.button("🔍 Predict Placement", type="primary", use_container_width=True)

    with col_result:
        st.markdown("#### Prediction Result")

        if predict_btn:
            # Build input row in same order as training features
            input_dict = {
                "gender":         gender,
                "ssc_p":          ssc_p,
                "ssc_b":          ssc_b,
                "hsc_p":          hsc_p,
                "hsc_b":          hsc_b,
                "hsc_s":          hsc_s,
                "degree_p":       degree_p,
                "degree_t":       degree_t,
                "workex":         workex,
                "etest_p":        etest_p,
                "specialisation": specialisation,
                "mba_p":          mba_p,
            }

            row = {}
            for feat in feature_names:
                val = input_dict[feat]
                if feat in le_dict:
                    try:
                        val = le_dict[feat].transform([val])[0]
                    except ValueError:
                        val = 0
                row[feat] = val

            X_input = pd.DataFrame([row])[feature_names]
            X_scaled = scaler.transform(X_input)

            prediction = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)[0]
            placed_prob = round(probability[1] * 100, 1)
            not_placed_prob = round(probability[0] * 100, 1)

            # Result box
            if prediction == 1:
                st.markdown(f"""<div class="predict-box">
                    <div style="font-size:48px">✅</div>
                    <div style="font-size:22px;font-weight:600;color:#2d7a5a;margin:8px 0">LIKELY PLACED</div>
                    <div style="font-size:36px;font-weight:700;color:#6BAF92">{placed_prob}%</div>
                    <div style="color:#555;font-size:14px">placement probability</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="predict-box-red">
                    <div style="font-size:48px">⚠️</div>
                    <div style="font-size:22px;font-weight:600;color:#a03030;margin:8px 0">PLACEMENT AT RISK</div>
                    <div style="font-size:36px;font-weight:700;color:#E07B7B">{placed_prob}%</div>
                    <div style="color:#555;font-size:14px">placement probability</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("")

            # Probability gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=placed_prob,
                number={"suffix": "%", "font": {"size": 28}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": PLACED_COLOR if prediction==1 else UNPLACED_COLOR},
                    "steps": [
                        {"range": [0, 40],  "color": "#ffe5e5"},
                        {"range": [40, 65], "color": "#fff8e1"},
                        {"range": [65, 100],"color": "#e5f5ee"},
                    ],
                    "threshold": {
                        "line": {"color": "#333", "width": 3},
                        "thickness": 0.75, "value": 65
                    }
                }
            ))
            fig.update_layout(height=220, margin=dict(t=20, b=0, l=30, r=30))
            st.plotly_chart(fig, use_container_width=True)

            # Advice
            st.markdown("#### Suggestions")
            if ssc_p < 60:
                st.warning("10th score is below average for placed students (avg ~67%)")
            if hsc_p < 65:
                st.warning("12th score is below average for placed students (avg ~69%)")
            if degree_p < 66:
                st.warning("Degree % is below average for placed students (avg ~73%)")
            if etest_p < 70:
                st.info("Improving employability test score can boost chances significantly")
            if workex == "No":
                st.info("Students with work experience have significantly higher placement rates")
            if prediction == 1:
                st.success("Profile looks strong! Focus on interview preparation.")

        else:
            st.info("👈 Fill in the student details and click **Predict Placement**")

            # Show model info
            st.markdown("---")
            st.markdown("**Model Performance**")
            c1, c2 = st.columns(2)
            c1.metric("Accuracy",  f"{meta['accuracy']*100:.2f}%")
            c2.metric("ROC-AUC",   f"{meta['roc_auc']:.4f}")
            c1.metric("CV Score",  f"{meta['cv_accuracy']*100:.2f}%")
            c2.metric("Algorithm", meta['best_model'])


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DETAILED ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Detailed Analysis":
    st.title("📈 Detailed Analysis")
    st.markdown("---")

    # ── Score distributions ──
    st.markdown('<p class="section-title">Score Distributions by Placement Status</p>', unsafe_allow_html=True)
    score_cols   = ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"]
    score_labels = ["10th %", "12th %", "Degree %", "E-Test %", "MBA %"]

    selected_score = st.selectbox("Select Score", score_labels)
    sel_col = score_cols[score_labels.index(selected_score)]

    fig = go.Figure()
    for status, color in [("Placed", PLACED_COLOR), ("Not Placed", UNPLACED_COLOR)]:
        fig.add_trace(go.Histogram(
            x=df[df["status"]==status][sel_col],
            name=status, opacity=0.7,
            marker_color=color, nbinsx=25
        ))
    fig.update_layout(
        barmode="overlay", height=350,
        xaxis_title=selected_score, yaxis_title="Count",
        legend=dict(orientation="h", y=1.05),
        margin=dict(t=20, b=20, l=20, r=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Salary analysis ──
    st.markdown('<p class="section-title">Salary Analysis — Placed Students</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            placed_df, x="salary", nbins=30,
            color_discrete_sequence=[PLACED_COLOR],
            labels={"salary": "Salary (₹)"},
        )
        fig.add_vline(
            x=placed_df["salary"].mean(), line_dash="dash",
            line_color="#333", annotation_text=f"Mean: ₹{placed_df['salary'].mean()/100000:.2f}L"
        )
        fig.update_layout(height=320, margin=dict(t=20, b=20, l=20, r=20),
                          yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sal_spec = placed_df.groupby("specialisation")["salary"].agg(["mean","median","count"]).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=sal_spec["specialisation"],
            y=(sal_spec["mean"]/100000).round(2),
            name="Mean", marker_color=PLACED_COLOR,
            text=(sal_spec["mean"]/100000).round(2).astype(str)+"L",
            textposition="outside"
        ))
        fig.add_trace(go.Bar(
            x=sal_spec["specialisation"],
            y=(sal_spec["median"]/100000).round(2),
            name="Median", marker_color="#7CBFB0",
            text=(sal_spec["median"]/100000).round(2).astype(str)+"L",
            textposition="outside"
        ))
        fig.update_layout(
            barmode="group", height=320,
            yaxis_title="Salary (LPA)", xaxis_title="MBA Specialisation",
            margin=dict(t=20, b=20, l=20, r=20),
            legend=dict(orientation="h", y=1.05)
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Scatter: Degree % vs Salary ──
    st.markdown('<p class="section-title">Degree % vs Salary Package</p>', unsafe_allow_html=True)
    fig = px.scatter(
        placed_df, x="degree_p", y="salary",
        color="specialisation", trendline="ols",
        color_discrete_sequence=[PLACED_COLOR, "#7CBFB0"],
        labels={"degree_p": "Degree Percentage (%)", "salary": "Salary (₹)"},
        opacity=0.7
    )
    fig.update_layout(height=380, margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)

    # ── HSC stream table ──
    st.markdown('<p class="section-title">Placement Rate by 12th Stream & Degree Type</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        hsc_ct = pd.crosstab(df["hsc_s"], df["status"])

        numeric_cols = hsc_ct.select_dtypes(include="number").columns
        hsc_ct["Total"] = hsc_ct[numeric_cols].sum(axis=1)

        hsc_ct["Placement Rate"] = (
            (hsc_ct.get("Placed", 0) / hsc_ct["Total"]) * 100
        ).round(1).astype(str) + "%"

        st.dataframe(hsc_ct, use_container_width=True)

    with col2:
        deg_ct = pd.crosstab(df["degree_t"], df["status"])

        deg_ct["Total"] = deg_ct.select_dtypes(include="number").sum(axis=1)

        deg_ct["Placement Rate"] = (
            (deg_ct.get("Placed", 0) / deg_ct["Total"]) * 100
        ).round(1)
        
        deg_ct["Placement Rate"] = deg_ct["Placement Rate"].astype(str) + "%"

        st.dataframe(deg_ct, use_container_width=True)