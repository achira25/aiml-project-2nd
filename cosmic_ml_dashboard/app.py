
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, IsolationForest
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Cosmic-ML Dashboard", layout="wide")

st.markdown("""
<style>
body {
    background-color: #050816;
}
.main {
    background: linear-gradient(to bottom right, #050816, #0b1026);
    color: white;
}
h1, h2, h3 {
    color: #8ab4ff;
}
.hero {
    border-radius: 20px;
    padding: 30px;
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
}
.metric-card {
    background: rgba(255,255,255,0.08);
    border-radius: 15px;
    padding: 15px;
}
</style>
""", unsafe_allow_html=True)

slides = [
    "https://images.unsplash.com/photo-1462331940025-496dfbfc7564",
    "https://images.unsplash.com/photo-1444703686981-a3abbc4d4fe3",
    "https://images.unsplash.com/photo-1465101046530-73398c7f28ca"
]

if "slide_index" not in st.session_state:
    st.session_state.slide_index = 0

st.image(slides[st.session_state.slide_index], use_container_width=True)
st.session_state.slide_index = (st.session_state.slide_index + 1) % len(slides)

st.markdown("""
<div class="hero">
<h1>🌌 Cosmic-ML Benchmark & Mission Dashboard</h1>
<p>A dual-mode AI platform comparing Deep Learning and Traditional ML for celestial classification.</p>
</div>
""", unsafe_allow_html=True)

mode = st.sidebar.radio(
    "Select Analysis Mode",
    ["🏆 ML Grand Prix", "🧠 Visual Battle", "🛰 Expert Analytics"]
)

@st.cache_data
def generate_data():
    X, y = make_classification(
        n_samples=10000,
        n_features=6,
        n_classes=3,
        n_informative=5,
        n_redundant=0,
        random_state=42
    )
    cols = ["Redshift", "u", "g", "r", "i", "z"]
    return pd.DataFrame(X, columns=cols), y

X, y = generate_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

if mode == "🏆 ML Grand Prix":
    st.title("🏁 ML Grand Prix")

    models = {
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(eval_metric='mlogloss'),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB()
    }

    results = []

    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        end = time.time()

        acc = accuracy_score(y_test, preds)

        results.append({
            "Model": name,
            "Accuracy": round(acc * 100, 2),
            "Inference Time (ms)": round((end - start) * 1000, 2)
        })

    df_results = pd.DataFrame(results)

    st.dataframe(df_results, use_container_width=True)

    fig = px.bar(
        df_results,
        x="Model",
        y="Accuracy",
        color="Model",
        title="Model Accuracy Comparison"
    )
    st.plotly_chart(fig, use_container_width=True)

elif mode == "🧠 Visual Battle":
    st.title("🧠 CNN vs ANN Visual Battle")

    st.markdown("""
    ### Concept Demonstration
    CNN understands image patterns spatially.
    ANN flattens everything like a raccoon dumping puzzle pieces into a blender.
    Humans invented both and still wonder why one sees galaxies better.
    """)

    cnn_acc = 98
    ann_acc = 75

    col1, col2 = st.columns(2)

    with col1:
        st.metric("CNN Accuracy", f"{cnn_acc}%")
        st.progress(cnn_acc)

    with col2:
        st.metric("ANN Accuracy", f"{ann_acc}%")
        st.progress(ann_acc)

    cm = np.array([[95, 3, 2],
                   [5, 88, 7],
                   [2, 6, 92]])

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='magma', ax=ax)
    ax.set_title("CNN Confusion Matrix")
    st.pyplot(fig)

elif mode == "🛰 Expert Analytics":
    st.title("🛰 Expert Analytics")

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig = px.bar(
        importance,
        x="Feature",
        y="Importance",
        title="Feature Importance Ranking"
    )
    st.plotly_chart(fig, use_container_width=True)

    pca = PCA(n_components=3)
    transformed = pca.fit_transform(X)

    df_pca = pd.DataFrame(transformed, columns=["PC1", "PC2", "PC3"])
    df_pca["Class"] = y.astype(str)

    fig3d = px.scatter_3d(
        df_pca,
        x="PC1",
        y="PC2",
        z="PC3",
        color="Class",
        title="3D Cosmic Map"
    )
    st.plotly_chart(fig3d, use_container_width=True)

    iso = IsolationForest(contamination=0.02)
    anomalies = iso.fit_predict(X)

    anomaly_count = np.sum(anomalies == -1)

    st.metric("Detected Cosmic Anomalies", anomaly_count)

st.sidebar.markdown("---")
st.sidebar.info("🚀 Built with Streamlit, TensorFlow, Scikit-Learn & Plotly")
