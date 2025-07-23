import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.metrics import (accuracy_score, precision_score,
                           recall_score, f1_score, confusion_matrix,
                           roc_auc_score, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit konfiqurasiyası
st.set_page_config(
    page_title="Kredit Risk Analiz Tətbiqi",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- Məlumatların yüklənməsi ---
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("GiveMeSomeCredit-training.csv")
        # Lazımsız sütunları sil
        if 'Unnamed: 0' in data.columns:
            data.drop('Unnamed: 0', axis=1, inplace=True)
        return data
    except Exception as e:
        st.error(f"Məlumat yüklənərkən xəta baş verdi: {str(e)}")
        return None


# --- Modellərin təlimi ---
@st.cache_resource
def train_models(data):
    # Hədəf dəyişən və xüsusiyyətlər
    X = data.drop('SeriousDlqin2yrs', axis=1)
    y = data['SeriousDlqin2yrs']

    # Əsas xüsusiyyətlər
    features = [
        'RevolvingUtilizationOfUnsecuredLines',
        'age',
        'NumberOfTime30-59DaysPastDueNotWorse',
        'DebtRatio',
        'MonthlyIncome',
        'NumberOfOpenCreditLinesAndLoans',
        'NumberOfTimes90DaysLate',
        'NumberRealEstateLoansOrLines',
        'NumberOfTime60-89DaysPastDueNotWorse',
        'NumberOfDependents'
    ]
    X = X[features]

    # Data preprocessing
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Modellər
    models = {
        "Lojistik Reqressiya": LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=1000
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight='balanced'
        ),
        "LightGBM": lgb.LGBMClassifier(
            random_state=42,
            class_weight='balanced',
            n_estimators=150
        )
    }

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Model təlimi və qiymətləndirmə
    results = {}
    pipelines = {}

    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        pipelines[name] = pipeline
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }

    return pipelines, results, features


# --- Azərbaycan dilində tərcümələr ---
feature_translations = {
    'RevolvingUtilizationOfUnsecuredLines': 'Kredit Limitindən İstifadə Dərəcəsi',
    'age': 'Yaş',
    'NumberOfTime30-59DaysPastDueNotWorse': '30-59 Gün Gecikmə Sayı',
    'DebtRatio': 'Borç/Gəlir Nisbəti',
    'MonthlyIncome': 'Aylıq Gəlir',
    'NumberOfOpenCreditLinesAndLoans': 'Açıq Kredit Sayı',
    'NumberOfTimes90DaysLate': '90+ Gün Gecikmə Sayı',
    'NumberRealEstateLoansOrLines': 'İpoteka Kreditləri',
    'NumberOfTime60-89DaysPastDueNotWorse': '60-89 Gün Gecikmə Sayı',
    'NumberOfDependents': 'Asılı Şəxslər'
}


# --- Əsas tətbiq hissəsi ---
def main():
    st.title("💳 GiveMeSomeCredit Kredit Risk Analizi")
    st.markdown("""
    Bu tətbiq **GiveMeSomeCredit** məlumat dəsti əsasında müştərilərin kredit ödəniş riskini qiymətləndirir.
    """)

    # Məlumatların yüklənməsi
    data = load_data()

    if data is not None:
        # Modellərin təlimi
        pipelines, results, features = train_models(data)

        # --- Səhifə konfiqurasiyası ---
        tab1, tab2, tab3 = st.tabs(["📊 Analiz", "🔍 Proqnoz", "ℹ️ Haqqında"])

        with tab1:
            st.header("Model Performans Metrikaları")

            # Ümumi metrikalar
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Ümumi Müştəri Sayı", len(data))
            with col2:
                st.metric("Riskli Müştərilər",
                          f"{data['SeriousDlqin2yrs'].sum()} ({data['SeriousDlqin2yrs'].mean():.1%})")
            with col3:
                st.metric("Xüsusiyyət Sayı", len(features))

            # Model müqayisəsi
            st.subheader("Modellərin Müqayisəsi")
            model_names = list(results.keys())
            selected_model = st.selectbox("Model seçin", model_names)

            if selected_model:
                st.write(f"### {selected_model} Performansı")
                metrics = results[selected_model]

                cols = st.columns(4)
                cols[0].metric("Accuracy", f"{metrics['accuracy']:.2%}")
                cols[1].metric("Precision", f"{metrics['precision']:.2%}")
                cols[2].metric("Recall", f"{metrics['recall']:.2%}")
                cols[3].metric("F1 Score", f"{metrics['f1']:.2%}")

                st.metric("ROC AUC Score", f"{metrics['roc_auc']:.2%}")

                # Confusion matrix
                st.write("#### Qarışıqlıq Matrisi")
                fig, ax = plt.subplots()
                sns.heatmap(metrics['confusion_matrix'],
                            annot=True, fmt='d',
                            cmap='Blues',
                            xticklabels=['Etibarlı', 'Riskli'],
                            yticklabels=['Etibarlı', 'Riskli'])
                ax.set_xlabel('Proqnoz')
                ax.set_ylabel('Həqiqi')
                st.pyplot(fig)

                # Classification report
                st.write("#### Sinifləndirmə Hesabatı")
                report_df = pd.DataFrame(metrics['classification_report']).transpose()
                st.dataframe(report_df.style.format("{:.2f}"))

        with tab2:
            st.header("Yeni Müştəri üçün Proqnoz")
            st.warning("Zəhmət olmasa bütün sahələri doldurun")

            # İnput form
            input_data = {}
            cols = st.columns(2)

            for i, feature in enumerate(features):
                with cols[i % 2]:
                    translated = feature_translations.get(feature, feature)
                    if feature == 'age':
                        input_data[feature] = st.number_input(
                            translated, min_value=18, max_value=100, value=45
                        )
                    elif feature == 'MonthlyIncome':
                        input_data[feature] = st.number_input(
                            translated, min_value=0, value=5000, step=100
                        )
                    else:
                        input_data[feature] = st.number_input(
                            translated, value=float(data[feature].median())
                        )

            # Model seçimi
            selected_model = st.selectbox(
                "Proqnoz üçün model seçin",
                list(pipelines.keys()),
                key='model_select'
            )

            if st.button("Risk Proqnozunu Hesabla", type="primary"):
                with st.spinner("Hesablanır..."):
                    try:
                        input_df = pd.DataFrame([input_data])
                        pipeline = pipelines[selected_model]
                        prediction = pipeline.predict(input_df)[0]
                        proba = pipeline.predict_proba(input_df)[0][1]

                        if prediction == 1:
                            st.error(f"🚨 **Riskli Müştəri** (Ehtimal: {proba:.1%})")
                            st.warning("Bu müştəri yüksək risk qrupuna aiddir. Ətraflı yoxlama tövsiyə olunur.")
                        else:
                            st.success(f"✅ **Etibarlı Müştəri** (Ehtimal: {proba:.1%})")
                            st.info("Bu müştəri aşağı risk qrupuna aiddir.")

                        # Probability distribution
                        st.write("### Risk Ehtimalının Paylanması")
                        fig, ax = plt.subplots()
                        ax.bar(['Etibarlı', 'Riskli'],
                               [1 - proba, proba],
                               color=['green', 'red'])
                        ax.set_ylim(0, 1)
                        ax.set_ylabel('Ehtimal')
                        st.pyplot(fig)

                    except Exception as e:
                        st.error(f"Xəta baş verdi: {str(e)}")

        with tab3:
            st.header("Haqqında")
            st.markdown("""
            **GiveMeSomeCredit Kredit Risk Analiz Tətbiqi**

            Bu tətbiq aşağıdakı funksiyaları təqdim edir:
            - 3 fərqli maşın öyrənmə modeli ilə kredit riski proqnozu
            - Model performansının hərtərəfli qiymətləndirilməsi
            - İnteraktiv proqnozlaşdırma paneli

            **İstifadə olunan texnologiyalar:**
            - Python
            - Scikit-learn
            - LightGBM
            - Streamlit

            **Məlumat dəsti:** [GiveMeSomeCredit](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
            """)

            st.write("### Məlumat Dəsti Statistikası")
            st.dataframe(data.describe().style.format("{:.2f}"))

            st.write("### Hədəf Dəyişənin Paylanması")
            fig, ax = plt.subplots()
            data['SeriousDlqin2yrs'].value_counts().plot(
                kind='bar',
                color=['green', 'red'],
                ax=ax
            )
            ax.set_xticklabels(['Etibarlı', 'Riskli'], rotation=0)
            st.pyplot(fig)


if __name__ == "__main__":
    main()