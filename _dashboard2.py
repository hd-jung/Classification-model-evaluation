import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
)
import matplotlib.pyplot as plt
import numpy as np


# Streamlit 앱 제목
st.title("이진 분류 모델 성능 평가 서비스")

# 1. 모델 업로드
st.sidebar.header("1. 모델 업로드")
model_file = st.sidebar.file_uploader("학습된 모델 업로드 (.pkl)", type=["pkl"])

# 2. 검증 데이터 업로드
st.sidebar.header("2. 검증 데이터 업로드")
data_file = st.sidebar.file_uploader("검증 데이터 업로드 (.csv)", type=["csv"])

# 3. F-beta 스코어의 Beta 값 입력
st.sidebar.header("3. F-beta 스코어 Beta 값 설정")
beta_value = st.sidebar.number_input("Beta 값 입력 (\u03B2):", min_value=0.1, value=1.0, step=0.1)

if model_file and data_file:
    # 4. 모델 로드
    model = joblib.load(model_file)
    st.success("모델이 성공적으로 로드되었습니다!")

    # 5. 검증 데이터 로드
    data = pd.read_csv(data_file)
    st.subheader("검증 데이터")
    st.write(data.head())

    # 데이터 전처리
    if "label" in data.columns:
        X_test = data.drop(columns=["label"])  # 입력 데이터
        y_test = data["label"]  # 실제 레이블
    else:
        st.error("검증 데이터셋에 'label' 열이 포함되어야 합니다.")
        st.stop()

    # 6. 모델 검증
    st.subheader("모델 검증 결과")
    y_pred = model.predict(X_test)  # 모델 예측 수행
    y_prob = model.predict_proba(X_test)[:, 1]  # 양성 클래스 확률

    # 주요 성능 지표 계산
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    specificity = recall_score(y_test, y_pred, pos_label=0)
    f1 = f1_score(y_test, y_pred)
    f_beta = fbeta_score(y_test, y_pred, beta=beta_value)

    st.write(f"**정확도 (Accuracy):** {accuracy:.2f}")
    st.write(f"**정밀도 (Precision):** {precision:.2f}")
    st.write(f"**재현율 (Recall):** {recall:.2f}")
    st.write(f"**특이도 (Specificity):** {specificity:.2f}")
    st.write(f"**F1 스코어:** {f1:.2f}")
    st.write(f"**F{beta_value:.1f} 스코어:** {f_beta:.2f}")

    # 혼동 행렬 시각화
    cm = confusion_matrix(y_test, y_pred)
    st.write("**혼동 행렬 (Confusion Matrix):**")
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    st.pyplot(fig)

    # ROC Curve 및 AUROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    st.write(f"**AUROC (Area Under the ROC Curve):** {roc_auc:.2f}")
    fig, ax = plt.subplots()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    st.pyplot(fig)

    # Precision-Recall Curve 및 AUPRC
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    auprc = auc(recall_vals, precision_vals)
    st.write(f"**AUPRC (Area Under the Precision-Recall Curve):** {auprc:.2f}")
    fig, ax = plt.subplots()
    plt.plot(recall_vals, precision_vals, color="blue", lw=2, label=f"PR curve (area = {auprc:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    st.pyplot(fig)

else:
    st.info("모델과 검증 데이터를 모두 업로드해야 검증을 진행할 수 있습니다.")
