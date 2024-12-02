import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Streamlit 앱 제목
st.title("이진 분류 모델 검증 서비스")

# 1. 모델 업로드
st.sidebar.header("1. 모델 업로드")
model_file = st.sidebar.file_uploader("학습된 모델 업로드 (.pkl)", type=["pkl"])

# 2. 검증 데이터 업로드
st.sidebar.header("2. 검증 데이터 업로드")
data_file = st.sidebar.file_uploader("검증 데이터 업로드 (.csv)", type=["csv"])

if model_file and data_file:
    # 3. 모델 로드
    model = joblib.load(model_file)
    st.success("모델이 성공적으로 로드되었습니다!")

    # 4. 검증 데이터 로드
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

    # 5. 모델 검증
    st.subheader("모델 검증 결과")
    y_pred = model.predict(X_test)  # 모델 예측 수행

    # 정확도 계산
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"**정확도 (Accuracy):** {accuracy:.2f}")

    # 혼동 행렬 시각화
    cm = confusion_matrix(y_test, y_pred)
    st.write("**혼동 행렬 (Confusion Matrix):**")
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    st.pyplot(fig)

    # 결과 요약
    st.write("**분류 보고서:**")
    st.json({
        "True Positives": int(cm[1, 1]),
        "True Negatives": int(cm[0, 0]),
        "False Positives": int(cm[0, 1]),
        "False Negatives": int(cm[1, 0]),
    })

else:
    st.info("모델과 검증 데이터를 모두 업로드해야 검증을 진행할 수 있습니다.")
