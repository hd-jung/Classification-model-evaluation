import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
def evaluate_model(model_path, test_data_path):
    # 1. 학습된 모델 로드
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")

    # 2. 검증 데이터 로드
    test_data = pd.read_csv(test_data_path)
    print(f"Test data loaded from {test_data_path}")

    # 피처와 라벨 분리
    X_test = test_data.drop(columns=["label"])
    y_test = test_data["label"]

    # 3. 모델 예측
    y_pred = model.predict(X_test)

    # 정확도
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
# 실행
model_path = "iris_logistic_model.pkl"  # 학습된 모델 파일 경로
test_data_path = "iris_test_data.csv"  # 검증 데이터 파일 경로
evaluate_model(model_path, test_data_path)