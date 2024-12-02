import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# 1. 데이터 로드 및 전처리
iris = load_iris()
X = iris.data
y = iris.target

# 이진 분류 문제로 변환 (클래스 0과 1만 사용)
binary_indices = y < 2
X_binary = X[binary_indices]
y_binary = y[binary_indices]

# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X_binary, y_binary, test_size=0.2, random_state=42)

# 데이터 스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. 모델 학습
model = LogisticRegression()
model.fit(X_train, y_train)

# 3. 모델 저장
joblib.dump(model, "iris_logistic_model.pkl")
print("Model saved as iris_logistic_model.pkl")

# 4. 검증 데이터 저장 (CSV 파일로 저장)
test_data = pd.DataFrame(X_test, columns=["feature1", "feature2", "feature3", "feature4"])
test_data["label"] = y_test
test_data.to_csv("iris_test_data.csv", index=False)
print("Test data saved as iris_test_data.csv")

# 5. 모델 평가
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")
