import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from casadi import SX, vertcat, nlpsol

# 예제 데이터
data = pd.DataFrame({
    'X1': [1.60, 1.80, 1.60, 1.80, 1.80, 1.40, 1.60, 1.60, 1.40, 1.80, 1.40, 1.40, 1.264, 1.60],
    'X2': [167, 100, 83, 150, 150, 150, 125, 125, 100, 100, 150, 100, 125, 125],
    'X3': [40, 20, 40, 60, 20, 20, 40, 6.36, 20, 60, 60, 60, 40, 73.6],
    'Y': [11.76, 11.98, 12.75, 14.41, 15.65, 16.78, 17.44, 17.86, 18.62, 18.79, 19.21, 19.22, 19.45, 20.29],
})

# 독립 변수의 범위 설정
X1_min, X1_max = data['X1'].min(), data['X1'].max()
X2_min, X2_max = data['X2'].min(), data['X2'].max()
X3_min, X3_max = data['X3'].min(), data['X3'].max()
# 반응표면 모델 정의
import statsmodels.formula.api as smf
reduced_model = smf.ols('Y ~ X1 + X2 + X3 + I(X1**2) + I(X2**2) + I(X3**2) + X1:X2 + X2:X3 + X1:X3', data=data).fit()

# CasADi를 사용한 최적화
X = SX.sym('X', 2)  # X1, X2 (X3는 고정값 15)

# X3를 15로 고정
X3_fixed = 15

# 반응표면 회귀 모델 식 정의
model_expr = (
    reduced_model.params['Intercept']
    + reduced_model.params['X1'] * X[0]
    + reduced_model.params['X2'] * X[1]
    + reduced_model.params['X3'] * X3_fixed
    + reduced_model.params['I(X1 ** 2)'] * X[0]**2
    + reduced_model.params['I(X2 ** 2)'] * X[1]**2
    + reduced_model.params['I(X3 ** 2)'] * X3_fixed**2
    + reduced_model.params['X1:X2'] * X[0] * X[1]
    + reduced_model.params['X1:X3'] * X[0] * X3_fixed
    + reduced_model.params['X2:X3'] * X[1] * X3_fixed
)

# 목표 PCE 값 설정
target_Y = 20

# 목적 함수: 목표 PCE 값과의 차이를 최소화
obj = (model_expr - target_Y) ** 2

# 경계값 설정 (X1, X2만 최적화)
lbx = [X1_min, X2_min]  # 하한
ubx = [X1_max, X2_max]  # 상한

# NLP 정의
nlp = {'x': X, 'f': obj}
solver = nlpsol('solver', 'ipopt', nlp)

# 초기 추정값
x0 = [data['X1'].mean(), data['X2'].mean()]

# 최적화 실행
solution = solver(x0=x0, lbx=lbx, ubx=ubx)

# 최적화된 X1, X2 값
optimal_X = np.array(solution['x']).flatten()

# 입력 데이터를 DataFrame으로 생성
input_data = pd.DataFrame({
    'X1': [optimal_X[0]],
    'X2': [optimal_X[1]],
    'X3': [X3_fixed]  # 고정된 X3 값
})

# 예측값 계산
optimal_Y = reduced_model.predict(input_data)[0]

# 결과 출력
print(f"🔹 X3 = {X3_fixed}로 고정한 상태에서 최적화 결과")
print("----------------------------------------------------")
print(f"   최적화된 Y 값 (PCE) = {optimal_Y:.4f}")
print(f"   최적 X1 = {optimal_X[0]:.4f}, X2 = {optimal_X[1]:.4f}, (X3 = {X3_fixed} 고정)")
print("----------------------------------------------------")
