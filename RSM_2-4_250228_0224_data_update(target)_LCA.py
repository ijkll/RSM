import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pygmo as pg
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from casadi import SX, vertcat, nlpsol


data = pd.DataFrame({
    'X1': [1.60, 1.80, 1.60, 1.80, 1.80, 1.40, 1.60, 1.60, 1.40, 1.80, 1.40, 1.40, 1.264, 1.60],  # 독립 변수, 농도
    'X2': [167, 100, 83, 150, 150, 150, 125, 125, 100, 100, 150, 100, 125, 125],  # 독립 변수 2, 온도
    'X3': [40, 20, 40, 60, 20, 20, 40, 6.36, 20, 60, 60, 60, 40, 73.6 ],  # 독립 변수 3, 시간
    'Y': [11.76, 11.98, 12.75, 14.41, 15.65, 16.78, 17.44, 17.86, 18.62, 18.79, 19.21, 19.22, 19.45, 20.29 ],   # 종속 변수
})

1.4-1.8
100-150
20-60


print(data.head())

import statsmodels.formula.api as smf


reduced_model = smf.ols('Y ~ X1 + X2 + X3 + I(X1**2) + I(X2**2) + I(X3**2) + X1:X2 + X2:X3 + X1:X3', data=data).fit()


print(reduced_model.summary())

new_data = pd.DataFrame({
    'X1': [0.5, -0.5],
    'X2': [0.3, -0.3],
    'X3': [0.7, -0.7],
})

# 예측값 계산
new_data['predicted_Y'] = reduced_model.predict(new_data)
print(new_data)


import matplotlib.pyplot as plt


# IQR(Interquartile Range)
Q1 = data['Y'].quantile(0.25)
Q3 = data['Y'].quantile(0.75)
IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers = data[(data['Y'] < lower_limit) | (data['Y'] > upper_limit)]
print("Outliers:")
print(outliers)


from statsmodels.stats.outliers_influence import variance_inflation_factor

# VIF 계산
variables = reduced_model.model.exog  # 독립 변수 행렬
vif = pd.DataFrame({
    'Variable': reduced_model.model.exog_names,
    'VIF': [variance_inflation_factor(variables, i) for i in range(variables.shape[1])]
})
print(vif)

# ANOVA TEST
anova_table = sm.stats.anova_lm(reduced_model, typ=2)
print(anova_table)

from statsmodels.stats.diagnostic import linear_harvey_collier

# X1, X2 값의 그리드 생성 (데이터 범위에 맞게 제한)
X1_min, X1_max = data['X1'].min(), data['X1'].max()
X2_min, X2_max = data['X2'].min(), data['X2'].max()
X3_min, X3_max = data['X3'].min(), data['X3'].max()


X1 = np.linspace(X1_min, X1_max, 50)
X2 = np.linspace(X2_min, X2_max, 50)
X3= np.linspace(X3_min, X3_max, 50)

X1, X2 = np.meshgrid(X1, X2)

# X3는 평균값으로 고정
X3_mean = data['X3'].mean()

# 모델에 입력할 데이터프레임 생성
grid_data = pd.DataFrame({ 
    'X1': X1.ravel(),
    'X2': X2.ravel(),
    'X3': 40  # X3 값을 평균값으로 고정
})

# 예측 데이터의 범위를 데이터에 맞게 필터링
grid_data = grid_data[
    (grid_data['X1'] >= X1_min) & (grid_data['X1'] <= X1_max) &
    (grid_data['X2'] >= X2_min) & (grid_data['X2'] <= X2_max)
]

# 예측값 계산
grid_data['Y'] = reduced_model.predict(grid_data)
Y = grid_data['Y'].values.reshape((len(np.unique(grid_data['X1'])), len(np.unique(grid_data['X2']))))

# 반응 표면 그리기
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X1, X2, Y, cmap='viridis', alpha=0.8)

# 원본 데이터 산점도 추가
ax.scatter(data['X1'], data['X2'], data['Y'], color='red', label='Data Points')

# 축 설정
ax.set_xlabel('Concentration (mol/L)')
ax.set_ylabel('Temperature (oC)')
ax.set_zlabel('PCE (%)')
ax.set_title('3D Response Surface')

# 범례 추가
ax.legend()

plt.show()

# 2D 등고선 그래프 그리기
plt.figure(figsize=(8, 6))
contour = plt.contourf(X1, X2, Y, levels=50, cmap='viridis')  # 등고선 채우기
plt.colorbar(contour, label='Y Value')  # 색상바 추가

# 데이터 점 추가
plt.scatter(data['X1'], data['X2'], c='red', label='Data Points')

# 축 설정
plt.xlabel('Concentration (mol/L)')
plt.ylabel('Temperature (oC)')
plt.title('2D Contour Plot')
plt.legend()

plt.show()

residuals = reduced_model.resid
rmse = np.sqrt(np.mean(residuals**2))
print(f"RMSE (Root Mean Square Error): {rmse:.4f}")


X1_min, X1_max = data['X1'].min(), data['X1'].max()
X2_min, X2_max = data['X2'].min(), data['X2'].max()
X3_min, X3_max = data['X3'].min(), data['X3'].max()


X1 = np.linspace(X1_min, X1_max, 50)
X2 = np.linspace(X2_min, X2_max, 50)
X3= np.linspace(X3_min, X3_max, 50)
X2, X3 = np.meshgrid(X2, X3)  # 2D 그리드 생성

# X3 값 고정
X1_fixed = 1.4  # X3를 고정된 값으로 설정

# Cost 함수 정의
def CO2_function(X1, X2, X3):
    return (6.556 * X1 + 0.648 * ((85.731 * X2) - 0.6907) * X3 / 60 + 16.9 * 10**3 + 32.55 * 10**3) / \
            (20 * 0.01 * 0.7 * 1510.44 * 0.84 * 1 * 0.9)

# Cost 값 계산
Y = CO2_function(X1_fixed, X2, X3)

# 데이터 점의 Y 값 계산
data['Calculated_Y'] = CO2_function(data['X1'], data['X2'], data['X3'])  # 각 행의 Y 값 계산

# 3D 반응 표면 그리기
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 두 점을 추가
additional_points = pd.DataFrame({
    'X1': [1.51, 1.39, 1.28],
    'X2': [125, 122, 124],
    'X3': [40, 38, 36],
    'Y': [338.39, 337.69, 335.28]
})

# 3D 표면 플롯
surf = ax.plot_surface(X2, X3, Y, cmap='jet', alpha=0.8)

# 축 설정
ax.set_xlabel('Temperature (°C)', fontsize=12)
ax.set_ylabel('Time (min)', fontsize=12)
ax.set_zlabel('Climate change (g CO2-Eq/kWh)', fontsize=12)
ax.set_title('3D Response Surface of Cost Function', fontsize=14)
ax.scatter(data['X2'], data['X3'], data['Calculated_Y'], color='red', label='Data Points')
# 추가된 점
ax.scatter(additional_points['X2'], additional_points['X3'], additional_points['Y'], color='blue', label='Additional Points')
# 색상 바 추가


plt.show()

# 2D 등고선 그래프 그리기
plt.figure(figsize=(8, 6))

# 등고선 플롯
contour = plt.contourf(X2, X3, Y, levels=50, cmap='jet')

# 색상바 추가
plt.colorbar(contour, label='Cost')
plt.scatter(data['X2'], data['X3'], color='red', label='cost')

# 추가된 점
plt.scatter(additional_points['X2'], additional_points['X3'], c='blue', label='Additional Points')
# 축 설정
plt.xlabel('Concentration (mol/L)', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.title('2D Contour Plot of Cost Function', fontsize=14)

plt.show()
