# 📦 Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ✅ Step 1: Load data
df = pd.read_csv('laptop_prices.csv')

# ✅ Step 2: Preprocessing
df_model = df.copy()

# Encode 'Yes'/'No' to 1/0
binary_cols = ['Touchscreen', 'IPSpanel', 'RetinaDisplay']
for col in binary_cols:
    df_model[col] = df_model[col].map({'Yes': 1, 'No': 0})

# New feature: Screen area
df_model['ScreenArea'] = df_model['ScreenW'] * df_model['ScreenH']

# New feature: Total storage
df_model['TotalStorage'] = df_model['PrimaryStorage'] + df_model['SecondaryStorage']

# Drop less useful or highly unique columns
df_model = df_model.drop(columns=[
    'Product', 'GPU_model', 'CPU_model', 'Screen', 'ScreenW', 'ScreenH'
])

# One-hot encode categorical columns
categorical_cols = ['Company', 'TypeName', 'OS', 'PrimaryStorageType', 
                    'SecondaryStorageType', 'CPU_company', 'GPU_company']
df_model = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)

# ✅ Step 3: Convert target prices to INR
exchange_rate = 90  # 1 euro = 90 INR (adjust as needed)
df_model['Price_inr'] = df_model['Price_euros'] * exchange_rate

# ✅ Step 4: Split data
X = df_model.drop(['Price_euros', 'Price_inr'], axis=1)
y = df_model['Price_inr']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ Step 5: Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Step 6: Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"✅ RMSE: ₹{rmse:,.0f}")
print(f"✅ R² Score: {r2:.2f}")