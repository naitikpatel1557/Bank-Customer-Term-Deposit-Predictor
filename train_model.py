#! C:\Users\Asus\Bank_Cust_Deposit\venv\Scripts\python.exe

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score # <-- Added this import
import joblib

# 1. Load Dataset
df = pd.read_csv('Data/bank-additional-full.csv')

# 2. Preprocess Target and Drop Data Leakage
df['deposit'] = df['deposit'].map({'yes': 1, 'no': 0})
X = df.drop(['deposit', 'duration'], axis=1)
y = df['deposit']

# 3. Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# 4. Create Preprocessing & Modeling Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ], remainder='passthrough')

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

# 5. Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# --- NEW: Evaluate the Model ---
# Ask the model to predict outcomes for the test data it hasn't seen before
y_pred = model.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
# -------------------------------

# 6. Save Pipeline
joblib.dump(model, 'bank_model.pkl')
print("Model trained and saved successfully as bank_model.pkl!")











# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from xgboost import XGBClassifier
# import joblib

# # 1. Load Dataset (Using standard comma separation)
# df = pd.read_csv('Data/bank-additional-full.csv')

# # 2. Preprocess Target and Drop Data Leakage
# # Your dataset uses 'deposit' instead of 'y' for the target variable
# df['deposit'] = df['deposit'].map({'yes': 1, 'no': 0})

# # Drop 'duration' because it is unknown before a call is made
# X = df.drop(['deposit', 'duration'], axis=1)
# y = df['deposit']

# # 3. Identify categorical columns
# categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# # 4. Create Preprocessing & Modeling Pipeline
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
#     ], remainder='passthrough')

# model = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
# ])

# # 5. Train Model
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model.fit(X_train, y_train)

# # 6. Save Pipeline
# joblib.dump(model, 'bank_model.pkl')
# print("Model trained and saved successfully as bank_model.pkl!")