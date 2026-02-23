import pandas as pd
from custom_clipper import PercentileClipper

# ==========================================
# THE DATASET 
# ==========================================
df = pd.read_csv('data/creditcard.csv')
# Step 1: Scikit-Learn Imports
# Step 2: Define column lists
# Step 3: Numerical Pipeline
# Step 4: Categorical Pipeline
# Step 5: ColumnTransformer
# Step 6: fit_transform and print!

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import  StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split


X = df.drop(columns=['Class'])
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
numeric_steps = [('clipper', PercentileClipper(lower_percentile=0.05, upper_percentile=0.95)),
         ('scaler', StandardScaler())]

numeric_pipe = Pipeline(numeric_steps)

t = [('numerical_transformer', numeric_pipe, X_train.columns)]
transformer = ColumnTransformer(transformers=t)

master_steps = [('transformer', transformer),
                ('classifier', XGBClassifier(scale_pos_weight=580, random_state = 42))]
master_pipe = Pipeline(master_steps)

master_pipe.fit(X_train, y_train)

probab = master_pipe.predict_proba(X_test)[:, 1]
y_pred = (probab >= 0.25).astype(int)
print(y_pred)


from sklearn.metrics import classification_report, confusion_matrix

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
import joblib
joblib.dump(master_pipe, 'ml_pipeline.joblib')

