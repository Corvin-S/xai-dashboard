import numpy as np, pandas as pd, shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
col_names = ['checking_account','duration','credit_history','purpose','credit_amount','savings_account','employment','installment_rate','personal_status','other_debtors','residence_since','property','age','other_installments','housing','existing_credits','job','liable_people','telephone','foreign_worker','target']
df = pd.read_csv('../data/german.data', sep=' ', header=None, names=col_names)
df['target'] = df['target'].map({1:0, 2:1})
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])
X = df.drop(columns=['target']); y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
rf = RandomForestClassifier(n_estimators=100, random_state=42); rf.fit(X_train, y_train)
sv = shap.TreeExplainer(rf).shap_values(X_test)
np.save('shap_values.npy', sv)
print('Done — shape:', sv.shape)
