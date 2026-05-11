import pandas as pd
import pickle

with open('model.pkl', 'rb') as f:
    placement_model = pickle.load(f)
with open('model1.pkl', 'rb') as f:
    salary_model = pickle.load(f)

df = pd.read_csv('Placement_Prediction_data.csv')

df['Internship'] = df['Internship'].map({'Yes': 1, 'No': 0})
df['Hackathon'] = df['Hackathon'].map({'Yes': 1, 'No': 0})

for col in [
    'CGPA', 'Major Projects', 'Workshops/Certificatios', 'Mini Projects',
    'Skills', 'Communication Skill Rating', '12th Percentage', '10th Percentage', 'backlogs'
]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

placement_features = [
    'CGPA', 'Major Projects', 'Workshops/Certificatios', 'Mini Projects', 'Skills',
    'Communication Skill Rating', 'Internship', 'Hackathon',
    '12th Percentage', '10th Percentage', 'backlogs'
]

df['placement_pred'] = placement_model.predict(df[placement_features])

df['PlacementStatus'] = (df['placement_pred'] == 'Placed').astype(int)
salary_features = placement_features + ['PlacementStatus']
df['salary_pred'] = salary_model.predict(df[salary_features])

df.to_csv('prediction_results.csv', index=False)
print('Predictions saved to prediction_results.csv')
print(df[['StudentId', 'placement_pred', 'salary_pred']].head())
