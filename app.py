import numpy as np
from flask import Flask, request, render_template
import pickle

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('salary_model.pkl', 'rb') as f2:
    salary_model = pickle.load(f2)

app = Flask(__name__, template_folder="templates")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    name = request.args.get('name', 'Student')
    cgpa = float(request.args.get('cgpa', 0))
    projects = int(request.args.get('projects', 0))
    workshops = int(request.args.get('workshops', 0))
    mini_projects = int(request.args.get('mini_projects', 0))
    skills = request.args.get('skills', '')
    communication_skills = float(request.args.get('communication_skills', 0))
    internship = request.args.get('internship', 'No')
    hackathon = request.args.get('hackathon', 'No')
    tw_percentage = float(request.args.get('tw_percentage', 0))
    te_percentage = float(request.args.get('te_percentage', 0))
    backlogs = int(request.args.get('backlogs', 0))

    internship = 1 if internship.lower() in ['yes', 'y'] else 0
    hackathon = 1 if hackathon.lower() in ['yes', 'y'] else 0

    skill_count = len([x.strip() for x in skills.split(',') if x.strip() != ''])

    features = np.array([
        cgpa, projects, workshops, mini_projects, skill_count,
        communication_skills, internship, hackathon,
        tw_percentage, te_percentage, backlogs
    ], dtype=float)

    output_num = model.predict([features])[0]  
    output_label = 'Placed' if output_num == 1 else 'Not Placed'

    salary_features = np.append(features, 1 if output_num == 1 else 0)
    salary = int(salary_model.predict([salary_features])[0])
    salary_str = f"{salary:,}"

    if output_label == 'Placed':
        output_text = f"Congratulations {name}!! You have high chances of getting placed!"
        output2 = f"Your expected salary will be INR {salary_str} per annum"
    else:
        output_text = f"Sorry {name}!! You have low chances of getting placed."
        output2 = "Improve your skills."

    return render_template('output.html', output=output_text, output2=output2)

if __name__ == "__main__":
    app.run(debug=True)
