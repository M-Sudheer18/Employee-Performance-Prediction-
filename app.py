import numpy as np
from flask import Flask, render_template, request
import pickle


app = Flask(__name__)

# Loading Model
model = pickle.load(open("model_rfg.pkl", "rb"))

# Encoding 
quarter_map = {
    "Quarter1": 0,
    "Quarter2": 1,
    "Quarter3": 2,
    "Quarter4": 3,
}
department_map = {
    "sweing": 0,
    "finishing": 1
}
day_map = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

@app.route('/', methods = ["GET"])
def home():
    return render_template('home.html')

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":

        # Collect form data
        quarter = quarter_map[request.form["quarter"]]
        department = department_map[request.form["department"].lower()]
        day = day_map[request.form["day"]]


        team = int(request.form["team"])
        targeted_productivity = float(request.form["targeted_productivity"])
        smv = float(request.form["smv"])
        over_time = float(request.form["over_time"])
        incentive = float(request.form["incentive"])
        idle_time = float(request.form["idle_time"])
        idle_men = int(request.form["idle_men"])
        no_of_style_change = int(request.form["no_of_style_change"])
        no_of_workers = float(request.form["no_of_workers"])
        year = int(request.form["year"])
        month = int(request.form["month"])
        day_num = int(request.form["day_num"])


        
        # Model Integration input_data
        features = np.array([[
            quarter,
            department,
            day,
            team,
            targeted_productivity,
            smv,
            over_time,
            incentive,
            idle_time,
            idle_men,
            no_of_style_change,
            no_of_workers,
            year,
            month,
            day_num
        ]])

        
        # Model Integration
        predicted_productivity = model.predict(features)[0]

        # Performance category
        if predicted_productivity >= 0.8:
            level = "High Performer"
        elif predicted_productivity >= 0.6:
            level = "Average Performer"
        else:
            level = "Low Performer"

        return render_template(
            "result.html",
            prediction=round(predicted_productivity, 3),
            performance_level=level
        )

    return render_template("predict.html")

if __name__ =="__main__":
    app.run()


