# Employee-Performance-Prediction-
Employee Productivity Prediction system using Machine Learning. The project analyzes operational and workforce data to predict productivity levels. Built with Python, Random Forest, XGBoost, and Flask, and deployed as a web application for real-time predictions.
This project predicts employee productivity in a manufacturing environment using machine learning techniques. It analyzes operational and workforce-related factors to estimate productivity levels and classify employee performance.

📌 Project Overview
Employee productivity plays a crucial role in organizational efficiency. This project uses historical production data to build machine learning models that predict actual productivity based on factors such as targeted productivity, SMV, overtime, incentives, idle time, team size, and work schedule.
A Flask-based web application is integrated to provide real-time predictions through a simple and user-friendly interface.

🚀 Features
Predicts employee productivity using machine learning
Implements Linear Regression, Random Forest, and XGBoost
Random Forest Regressor selected as final deployed model
Real-time predictions via Flask web application
Dropdown-based categorical inputs for consistency
Performance classification: High / Average / Low Performer

🧠 Algorithms Used
Linear Regression – Baseline model
Random Forest Regressor – Final deployed model
XGBoost Regressor – Used for performance comparison

🛠️ Tech Stack
Python
Pandas, NumPy
Scikit-learn
XGBoost
Flask
HTML & CSS


⚙️ Installation & Setup

Clone the repository:
git clone https://github.com/your-username/employee-productivity-prediction.git

Navigate to the project directory:
cd employee-productivity-prediction

Install required packages:
pip install -r requirements.txt


Run the Flask application:
python app.py


Open browser and go to:
http://127.0.0.1:5000/

🧪 Input Parameters
Quarter
Department
Day
Team
Targeted Productivity
SMV
Overtime
Incentive
Idle Time
Idle Men
Number of Style Changes
Number of Workers
Year, Month, Day Number

📊 Output
Predicted Productivity Score

Performance Level:
High Performer
Average Performer
Low Performer


📈 Final Model Selection

After evaluating all models, Random Forest Regressor was chosen due to:
Better handling of non-linear data
Robust performance
Stable and reliable predictions


📌 Conclusion
This project demonstrates an end-to-end machine learning workflow, from data preprocessing and model training to deployment using Flask. It highlights the practical application of ML in workforce productivity analysis.


👤 Author
Sudheer Muthyala
B.Tech (ECE) | Aspiring Software & ML Engineer



