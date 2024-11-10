import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import messagebox

# Generate synthetic data
np.random.seed(42)
n_samples = 1000

# Features
age = np.random.randint(18, 70, size=n_samples)
heart_rate = np.random.randint(60, 100, size=n_samples)
calories_burned = np.random.randint(150, 500, size=n_samples)
steps = np.random.randint(1000, 10000, size=n_samples)
sleep_hours = np.random.randint(4, 10, size=n_samples)
stress_level = np.random.randint(1, 10, size=n_samples)

# Target (Risk Level): 0 - Low, 1 - Medium, 2 - High
risk_level = np.random.choice([0, 1, 2], size=n_samples)

# Create a DataFrame
data = pd.DataFrame({
    'age': age,
    'heart_rate': heart_rate,
    'calories_burned': calories_burned,
    'steps': steps,
    'sleep_hours': sleep_hours,
    'stress_level': stress_level,
    'risk_level': risk_level
})

# Data Preprocessing
# Features and target
X = data.drop(columns=['risk_level'])
y = data['risk_level']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of the model: {accuracy * 100:.2f}%')

# Tkinter GUI Application
def make_prediction():
    # Collect user input from the entry fields
    try:
        age_input = int(age_entry.get())
        heart_rate_input = int(heart_rate_entry.get())
        calories_input = int(calories_entry.get())
        steps_input = int(steps_entry.get())
        sleep_input = float(sleep_entry.get())
        stress_input = int(stress_entry.get())

        # Standardize the user input
        sample_data = np.array([[age_input, heart_rate_input, calories_input, steps_input, sleep_input, stress_input]])
        sample_scaled = scaler.transform(sample_data)

        # Make the prediction
        predicted_risk = model.predict(sample_scaled)
        risk_levels = {0: 'Low', 1: 'Medium', 2: 'High'}
        result = risk_levels[predicted_risk[0]]

        # Show prediction result in a message box
        messagebox.showinfo("Prediction Result", f'The predicted risk level is: {result}')
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values for all fields.")

# Create the main window
root = tk.Tk()
root.title("Health Risk Level Prediction")

# Set the window size
root.geometry("400x500")

# Add input labels and entry widgets
tk.Label(root, text="Age:").pack(pady=5)
age_entry = tk.Entry(root)
age_entry.pack(pady=5)

tk.Label(root, text="Heart Rate:").pack(pady=5)
heart_rate_entry = tk.Entry(root)
heart_rate_entry.pack(pady=5)

tk.Label(root, text="Calories Burned:").pack(pady=5)
calories_entry = tk.Entry(root)
calories_entry.pack(pady=5)

tk.Label(root, text="Steps:").pack(pady=5)
steps_entry = tk.Entry(root)
steps_entry.pack(pady=5)

tk.Label(root, text="Sleep Hours:").pack(pady=5)
sleep_entry = tk.Entry(root)
sleep_entry.pack(pady=5)

tk.Label(root, text="Stress Level (1-10):").pack(pady=5)
stress_entry = tk.Entry(root)
stress_entry.pack(pady=5)

# Add a button to trigger the prediction
predict_button = tk.Button(root, text="Predict Risk Level", command=make_prediction)
predict_button.pack(pady=20)

# Start the GUI loop
root.mainloop()
