# SmartHealthMonitoringSystem

SmartHealthMonitoringSystem is a machine learning-powered health monitoring application designed to assess health risks based on user-provided data. This project leverages a Voting Classifier to predict risk levels and visualize key health metrics to assist users in monitoring their health proactively.

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Dataset](#dataset)
4. [Models](#models)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Results](#results)
8. [Contributing](#contributing)
9. [License](#license)

## Overview
SmartHealthMonitoringSystem collects and analyzes health-related data, including metrics like heart rate, BMI, and stress levels, to predict the user's risk level. The app uses a Voting Classifier for a robust, ensemble-based classification.

## Features
- Predicts health risk levels based on input data.
- Visualizes health metrics like heart rate, calories burned, steps, and more.
- Supports a three-class classification: Low, Medium, and High risk.
- Built using synthetic health data to simulate real-world application.

## Dataset
The dataset `synthetic_health_data.csv` includes the following columns:
- **user_id**: Unique identifier for each user.
- **age**, **gender**, **height_cm**, **weight_kg**, **bmi**
- **heart_rate**, **blood_pressure**, **calories_burned**, **steps**
- **sleep_hours**, **stress_level**, **risk_level**

## Models
The Voting Classifier combines multiple machine learning models to improve predictive accuracy. The app uses:
- Decision Tree
- K-Nearest Neighbors (KNN)
- Random Forest

These models are combined to vote on the predicted risk level, achieving a more balanced classification outcome.

## Installation
1. Clone this repository:
    ```bash
    git clone https://github.com/your-username/SmartHealthMonitoringSystem.git
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Run the main app script:
    ```bash
    python Health_App.py
    ```
2. Enter your health data as prompted, or load a sample data file to test.

## Results
The Voting Classifier model achieves an accuracy of approximately **53%** with the current dataset and configuration. Further tuning and data preprocessing may be applied to enhance model performance.

## Contributing
Contributions are welcome! Please fork this repository and create a pull request with your changes. For major updates, open an issue first to discuss potential enhancements.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
