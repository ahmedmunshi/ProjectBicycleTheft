# Bicycle Theft Prediction Project

This project uses machine learning to predict bicycle theft outcomes based on historical data from the Toronto Police Service. It includes SMOTE balancing for handling class imbalance and a Flask web interface for making predictions.

## Features

- Data analysis and visualization of bicycle theft patterns
- SMOTE (Synthetic Minority Over-sampling Technique) for handling imbalanced classes
- Decision Tree Classifier for prediction
- Flask web interface for real-time predictions
- Responsive frontend design

## Project Structure

```
bike_theft_project/
│
├── model/
│   └── model_training.py     # Model training and SMOTE implementation
│
├── templates/
│   └── index.html           # Frontend template
│
├── static/
│   └── style.css           # CSS styling
│
├── app.py                  # Flask application
│
└── requirements.txt        # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ahmedmunshi/ProjectBicycleTheft.git
cd ProjectBicycleTheft
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Train the model:
```bash
python model/model_training.py
```

4. Run the Flask application:
```bash
python app.py
```

5. Open your browser and go to `http://localhost:5000`

## Usage

Enter the following information in the web interface:
- Year
- Hour of day (0-23)
- Bicycle cost

The model will predict whether the bicycle is likely to be:
- STOLEN
- RECOVERED
- UNKNOWN

## Data Source

The data used in this project comes from the Toronto Police Service's public dataset on bicycle thefts.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- SMOTE (imbalanced-learn)
- Flask
- HTML/CSS