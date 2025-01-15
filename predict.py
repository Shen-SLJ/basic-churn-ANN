from core.ChurnPredictor import ChurnPredictor

# fake example input data
input_data = {
    'CreditScore': 600,
    'Geography': 'France',
    'Gender': 'Male',
    'Age': 40,
    'Tenure': 3,
    'Balance': 60000,
    'NumOfProducts': 2,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 50000
}

if __name__ == '__main__':
    churn_predictor = ChurnPredictor()
    prediction_probability = churn_predictor.predict_likelihood(input_data)

    print(f"Prediction: {prediction_probability}")
    if prediction_probability > 0.5:
        print("Likely to churn")
    else:
        print("Not likely to churn")
