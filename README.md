# Basic-Churn-ANN
Basic neural network trained to solve binary classification problem of whether a 
customer will exit based on input data.

### Pre-reqs
* Python 3.12 (and not higher)
  
### To run
1. Setup virtual environment and install packages in ```requirements.txt```
2. Optionally run ```training/trainer.py``` to generate a new model with an 
optimal set of hyperparameters, or ```tuning/tuner.py``` to find an optimal set 
of hyperparameters.
3. Open up the terminal (powershell/bash) and run ```streamlit run app.py```