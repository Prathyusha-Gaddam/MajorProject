from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import BorderlineSMOTE
import os
from IPython.display import HTML
from pathlib import Path

app = Flask(__name__)


window_size=50
def extract_features(window):
    features = []
    features.append(window['StartTime'].mean())
    features.append(window['Duration'].mean())
    features.append(window['StartVerticalPosition'].mean())
    features.append(window['VerticalSize'].mean())
    features.append(window['PeakVerticalVelocity'].mean())
    features.append(window['PeakVerticalAcceleration'].mean())
    features.append(window['StartHorizontalPosition'].mean())
    features.append(window['HorizontalSize'].mean())
    features.append(window['StraightnessError'].mean())
    features.append(window['Slant'].mean())
    features.append(window['LoopSurface'].mean())
    return features

def optimal_number_of_bins(data):
    """
    Determine the optimal number of bins using the Freedman-Diaconis rule.
    """
    IQR = np.percentile(data, 75) - np.percentile(data, 25)
    bin_width = 2 * IQR / (len(data) ** (1/3))
    if bin_width == 0:
        n_bins = 1
    else:
        n_bins = int(np.ceil((data.max() - data.min()) / bin_width))
    return n_bins



def detect_dysgraphia(input_file):
    # Process the uploaded file (input_file) here if needed
    # For this example, we'll simply return the success message
    # Load the time series data for each participant
    print(input_file)
    if not os.path.isfile(input_file):
        return "File not found"

    # Load the time series data for each participant
    try:
        data = pd.read_csv(input_file)
    except pd.errors.EmptyDataError:
        return "Empty file or invalid CSV format"
    # Apply the moving window algorithm to extract features for each participant
    features = []
    for i in range(len(data)):
        participant_features = []
        for j in range(0, len(data.iloc[i]), window_size):
            if j + window_size < len(data.iloc[i]):
                window = data.iloc[i][j:j+window_size]
            else:
                window = data.iloc[i][j:len(data.iloc[i])]
            window_features = extract_features(window)
            participant_features.extend(window_features)
        features.append(participant_features)
        
        # Convert the features to a numpy array
    features_array = np.array(features)

    # Save the features to a CSV file
    pd.DataFrame(features_array).to_csv('selected_test_new1.csv', index=False)
    # Define time series data (10 features x 10 time steps)
    data = pd.read_csv('selected_test_new1.csv')
    my_arr=[]
    # Iterate over columns (features)
    for i in range(data.shape[1]):
        # Compute optimal number of bins
        n_bins = optimal_number_of_bins(data.iloc[:, i].to_numpy())

        # Compute bin edges
        min_val = np.min(data.iloc[:, i])
        max_val = np.max(data.iloc[:, i])
        bin_edges = np.linspace(min_val, max_val, n_bins + 1)

        # Discretize data using bin edges
        discrete_data = np.digitize(data.iloc[:, i], bin_edges)

        mean = np.mean(discrete_data, axis=0)
        rounded_mean = round(mean, 3)
        my_arr.append(rounded_mean)
    # Load the data into a pandas DataFrame
    data = pd.read_csv('Selected_features.csv')

    # Convert all columns to float except the label and Id columns
    data = data.astype({'StartTime': float, 'Duration': float, 'StartVerticalPosition': float, 
                        'VerticalSize': float, 'PeakVerticalVelocity': float, 'PeakVerticalAcceleration': float, 
                        'StartHorizontalPosition': float, 'HorizontalSize': float, 'StraightnessError': float, 
                        'Slant': float, 'LoopSurface': float, 'label': int})

    # Split the data into features (X) and labels (y)
    X = data.drop(['label', 'Id'], axis=1) 
    y = data['label']
    # Define RandomOverSampler parameters
    ros = RandomOverSampler(sampling_strategy='minority', random_state=42)

    # Fit and transform data
    X_resampled, y_resampled = ros.fit_resample(X, y)
    #print(X_resampled)
    #print(y_resampled)
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=45)

    # Create an AdaBoost classifier model
    model = AdaBoostClassifier(random_state=42)

    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.1, 0.5, 1.0]
    }

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(model, param_grid=param_grid, scoring='accuracy', cv=5)
    grid_search.fit(X_train, y_train)

    # Get the best estimator and make predictions on the test data
    best_model = grid_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = best_model.predict(X_test)
    my_arr = np.array(my_arr)

    # Convert my_arr into a 2-D array with one column
    my_2d_arr = my_arr.reshape(1, -1)
    y_pred1=best_model.predict(my_2d_arr)
    #print("ypred1")
    ans=y_pred1[0]
    if ans==0:
        return "Healthy"
    return "Dysgraphia"

@app.route('/', methods=['GET','POST'])
def index():
    result = None
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return render_template('upload_form.html', result="no_file_message")
            else:
                file_path = "uploads/" + file.filename
                file.save(file_path)
                result = detect_dysgraphia(file_path)
    return render_template('index.html',result=result)


@app.route('/upload', methods=['POST'])
def upload():
    # Handle file upload and processing here
    # Assuming you store the result in a variable named 'result'
    result = None
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return render_template('upload_form.html', result="no_file_message")
            else:
                file_path = "uploads/" + file.filename
                file.save(file_path)
                result = detect_dysgraphia(file_path)
    return render_template('result.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
