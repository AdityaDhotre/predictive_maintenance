import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('predictive_maintenance/sensor_data.csv')


# Preprocess the data
def preprocess_data(df):
    return df


# Train the model
def train_model(df,X_train,y_train):
    # Create a random forest classifier
    clf = RandomForestClassifier()

    # Train the model
    clf.fit(X_train, y_train)

    return clf


# Predict maintenance requirements
def predict_maintenance(model, data):
    # Perform prediction using the trained model
    predictions = model.predict(data)

    return predictions


# Streamlit app
def main():
    st.title('Predictive Maintenance Demo')

    # Load the data
    data = load_data()

    # Preprocess the data
    data = preprocess_data(data)

    # Split the data into features and labels
    X = data.drop('maintenance_required', axis=1)
    y = data['maintenance_required']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(data,X_train,y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    print(accuracy)


    #Calculating Feature Importance
    feature_importance = model.feature_importances_
    estimators = model.estimator_
    print('features')
    for i,importance in enumerate(feature_importance):
        print(f"feature {i+1}:{importance}")
    
    # User input for new data
    st.subheader('Enter New Sensor Readings')
    

    # Get user inputs
    feature1 = st.number_input('Feature 1', value=0.0)
    feature2 = st.number_input('Feature 2', value=0.0)
    feature3 = st.number_input('Feature 3', value=0.0)
    feature4 = st.number_input('Feature 4', value=0.0)

    # Create a DataFrame for the user input
    new_data = pd.DataFrame({
        'feature1': [feature1],
        'feature2': [feature2],
        'feature3': [feature3],
        'feature4': [feature4]
    })

    # Make predictions for the new data
    predictions = predict_maintenance(model, new_data)

    # Display the predictions
    st.subheader('Predictions')
    if predictions[0] == 1:
        st.write('Maintenance is required.')
    else:
        st.write('Maintenance is not required.')

    # Display the accuracy
    st.subheader('Accuracy')
    st.write(accuracy)
    # Show the original dataset
    st.subheader('Original Dataset')
    st.write(data)


if __name__ == '__main__':
    main()
