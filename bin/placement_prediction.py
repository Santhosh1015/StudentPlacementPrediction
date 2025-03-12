# Importing required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
file_path = "/home/santhosh/Documents/Project/myenv/bin/collegePlace.csv"  # Replace with your file path
df = pd.read_csv(file_path)

# Display dataset overview
print("Dataset Overview:")
print(df.head())

# Preprocessing
# Handle missing values
df.fillna(method='ffill', inplace=True)

# Encode categorical columns
label_encoders = {}
categorical_columns = ['Gender', 'Stream', 'Hostel', 'HistoryOfBacklogs']  # Columns to encode
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Define features (X) and target (y)
X = df[['Age', 'Gender', 'Stream', 'Internships', 'CGPA', 'Hostel', 'HistoryOfBacklogs']]  # Features
y = df['PlacedOrNot']  # Target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred) * 100
print("\nModel Performance:")
print(f"Accuracy: {accuracy:.2f}%")  # Display accuracy as a percentage
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Define stream mapping
stream_mapping = {
    "CSE": 0,
    "ECE": 1,
    "Mechanical": 2,
    "Civil": 3,
    "IT": 4
}

# Function to take user input and predict
def get_user_input():
    print("\nEnter the following details to predict placement:")
    age = int(input("Age: "))
    gender = input("Gender (Male/Female): ").strip().capitalize()
    print("Available Streams: CSE, ECE, Mechanical, Civil, IT")
    stream = input("Stream: ").strip()
    internships = int(input("Number of Internships: "))
    cgpa = float(input("CGPA: "))
    hostel = int(input("Hostel (1 for Yes, 0 for No): "))
    history_of_backlogs = int(input("History of Backlogs (1 for Yes, 0 for No): "))

    # Encode categorical inputs
    gender = label_encoders['Gender'].transform([gender])[0]

    # Validate and map the stream input
    if stream not in stream_mapping:
        raise ValueError("Invalid stream entered. Please use one of the predefined streams.")
    stream = stream_mapping[stream]  # Map the stream using stream_mapping

    # Prepare input for prediction as a DataFrame
    user_data = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Stream': stream,
        'Internships': internships,
        'CGPA': cgpa,
        'Hostel': hostel,
        'HistoryOfBacklogs': history_of_backlogs
    }])

    user_data = scaler.transform(user_data)  # Scale the input data
    return user_data

# Predict placement based on user input
user_input = get_user_input()
prediction = model.predict(user_input)
prediction_prob = model.predict_proba(user_input)[0][1] * 100  # Get probability for placement

# Display prediction result
if prediction[0] == 1:
    print(f"\nPrediction: The student is likely to be placed with a confidence of {prediction_prob:.2f}%.")
else:
    print(f"\nPrediction: The student is unlikely to be placed with a confidence of {100 - prediction_prob:.2f}%.")
