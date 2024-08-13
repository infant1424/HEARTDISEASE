import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load the dataset
data = pd.read_csv('C:\\Users\\richa\\Desktop\\project\\heart.csv')

# Select features (first 12 columns) and target (13th column)
X = data.iloc[:, 0:13]  # Selecting the first 12 features
Y = data.iloc[:, 13]    # Selecting the 13th column as the target

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree model
DecisionTree = DecisionTreeClassifier(criterion="entropy", random_state=2, max_depth=5)
DecisionTree.fit(X_train, Y_train)

# Save the model using pickle
with open('decision_tree_model.pkl', 'wb') as model_file:
    pickle.dump(DecisionTree, model_file)

print("Model trained and saved successfully.")
