import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load the breast cancer dataset from sklearn
data = load_breast_cancer()

# Convert dataset into pandas DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)

# Add target column (0 = malignant, 1 = benign)
df['target'] = data.target

# Show first 5 rows
print(df.head())

# Check unique target values (0 and 1)
print(df['target'].unique())

# Show shape of dataset (rows, columns)
print("Data Shape:", data.data.shape)

# Count of target classes
print("Exploring the Dataset (Target Counts):\n", df['target'].value_counts())
# Normalized count (percentage)
print("Exploring the Dataset (Normalized Target Counts):\n", df['target'].value_counts(normalize=True))

# Plot histogram for target distribution
df['target'].plot.hist(title="Target Label Distribution")
plt.show()

# Summary statistics for all features
print("Dataset Summary:\n", df.describe().T)

# Plot histogram for each feature
for col in df.columns[:-1]:   # exclude target
    plt.title(col)
    df[col].plot.hist(title=col)  
    plt.show()

# Pairplot to visualize relationships between features, colored by target
sns.pairplot(df, hue='target')
plt.show()

# Split features (X) and target (y)
X = df.drop('target', axis=1)  
y = df['target']

SEED = 42  # random seed for reproducibility

# Split into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# Number of samples in training and testing sets
xtrain_samples = X_train.shape[0]
xtest_samples = X_test.shape[0]
print(f'There are {xtrain_samples} samples for training and {xtest_samples} samples for testing.')

# Import Support Vector Classifier
from sklearn.svm import SVC

# Create SVC model with linear kernel
svc = SVC(kernel='linear')
# Train the model
svc.fit(X_train, y_train)

# Predict target for test set
y_pred = svc.predict(X_test)

# Import evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix heatmap
gg = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                 xticklabels=['Malignant', 'Benign'],
                 yticklabels=['Malignant', 'Benign'])
gg.set_title('Confusion Matrix of Linear SVM')
plt.show()

# Print classification report
print("Classification Report:\n", classification_report(y_test, y_pred))
