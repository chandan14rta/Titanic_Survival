#import Libraries 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_csv(r'C:\Users\Chandan\OneDrive - iitkgp.ac.in\Desktop\Git\Titanic_Survival\tested.csv')

# Fill nan values
train['Age'].fillna(train['Age'].median(), inplace=True)
train.dropna(inplace=True)

# Plot 1: Distribution of survivors
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=train)
plt.title('Distribution of Survivors')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()

# Plot 2: Age distribution by survival status
plt.figure(figsize=(8,6))
sns.histplot(data=train, x='Age', hue='Survived', kde=True, bins=30)
plt.title('Age Distribution by Survival Status')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# One-Hot Encoding for categorical variables
sex = pd.get_dummies(train['Sex'], drop_first=True)
embark = pd.get_dummies(train['Embarked'], drop_first=True)
train.drop(['Sex','Embarked','Name','Ticket','Cabin'], axis=1, inplace=True)
train = pd.concat([train, sex, embark], axis=1)

# Plot 3: Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(train.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.show()

# Train test split
X = train.drop('Survived', axis=1)
y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

# Create logistic regression model and fit
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on test set
test_y = model.predict(X_test)
print("Model Accuracy:", model.score(X_test, y_test))

# Evaluation Metrics and Visualizations

# Confusion Matrix
cm = confusion_matrix(y_test, test_y)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
y_pred_prob = model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Print Classification Report
print("Classification Report:")
print(classification_report(y_test, test_y))
