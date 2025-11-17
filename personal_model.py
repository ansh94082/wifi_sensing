import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("/kaggle/input/wimans/annotation.csv")


df = df[["label", "number_of_users"]]


print("Distribution of user counts:")
print(df['number_of_users'].value_counts())


def extract_csi_features(path):
    """Load .npy CSI file and return 1D feature vector (mean, std, min, max)."""
    try:
        csi_array = np.load(f"/kaggle/input/wimans/wifi_csi/amp/{path}.npy")
    except FileNotFoundError:
        print(f"Missing file: {path}.npy")
        return None
    
    features = []
    for tx in range(csi_array.shape[1]):
        for rx in range(csi_array.shape[2]):
            for sc in range(csi_array.shape[3]):
                signal = csi_array[:, tx, rx, sc]
                features.extend([signal.mean(), signal.std(), signal.min(), signal.max()])
    return features

feature_rows = []
for idx, row in df.iterrows():
    file_id = row['label']
    features = extract_csi_features(file_id)
    
    if features is None:
        features = [np.nan] * 1080
        
    feature_rows.append([file_id] + features)

features_df = pd.DataFrame(feature_rows)
features_df.columns = ["file_id"] + [f"f_{i}" for i in range(1080)]


df_combined = pd.merge(df, features_df, left_on='label', right_on='file_id', how='left')


df_combined = df_combined.drop(["label", "file_id"], axis=1)
df_combined.dropna(inplace=True) # Drop rows where feature extraction failed


X = df_combined.drop("number_of_users", axis=1)
y = df_combined["number_of_users"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


model = RandomForestClassifier(
    n_estimators=200,       
    random_state=42,
    class_weight='balanced', 
    n_jobs=-1               
)


print("Training the model...")
model.fit(X_train, y_train)
print("Training complete!")


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {accuracy:.2f}\n")

#Print the classification report

print("Classification Report:")
print(classification_report(y_test, y_pred))

#Generate and plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.xlabel('Predicted Number of Users')
plt.ylabel('Actual Number of Users')
plt.title('Confusion Matrix for Human Counting')
plt.show()