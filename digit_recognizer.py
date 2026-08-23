from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import joblib

print("=" * 60)
print("HANDWRITTEN DIGIT RECOGNIZER")
print("=" * 60)

# Load built-in handwritten digit dataset.
digits = load_digits()
X, y = digits.data, digits.target

print(f"Total images: {len(X)}")
print(f"Image size: {digits.images[0].shape[0]} x {digits.images[0].shape[1]}")
print("Classes:", sorted(set(y)))

# Split data.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Build model.
model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=3))
])

print("\nTraining model...")
model.fit(X_train, y_train)

# Evaluate.
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# Save model.
joblib.dump(model, "digit_recognizer_model.pkl")
print("\nSaved model: digit_recognizer_model.pkl")

# Generate prediction output image.
sample_count = 12
sample_images = digits.images[:sample_count]
sample_features = digits.data[:sample_count]
sample_labels = digits.target[:sample_count]
sample_predictions = model.predict(sample_features)

fig = plt.figure(figsize=(10, 7))
for i in range(sample_count):
    ax = fig.add_subplot(3, 4, i + 1)
    ax.imshow(sample_images[i], cmap="gray")
    ax.set_title(f"Pred: {sample_predictions[i]} | Actual: {sample_labels[i]}")
    ax.axis("off")

fig.suptitle("Handwritten Digit Recognition - Sample Predictions")
plt.tight_layout()
plt.savefig("digit_predictions.png", dpi=200)
print("Saved result image: digit_predictions.png")
plt.show()
