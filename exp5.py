import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=5, random_state=42)

# Split into labeled and unlabeled data
X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(X, y, test_size=0.9, random_state=42)

# Initial classifier trained on labeled data
clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X_labeled, y_labeled)

# Self-training loop
n_iterations = 5
confidence_threshold = 0.9

for i in range(n_iterations):
    if len(X_unlabeled) == 0:
        break

    # Get prediction probabilities
    probs = clf.predict_proba(X_unlabeled)
    preds = clf.predict(X_unlabeled)
    confidences = np.max(probs, axis=1)

    # Select high-confidence predictions
    confident_indices = np.where(confidences >= confidence_threshold)[0]

    if len(confident_indices) == 0:
        print(f"No confident predictions in iteration {i+1}. Stopping early.")
        break

    X_new = X_unlabeled[confident_indices]
    y_new = preds[confident_indices]

    # Add to labeled set
    X_labeled = np.vstack([X_labeled, X_new])
    y_labeled = np.concatenate([y_labeled, y_new])

    # Remove from unlabeled set
    mask = np.ones(len(X_unlabeled), dtype=bool)
    mask[confident_indices] = False
    X_unlabeled = X_unlabeled[mask]
    y_unlabeled = y_unlabeled[mask]

    # Retrain classifier
    clf.fit(X_labeled, y_labeled)

    print(f"Iteration {i+1}: Labeled data size = {len(y_labeled)}")

# Final evaluation
if len(X_unlabeled) > 0:
    y_pred = clf.predict(X_unlabeled)
    print("Final Accuracy on remaining unlabeled (test) data:", accuracy_score(y_unlabeled, y_pred))
else:
    print("No data left for final evaluation.")