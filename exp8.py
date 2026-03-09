from sklearn.datasets import make_classification
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# 1. Generate a synthetic dataset
X, y = make_classification(n_samples=100, n_features=10, n_informative=5, n_redundant=2, random_state=42)
# 2. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 3. Create a logistic regression model
model = LogisticRegression()
# 4. Apply RFE for feature selection
rfe = RFE(estimator=model, n_features_to_select=5)  # Select top 5 features
rfe.fit(X_train, y_train)
# 5. Get selected features
selected_features = rfe.support_
feature_ranking = rfe.ranking_
# 6. Display the results
print("Selected Features (True = Selected, False = Not Selected):")
print(selected_features)
print("\nFeature Ranking (1 = Best, Higher = Less Important):")
print(feature_ranking)
