import pandas as pd
import numpy as np
import random
from textblob import TextBlob
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


# Preprocessing Function
def preprocess_features(customer_feedbacks):
    """
    Preprocess the customer dataset to generate a feature matrix.

    Parameters
    ----------
    customer_feedbacks : pd.DataFrame
        DataFrame containing customer data.

    Returns
    -------
    np.ndarray
        Preprocessed feature matrix.
    """
    # Define numeric and categorical columns
    numeric_features = [
        "Age", "TimeSpent (minutes)", "PagesViewed", "FormSubmissions",
        "Downloads", "CTR_ProductPage", "ResponseTime (hours)",
        "FollowUpEmails", "SocialMediaEngagement"
    ]
    categorical_features = [
        "Gender", "Location", "DeviceType",
        "ReferralSource", "PaymentHistory"
    ]

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    # Fit and transform the data
    feature_matrix = preprocessor.fit_transform(customer_feedbacks)
    return feature_matrix

# Bayesian Optimization
class BayesianOptimization:
    def __init__(self):
        kernel = DotProduct()
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1**2)
        self.points, self.scores = [], []

    def acquisition_function(self, x):
        mean, std = self.gp.predict(x, return_std=True)
        return mean + 0.5 * std  # UCB: balance between exploration and exploitation

    def recommend_next(self, customer_indices, feature_matrix):
        """
        Recommend the next customer to query based on unqueried customers.

        Parameters
        ----------
        customer_indices : list
            Indices of the unqueried customers in the dataset.
        feature_matrix : np.ndarray
            Preprocessed feature matrix of all customers.

        Returns
        -------
        int
            The index of the next customer to query.
        """
        best_idx = None
        best_value = float('-inf')

        for idx in customer_indices:
            customer_features = feature_matrix[idx].reshape(1, -1)  # Features for customer
            acquisition_value = self.acquisition_function(customer_features)

            if acquisition_value > best_value:
                best_value = acquisition_value
                best_idx = idx

        return best_idx

    def update(self, customer_features, score):
        """
        Update the GP model with new observations.

        Parameters
        ----------
        customer_features : np.ndarray
            Features of the queried customer.
        score : float
            Sentiment score for the queried customer.
        """
        self.points.append(customer_features)
        self.scores.append(score)
        self.gp.fit(np.vstack(self.points), self.scores)

# Query and feedback simulation
def get_observations(customer_feedbacks, idx):
    """
    Fetch the feedback of the queried customer.
    """
    return (customer_feedbacks.iloc[idx]["feedback"], customer_feedbacks.iloc[idx]["Conversion"])

def compute_sentiment_score(response, conversion):
    """
    Compute the sentiment score using TextBlob.
    """
    sentiment = TextBlob(response).sentiment
    # Use polarity as the sentiment score (ranges from -1 to 1)
    sentiment_score = (sentiment.polarity + 1) / 4 # Map to range [0, 1/2]
    return (sentiment_score + conversion) / 2  

def main():
    BO_method = False # If false, then we simply choose customer randomly (for comparision reason)
    train_available = False
    # Load the training and testing datasets
    train_feedbacks = pd.read_csv("train_small_customer.csv")
    test_feedbacks = pd.read_csv("test_dataset.csv")

    # Preprocess features for both training and testing
    train_features = preprocess_features(train_feedbacks)
    test_features = preprocess_features(test_feedbacks)

    # Initialize Bayesian Optimization
    bo = BayesianOptimization()

    # # Fit the GP model on the training data
    if train_available:
        for idx in range(len(train_feedbacks)):
            feedback, conversion = get_observations(train_feedbacks, idx)
            sentiment_score = compute_sentiment_score(feedback, conversion)
            bo.update(train_features[idx].reshape(1, -1), sentiment_score)

    # Determine the number of conversion = 1 in the test set
    k = test_feedbacks["Conversion"].sum()
    n = 400  # Number of customers that AI agent needs to query

    # Query n customers from the test set
    test_indices = list(range(len(test_feedbacks)))
    queried_customers = []

    if BO_method:
        for j in range(int(n)):
            # Query the next customer
            next_customer_idx = bo.recommend_next(test_indices, test_features)
            next_customer = test_feedbacks.iloc[next_customer_idx]
            queried_customers.append(next_customer)

            # Remove the queried customer from the test set
            test_indices.remove(next_customer_idx)

            # Get feedback and sentiment score
            feedback, conversion = get_observations(test_feedbacks, next_customer_idx)
            sentiment_score = compute_sentiment_score(feedback, conversion)

            # Update GP model
            customer_features = test_features[next_customer_idx].reshape(1, -1)
            bo.update(customer_features, sentiment_score)

            # Compute and print the conversion rate every 25 queries
            if j % 25 == 0:
                queried_df = pd.DataFrame(queried_customers)
                true_positive_count = queried_df["Conversion"].sum()
                conversion_rate = true_positive_count / k if k > 0 else 0
                print(f"After querying {j} customer, rate of converted customer found by agent: {conversion_rate * 100:.2f}%")
    else:
        # Simulate random querying
        random.seed(42)  # Set seed for reproducibility
        queried_indices = random.sample(test_indices, n)
        for j, next_customer_idx in enumerate(queried_indices):
            next_customer = test_feedbacks.iloc[next_customer_idx]
            queried_customers.append(next_customer)

            # Compute and print the conversion rate every 25 queries
            if j % 25 == 0:
                queried_df = pd.DataFrame(queried_customers)
                true_positive_count = queried_df["Conversion"].sum()
                conversion_rate = true_positive_count / k if k > 0 else 0
                print(f"Query {j}: Naive Baseline Conversion Rate: {conversion_rate * 100:.2f}%")

    # Evaluate the AI agent
    queried_df = pd.DataFrame(queried_customers)
    true_positive_count = queried_df["Conversion"].sum()
    conversion_rate = true_positive_count / k if k > 0 else 0

    print(f"Rate of converted customer found by agent: {conversion_rate * 100:.2f}%")

if __name__ == "__main__":
    main()
