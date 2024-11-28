import pandas as pd
from textblob import TextBlob

customer_data = pd.read_csv("customer.csv")

# Load the training dataset to exclude those datapoints
train_data = pd.read_csv("train_small_customer.csv")

# Exclude training datapoints from the original dataset
test_candidates = customer_data[~customer_data.index.isin(train_data.index)]

# Randomly sample 5000 datapoints from the remaining data
test_data = test_candidates.sample(n=5000, random_state=42)

# Save the new test dataset
test_csv_path = "test_dataset.csv"
test_data.to_csv(test_csv_path, index=False)


# Count the number of conversions (0 and 1)
conversion_counts = test_data["Conversion"].value_counts()

# Print the counts
print("Number of customers with Conversion = 0:", conversion_counts.get(0, 0))
print("Number of customers with Conversion = 1:", conversion_counts.get(1, 0))

# def compute_sentiment_score(response):
#     """
#     Compute the sentiment score using TextBlob.
#     """
#     sentiment = TextBlob(response).sentiment
#     # Use polarity as the sentiment score (ranges from -1 to 1)
#     return (sentiment.polarity + 1) / 2  # Map to range [0, 1]

# positive_feedback = [
#     "The product is amazing!", "I really liked the product.", "Great experience!",
#     "This is exactly what I was looking for.", "Very satisfied with the service.",
#     "The product works well for my needs.", "It has improved my workflow significantly.",
#     "A decent product with good value.", "I am generally pleased with the purchase.",
#     "Good quality, but there’s room for improvement.", "It does the job as advertised.",
#     "Happy with the purchase, though not exceptional.", "It’s a reliable product.",
#     "I use it regularly and it meets my expectations.", "This product is quite useful.",
#     "Satisfied overall, no major complaints.", "The quality is impressive for the price.",
#     "It’s nice to have, though not life-changing.", "A good choice for anyone considering it.",
#     "Solid product, works as expected."
# ]

# negative_feedback = [
#     "I don't like the product.", "Not satisfied at all.", "It did not meet my expectations.",
#     "The quality is below average.", "Disappointing experience.", 
#     "It’s okay, but not worth the price.", "Not as reliable as I hoped.", 
#     "I expected more for the cost.", "This product didn’t work well for me.", 
#     "Could have been better.", "The experience was frustrating.", 
#     "It’s functional but needs improvements.", "Not a fan of the design.",
#     "It’s passable, but I wouldn’t recommend it.", "Feels like it’s missing something.",
#     "Didn’t quite work as expected.", "A bit underwhelming overall.",
#     "I might look for alternatives.", "It’s just okay, not great.", 
#     "Not terrible, but I’ve seen better."
# ]

# neutral_feedback = [
#     "The product is not bad.", "I quite like the product.", "It’s decent, but not for me.",
#     "The product quality is okay.", "I like it, but the price is too high.",
#     "A good product, but I’m not ready to buy.", "It’s nice, but not what I need.",
#     "The product has potential.", "It’s useful, but not for my current situation.",
#     "I’m satisfied, but it’s not a fit for me."
# ]

# print([compute_sentiment_score(p) for p in positive_feedback])
# print([compute_sentiment_score(n) for n in negative_feedback])
# print([compute_sentiment_score(ne) for ne in neutral_feedback])

