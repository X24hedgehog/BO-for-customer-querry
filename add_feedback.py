import pandas as pd
import numpy as np

# Load the customer_conversion_dataset
customer_conversion_dataset = pd.read_csv('customer_conversion_dataset.csv')

# Generate varied positive, negative, and neutral feedback
positive_feedback = [
    "The product is amazing!", "I really liked the product.", "Great experience!",
    "This is exactly what I was looking for.", "Very satisfied with the service.",
    "The product works well for my needs.", "It has improved my workflow significantly.",
    "A decent product with good value.", "I am generally pleased with the purchase.",
    "Good quality, but there’s room for improvement.", "It does the job as advertised.",
    "Happy with the purchase, though not exceptional.", "It’s a reliable product.",
    "I use it regularly and it meets my expectations.", "This product is quite useful.",
    "Satisfied overall, no major complaints.", "The quality is impressive for the price.",
    "It’s nice to have, though not life-changing.", "A good choice for anyone considering it.",
    "Solid product, works as expected."
]

negative_feedback = [
    "I don't like the product.", "Not satisfied at all.", "It did not meet my expectations.",
    "The quality is below average.", "Disappointing experience.", 
    "It’s okay, but not worth the price.", "Not as reliable as I hoped.", 
    "I expected more for the cost.", "This product didn’t work well for me.", 
    "Could have been better.", "The experience was frustrating.", 
    "It’s functional but needs improvements.", "Not a fan of the design.",
    "It’s passable, but I wouldn’t recommend it.", "Feels like it’s missing something.",
    "Didn’t quite work as expected.", "A bit underwhelming overall.",
    "I might look for alternatives.", "It’s just okay, not great.", 
    "Not terrible, but I’ve seen better."
]

neutral_feedback = [
    "The product is not bad.", "I quite like the product.", "It’s decent, but not for me.",
    "The product quality is okay.", "I like it, but the price is too high.",
    "A good product, but I’m not ready to buy.", "It’s nice, but not what I need.",
    "The product has potential.", "It’s useful, but not for my current situation.",
    "I’m satisfied, but it’s not a fit for me."
]

# Expand feedback pools to approximately 200 samples each by varying tones
positive_feedback = positive_feedback * 10
negative_feedback = negative_feedback * 10
neutral_feedback = neutral_feedback * 10

np.random.shuffle(positive_feedback)
np.random.shuffle(negative_feedback)
np.random.shuffle(neutral_feedback)

# Generate feedback function
def generate_feedback(row):
    """
    Generate realistic feedback based on the Conversion label.
    """
    if row["Conversion"] == 1:
        # Choose from varied positive feedback
        return np.random.choice(positive_feedback)
    else:
        # For 30% of Conversion = 0, give neutral feedback
        if np.random.rand() < 0.3:
            return np.random.choice(neutral_feedback)
        else:
            return np.random.choice(negative_feedback)

# Add the feedback column
customer_conversion_dataset["feedback"] = customer_conversion_dataset.apply(generate_feedback, axis=1)

# Save the new dataset as customer.csv
customer_csv_path = "customer.csv"
customer_conversion_dataset.to_csv(customer_csv_path, index=False)
