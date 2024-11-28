import pandas as pd
import numpy as np

# Load the customer dataset
customer_data = pd.read_csv("customer.csv")

# Separate data by Conversion value
conversion_1_data = customer_data[customer_data["Conversion"] == 1]
conversion_0_data = customer_data[customer_data["Conversion"] == 0]

# Select all conversion 1 and sample 600 from conversion 0
sampled_conversion_0 = conversion_0_data.sample(n=600, random_state=42)

# Combine the data to create a small dataset
small_customer_data = pd.concat([conversion_1_data, sampled_conversion_0])

# Shuffle the dataset
small_customer_data = small_customer_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to a new CSV file
small_customer_csv_path = "small_customer.csv"
small_customer_data.to_csv(small_customer_csv_path, index=False)

# Split into training and testing
train_data = pd.concat([
    conversion_1_data.iloc[:350],
    sampled_conversion_0.iloc[:350]
])
test_data = pd.concat([
    conversion_1_data.iloc[350:],
    sampled_conversion_0.iloc[350:]
])

# Save the training and testing datasets
train_csv_path = "train_small_customer.csv"
test_csv_path = "test_small_customer.csv"
train_data.to_csv(train_csv_path, index=False)
test_data.to_csv(test_csv_path, index=False)

print(f"Small customer dataset saved to: {small_customer_csv_path}")
print(f"Training dataset saved to: {train_csv_path}")
print(f"Testing dataset saved to: {test_csv_path}")
