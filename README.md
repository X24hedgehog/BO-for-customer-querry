# **Customer Query Optimization with Bayesian Learning**

## **Problem Statement**

Suppose your company just launch a new product and want to sell it. The product attracts a lot of people but not all of them will purchase it, so the company want to offer some free trial of the product to advertise the new characteristic of the product. The dilemma is, we have too many "potential customers", and it is often impractical and economically infeasible to provide such special treatment to all of them, but we still need to offer some free trials to see how the product works and how do people respond to it.

In this context, we only observe customer feedback and the binary conversion label (`0` or `1`) **after** providing them with the free trial offer. This creates a challenge:

**How can we strategically select which customers to query in order to maximize conversions while exploring a diverse range of customer profiles?**

---

## **Why using Bayesian Optimization for Problem**

1. **No Analytical Expression**:  
   The customer score is based on feedback and conversion data, which cannot be expressed analytically. This is similar to optimizing the taste of a cookie based on its ingredients—there’s no clear formula.

2. **Smooth and Continuous**:  
   Small changes in customer features (e.g., age, location, or social media activity) are expected to result in small changes to the customer score. The intuition is simple: customers with similar background and similar behaviour might have similar response, given that the set of features and behaviour is comprehensive enough. This smoothness allows BO to make informed decisions based on predictions from a surrogate model.

3. **Expensive to Evaluate**:  
   Each customer query involves offering a free trial and waiting for feedback, making it a costly operation. BO helps in minimizing the number of evaluations needed to find the best candidates.

4. **Noisy Observations**:  
   Feedback and conversions can be noisy due to factors like customer mood, external circumstances, or random variation. BO handles such noisy observations by incorporating Gaussian likelihoods into its probabilistic model, in fact, the observation is the latent variable with a Gaussian additive noise.

By addressing these characteristics, BO allows us to efficiently navigate the trade-off between exploring a variety of profile of customers and at the same time maximizing the probability that each customer chosen will actually be converted (buy the product). 

---


### **Core Algorithm: Upper Confidence Bound (UCB)**

We achieve the balance between exploration and exploitation using the **Upper Confidence Bound (UCB)** algorithm. This algorithm selects customers based on a combination of:

- **Mean**: The expected conversion potential (**customer score**).
- **Variance**: The uncertainty of the prediction, ensuring diverse exploration. It has been proof that maximizing the variance is equivalent to maximizing the information gain given a specific set of observed customers.

The **customer score** is derived from two components:
1. **Feedback**: Customer's response after using the free trial.
2. **Conversion Label**: Whether the customer ultimately purchased the product.

---

## **Dataset and Processing**

### **Dataset**

The original dataset, customer_conversion_dataset.csv, is from a Kaggle challenge

I create a set of feedback based on positivity and negativity (feedback that AI agent can have access if they querry the customer), and add them to the original dataset to get customer.csv

TO conclude, the AI agent works with a dataset containing:

- **Customer Features**: Attributes such as age, location, interaction information (e.g., social media activity), and other demographic/behavioral details.
- **Feedback and Conversion Status**: Observed **only after querying a customer** (offering a free trial).

(Note: The feedback is simulated as this is a work sample, in real life it can be collected by the AI agent)

### **Data Processing Workflow**

1. **Feature Conversion**: Converts raw customer attributes into numerical features ready to be fit into the GP model.
2. **Gaussian Process Regressor (GP Regressor)**:
   - Initializes a GPR model to predict customer scores. I choose to use a Linear kernel (Dot product in sklearn)
   - The model is trained based on available training data if the training mode is allowed (correspond to when the company already has a dataset with feedback and conversion label available). Otherwise, it will update on the way when querrying customers from the test dataset.
3. **Iterative Customer Querying**:
   - The AI agent selects customers to query using the current GPR model and the UCB algorithm.
   - After querying, the agent observes feedback and conversion status and updates the GPR model.
   - This iterative process continues until the maximum number of queries is reached (e.g., a fixed budget for free trials).

---

## **Evaluation Metrics**

To evaluate the AI agent's performance, we measure its ability to select high-converting customers. Specifically, we calculate:

- **Hit Rate**: Among all customers who purchased the product (`conversion = 1`) in the test dataset, the percentage of those queried by the AI agent after exhausting the query budget (e.g., 400 queries in the example code).

This metric reflects the agent’s efficiency in identifying customers most likely to convert, balancing exploration and exploitation within the given constraints.

---

## **How to run the project**

Run `pip install -r requirements.txt` to install the required packages

Run query_customer.py file to see the result

---

