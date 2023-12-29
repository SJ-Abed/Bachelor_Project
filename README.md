
This project aims to predict customer shopping baskets in their subsequent purchases.
Its analytical method involves creating four types of attributes (product, product-customer, customer, time) for each product-customer pair and predicting the likelihood of the presence of that product based on these features. 
The project's goal is to forecast the repetition of product purchases, not the purchase of new products.
 Here, after performing preprocessing and visualization in file number 1, features are constructed, and machine learning models are trained.
In the second section, each product category is referred to as a separate record, effectively ignoring differences between different brands and sizes of a product type.
Sections 3 and 4 are similar to sections 1 and 2, with the difference that they are designed for customers with a relatively high number of orders, allowing for better predictions.
Although this approach results in the loss of a significant portion of customers, the model's accuracy significantly improves.
In folder number 3, various methods are employed to enhance the model's efficiency, such as tuning hyperparameters, changing the minimum requirements for each customer to be included in the model, and utilizing different neural networks for better predictions.
In file number 5, dimension reduction methods are also used to improve the model's performance and reduce the model-building time.
