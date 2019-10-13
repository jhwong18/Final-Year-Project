# Demand Estimation of Full-Cut Promotion on E-Commerce Company
# ABSTRACT

Full-cut promotion is a type of promotion where customers spend a minimum threshold amount in a single order to enjoy a fixed discount amount for that order. This type of promotion is popular in China, particularly in E-commerce. However, there is no existing literature that estimates the promotion driven demand and optimizes the threshold and discount amount in the full-cut promotion. 

The purpose of this thesis is to construct a methodology that enables E-commerce companies to measure the performance of full-cut promotion and optimize the promotion for revenue management. The proposed methodology has four main steps:

-exploratory data analysis
-proposed model to measure promotion driven demand
-customer segmentation based on proposed model
-benchmark model to compare performance against proposed model 
-machine learning

The benchmark model we have implemented is a modified conditional gradient approach from Jagabathula’s paper. The significance of the research is that we have enabled E-commerce companies to measure a customer’s attraction towards the full-cut promotion and created an approach to segment customers simply based on the product-level features in an aggregated transactional dataset. Many existing customer segmentation techniques involve customer demographics whereby empirical researchers do not have access to such information. Further, we have constructed a model, using the outputs from demand estimation and customer types, to estimate their utility towards the full-cut promotion and hence, optimize the threshold and discount amount. We have also used various machine learning models to investigate the relationship between features like market price towards the attraction for full-cut promotion. Therefore, E-commerce will not only be able to obtain the optimal full-cut promotion from a given transactional dataset, they will also obtain insights on why customers react towards the different discount & threshold amounts differently as well as how to fine-tune full-cut promotions.

REFERENCES:
[1] Jagabathula, S., Subramanian, L., & Venkataraman, A. (2018). A Conditional Gradient Approach for Nonparametric Estimation of Mixing Distributions.
