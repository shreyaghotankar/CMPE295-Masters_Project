# CMPE295-Masters Project
## University : [San Jose State University](http://www.sjsu.edu/)
## Project: Sustainable Fashion Recommendation Application using Machine Learning
## Advisor: Dr. KaiKai Liu
## Instructor: Prof. Dan Harkey

## Machine Learning process techniques

1. Machine learning typically involves cleaning the datasets; choosing a model; training, evaluating and tuning the model; deploying the model; and recommending in production using the model.
   ![](images/mltechniques.png)

## Machine Leaning Algorithm
### Deep fashion data cleaning

1. We start with the DeepFashion dataset with ~1K+ attributes and 280K+ entries, and clean and reduce the dimensionality of the dataset to less than 20 features each for upper and lower-body dress types. We also populate a dataset with ratings between matching upper and lower-body dress types to seed the model. Link to the data cleaning notebook is [here](https://github.com/shreyaghotankar/CMPE295-Masters_Project/blob/master/datacleaning/dataframedeepfashion/AttributesCleaningDeepFashionDataset.ipynb)
   
### Collaborative filtering with K-means clustering

1. Clustering the dataset by the above reduced attributes using K-Means, and creating a matrix of aggregate ratings between upper/lower-body dress types improves  model evaluation results. Link to the notebook is [here](https://colab.research.google.com/drive/1sB19cdBUqEyjBk7Y3W2O4VDz8Ph_V296?authuser=4#scrollTo=pbT1Vyty1TaZ)

2.  We then apply model-based Collaborative Filtering (CF) with SVD matrix factorization to compute the embeddings necessary to predict ratings between any upper/lower-body dress items.
   
3.  Evaluating the model helps tune the hyperparameters to K-Means and CF and guide re-training the model. We used techniques such as elbow method to effeciently determine the clusters.

4. The trained model is periodically deployed to AWS as a callable API. A request dress type is mapped to its corresponding cluster, matched by the model to its target dress type, and used to recommend matching dresses from the inventory in the response. Link to the code is [here](https://github.com/shreyaghotankar/CMPE295-Masters_Project/blob/master/datacleaning/deployment/attributeconversion.py)
   
   ![](images/algorithm.png)

5. Link to the manually mapped images: [here](https://github.com/shreyaghotankar/CMPE295-Masters_Project/blob/master/datacleaning/deployment/attributeconversion.py)





