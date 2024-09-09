Contains the Code and the Dataset used.

A comprehensive approach to spam email classification using natural language processing 
(NLP) and machine learning techniques is presented. The code begins by importing essential 
libraries such as NumPy, Pandas, Matplotlib, NLTK, and scikit-learn for basic calculations, 
data manipulation, plotting, and natural language processing. It further utilizes the TF-IDF 
vectorization technique for text data representation. 

The dataset is loaded from a CSV file containing email messages, and an initial exploration of 
the data is conducted. Visualizations, such as a count plot of email categories (spam or not 
spam) and a histogram depicting the distribution of email lengths, provide insights into the 
dataset's structure. 

A word cloud generation technique is employed to visualize the most frequent words in both 
spam and non-spam emails, offering a qualitative understanding of the dataset. The code then 
proceeds to preprocess the data, including labeling the email categories as 0 or 1 and splitting 
the dataset into training and testing sets. 

Three different classifiers, namely Multinomial Naive Bayes, Bernoulli Naive Bayes, and a 
Multilayer Perceptron (MLP) neural network, are implemented and evaluated for their 
predictive performance. The classifiers are compared based on metrics such as accuracy, 
precision, recall, and F1 score. 

Moreover, the code includes additional functionalities, such as generating precision, recall, and 
F1 score comparison visualizations using Plotly. The implementation of confusion matrices 
and classification reports further aids in assessing the models' overall performance.

Lastly, the code demonstrates user interaction by allowing the user to input a test message and 
predict whether it is classified as spam or not using the trained MLP classifier. 

In summary, the presented code exemplifies a systematic and thorough approach to spam email 
classification, leveraging machine learning models and NLP techniques to achieve accurate 
predictions. 
 
The proliferation of digital communication has given rise to an unprecedented volume of 
emails, making effective email categorization a critical task. The code presented herein 
addresses the significant challenge of classifying emails into spam and non-spam categories. 
The comprehensive approach combines natural language processing (NLP) techniques and 
machine learning models to discern patterns within the textual content of emails. 


Design Overview: 

1. Data Loading and Exploration: 
• Design: Imported necessary libraries (numpy, pandas, matplotlib.pyplot, seaborn) 
for data handling and visualization. 
• Algorithm: Used pd.read_csv to load the dataset and displayed the first 10 rows for 
initial exploration. 
• Implementation: Checked the distribution of the target variable ("Category") using 
seaborn.countplot and visualized the length distribution of messages. 
 
2. Data Analysis and Visualization: 
• Design: Analyzed the distribution of spam and ham messages. 
• Algorithm: Used seaborn.countplot and calculated the percentage of each category. 
• Implementation: Created a histogram to visualize the length distribution of 
messages. 
 
3. Text Processing and Feature Extraction: 
• Design: Prepared the data for model training by encoding labels and transforming 
text data. 
• Algorithm: Used LabelEncoder to encode 'ham' as 0 and 'spam' as 1. Employed 
TfidfVectorizer or CountVectorizer for feature extraction. 
• Implementation: Transformed the text data into numerical vectors. 
4. Data Splitting: 
• Design: Divided the dataset into training and testing sets. 
• Algorithm: Utilized train_test_split from sklearn.model_selection. 
• Implementation: Split the data into X_train, X_test, y_train, y_test.

6. Model Training and Evaluation: 
• Design: Trained three models: Multilayer Perceptron (MLP), Multinomial Naive 
Bayes (MNC), and Bernoulli Naive Bayes (BC). 
• Algorithm: Implemented each model using MLPClassifier, MultinomialNB, and 
BernoulliNB. Evaluated the performance metrics. 
• Implementation: Trained and evaluated models, comparing accuracy, precision, 
recall, and F1-score.

8. Additional Testing: 
• Design: Tested models with external input to classify messages. 
• Algorithm: Implemented functions to input messages for classification. 
• Implementation: Tested models with external input for practical application.
