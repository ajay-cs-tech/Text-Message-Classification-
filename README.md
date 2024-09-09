
# Text-Message-Classification

A comprehensive approach to spam email classification using natural language processing (NLP) and machine learning techniques is presented. The code begins by importing essential libraries such as NumPy, Pandas, Matplotlib, NLTK, and scikit-learn for basic calculations, data manipulation, plotting, and natural language processing. It further utilizes the TF-IDF vectorization technique for text data representation.

The dataset is loaded from a CSV file containing email messages, and an initial exploration of the data is conducted. Visualizations, such as a count plot of email categories (spam or not spam) and a histogram depicting the distribution of email lengths, provide insights into the dataset's structure.

A word cloud generation technique is employed to visualize the most frequent words in both spam and non-spam emails, offering a qualitative understanding of the dataset. The code then proceeds to preprocess the data, including labeling the email categories as 0 or 1 and splitting the dataset into training and testing sets.

Three different classifiers, namely Multinomial Naive Bayes, Bernoulli Naive Bayes, and a Multilayer Perceptron (MLP) neural network, are implemented and evaluated for their predictive performance. The classifiers are compared based on metrics such as accuracy, precision, recall, and F1 score.

**Refer and download the CSV file while using the code.**






## Requirements

**Ensure you have Python 3.6+ installed along with the following libraries:**

• numpy: For basic numerical calculations. 

• pandas: For importing and handling the dataset. 

• matplotlib.pyplot: For plotting graphs and visualizations. 

• nltk: Natural Language Toolkit for text processing. 

• sklearn.feature_extraction.text.TfidfVectorizer: For transforming text data into 
numerical vectors. 

• warnings: For handling warnings in the code. 

• seaborn: Data visualization library based on matplotlib. 

• plotly.express: Interactive plotting library for creating histograms. 

• WordCloud, STOPWORDS: For generating word clouds. 

• sklearn.feature_extraction.text.CountVectorizer: For converting a collection of text 
documents to a matrix of token counts. 

• sklearn.model_selection.train_test_split: For splitting the dataset into training and 
testing sets. 

• sklearn.neural_network.MLPClassifier: Multilayer Perceptron classifier for 
classification tasks. 

• sklearn.naive_bayes.MultinomialNB, BernoulliNB: Naive Bayes classifiers for text 
classification. 

• sklearn.metrics: For evaluating the performance of machine learning models. 

• plotly.express: For creating interactive and expressive visualizations.


**Install the required packages using:**

```bash
pip install -r requirements.txt
```
## Usage

1. Setup Environment:
* Clone the repository.
* Install dependencies using the above command.

2. Run the Application:

* Run the main code.
* Enter the text for Spam Detection.

3. Interpret Results:

* Follow the graphs and results.

## Contributing

Contributions are welcome! If you have suggestions, enhancements, or issues, please submit them via GitHub issues.

