import numpy as np # for basic calculations
import pandas as pd #for importing data
import matplotlib.pyplot as plt #for plotting graphs
import nltk #the heart of natural language processing
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings("ignore") #To not get any warnings
df = pd.read_csv("data.csv",encoding='latin-1')
df.head(10)
import seaborn as sns
print("Count of", np.round(df.Category.value_counts(normalize=True),2)*100)
sns.countplot(x="Category",data=df).set(title="Status Distribution")
# Now we can also check the length of the mail text 
df["length"] = df["Message"].apply(len)
df
import plotly.express as px
px.histogram(df,x="length",nbins=500,title="length Distribution in data",
            color_discrete_sequence=['indianred'],
            opacity=0.8)
# Label the status as 1 or 0
df.loc[:,'Category']=df.Category.map({'ham':0, 'spam':1})
df['Category'] = df['Category'].astype(int)
df.head()
from wordcloud import WordCloud, STOPWORDS
spam = df[df['Category']==1]
ham = df[df['Category']==0]

def wordcloud_generation(data,title):
    words = " ".join(df["Message"])
    wordcloud = WordCloud(stopwords=STOPWORDS,
                         max_words=1500,
                         max_font_size=350,random_state=42,
                         width=2000,height=800,
                         colormap='tab20c',
                         repeat=False,
                         include_numbers=False,
                         collocations=True).generate(words)
    plt.figure(figsize=(20,10))
    plt.imshow(wordcloud,interpolation="nearest")
    plt.axis("off")
    plt.title(title)
    plt.show()
wordcloud_generation(spam, "Spam WordCloud")
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

vector = CountVectorizer()
text = vector.fit_transform(df["Message"])

X_train, X_test, y_train, y_test = train_test_split(text, df["Category"], test_size=0.3,
                                                   random_state=33)
print("The shape of respective train and test values : ")
print("X_train : " , X_train.shape)
print("X_test : " , X_test.shape)
print("y_train : " , y_train.shape)
print("y_test : " , y_test.shape)
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
MLP = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000,
                    activation="relu",
                   alpha=0.001,
                   batch_size="auto",
                    early_stopping=False,
                    verbose=True,
                    learning_rate="adaptive")
MLP.fit(X_train,y_train)

predictions1 = MLP.predict(X_test)

print("Prediction results of MLP Classifier :: ")
print("----------------------------------------")
print("Accuracy score: {}". format(accuracy_score(y_test, predictions1)) )
print("Precision score: {}". format(precision_score(y_test, predictions1)) )
print("Recall score: {}". format(recall_score(y_test, predictions1)))
print("F1 score: {}". format(f1_score(y_test, predictions1)))
MNC = MultinomialNB()
MNC.fit(X_train,y_train)
predictions2 = MNC.predict(X_test)

print("Prediction results of MNC Classifier :: ")
print("----------------------------------------")
print("Accuracy score: {}". format(accuracy_score(y_test, predictions2)) )
print("Precision score: {}". format(precision_score(y_test, predictions2)) )
print("Recall score: {}". format(recall_score(y_test, predictions2)))
print("F1 score: {}". format(f1_score(y_test, predictions2)))
BC = BernoulliNB()
BC.fit(X_train,y_train)
predictions3 = MNC.predict(X_test)

print("Prediction results of BC Classifier :: ")
print("----------------------------------------")
print("Accuracy score: {}". format(accuracy_score(y_test, predictions3)) )
print("Precision score: {}". format(precision_score(y_test, predictions3)) )
print("Recall score: {}". format(recall_score(y_test, predictions3)))
print("F1 score: {}". format(f1_score(y_test, predictions3)))
compare_df =[]
models = [("Multinomial NB", MNC), ("Bernoulli NB", BC),("MLP Classifier", MLP) ]
for model_name, model in models:
    predictions = model.predict(X_test)
    Accuracy_Score = accuracy_score(y_test,predictions)
    Precision_Score = precision_score(y_test,predictions)
    Recall_Score = recall_score(y_test,predictions)
    F1_Score = f1_score(y_test,predictions)
    
    compare_df.append([model_name,Accuracy_Score,Precision_Score,Recall_Score,F1_Score])

compare_df = pd.DataFrame(compare_df, columns=["Model_Name","Accuracy_Score",
                                              "Precision_Score",
                                              "Recall_Score",
                                              "F1_Score"])
compare_df
fig = px.bar(compare_df,x="Model_Name",y="Precision_Score",title="Precision_Score Comparison",)
fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
fig = px.bar(compare_df,x="Model_Name",y="Recall_Score",title="Recall_Score Comparison",)
fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
fig = px.bar(compare_df,x="Model_Name",y="F1_Score",title="F1_Score Comparison",)
fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Assuming X_train, y_train, X_test, y_test are properly defined

# Multinomial Naive Bayes
nb = MultinomialNB()
nb.fit(X_train, y_train)

print("Multinomial Naive Bayes Scores:")
print("Training Accuracy:", nb.score(X_train, y_train))
print("Test Accuracy:", nb.score(X_test, y_test))

# Predict using Multinomial Naive Bayes
y_pred_nb = nb.predict(X_test)

# Confusion Matrix and Classification Report for Multinomial Naive Bayes
conf_nb = confusion_matrix(y_test, y_pred_nb)
print("Confusion Matrix (Multinomial Naive Bayes):")
print(conf_nb)

classif_nb = classification_report(y_test, y_pred_nb)
print("Classification Report (Multinomial Naive Bayes):")
print(classif_nb)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

def test_model_with_input(model, vectorizer):
    # Get user input for the test data file
    test_file_path = input("Enter the path of the test data file: ")

    # Read the test data from the specified file
    test_data = pd.read_csv(test_file_path, encoding='latin-1')

    # Extract messages from the test data
    test_messages = test_data['Message']

    # Transform messages using the provided vectorizer
    test_text = vectorizer.transform(test_messages)

    # Predict labels for the test data using the given model
    test_predictions = model.predict(test_text)

    # Display the results
    print("\nPredictions:")
    for message, prediction in zip(test_messages, test_predictions):
        if prediction == 1:
            print(f"The message '{message}' is classified as spam.")
        else:
            print(f"The message '{message}' is classified as not spam.")
        
        print()
# Assuming df, vector, and MLP are already defined as in your code
test_model_with_input(MLP, vector)
def test_model_with_input(model, vectorizer):
    # Get user input for a single message
    user_input_message = input("Enter a message to classify: ")

    # Transform the user input using the provided vectorizer
    user_input_text = vectorizer.transform([user_input_message])

    # Predict the label for the user input using the given model
    user_input_prediction = model.predict(user_input_text)[0]

    # Display the result
    if user_input_prediction == 1:
        print(f"The message '{user_input_message}' is classified as spam.")
    else:
        print(f"The message '{user_input_message}' is classified as not spam.")

# Assuming vector and MLP are already defined as in your code
test_model_with_input(MLP, vector)