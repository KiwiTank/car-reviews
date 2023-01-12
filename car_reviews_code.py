import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics
import matplotlib.pyplot as plt


def nb_sent_class():
    df = pd.read_csv('stemmed_reviews.csv')

    vec = CountVectorizer()

    x = df['Reviews']
    y = df['Sentiment']

    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=99)

    x_train = vec.fit_transform(x_train).toarray()
    x_test = vec.transform(x_test).toarray()

    model = MultinomialNB()
    model.fit(x_train, y_train)

    prediction = model.predict(x_test)
    accuracy = metrics.accuracy_score(prediction, y_test)
    print(str('Predicted accuracy of {:04.2f}'.format(accuracy * 100)) + '%')
    con_mat = confusion_matrix(y_test, prediction)
    figure = ConfusionMatrixDisplay(confusion_matrix=con_mat, display_labels=model.classes_)
    figure.plot()
    return plt.show()


if __name__ == "__main__":
    nb_sent_class()
