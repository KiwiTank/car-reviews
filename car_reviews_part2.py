import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics
import matplotlib.pyplot as plt


def svm_sent_class(test_perc=0.2, rand=99)
    df = pd.read_csv('stemmed_reviews.csv')

    vec = TfidfVectorizer(ngram_range=(1, 2),
                          min_df=3,
                          max_df=0.8,
                          sublinear_tf=True,
                          use_idf=True)

    x = df['Reviews']
    y = df['Sentiment']

    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=test_perc, random_state=rand)

    x_train = vec.fit_transform(x_train).toarray()
    x_test = vec.transform(x_test).toarray()

    model = svm.SVC()
    model.fit(x_train, y_train)

    prediction = model.predict(x_test)
    accuracy = metrics.accuracy_score(prediction, y_test)
    print(str('Predicted accuracy of {:04.2f}'.format(accuracy * 100)) + '%')
    con_mat = confusion_matrix(y_test, prediction)
    figure = ConfusionMatrixDisplay(confusion_matrix=con_mat, display_labels=model.classes_)
    figure.plot()
    plt.show()


if __name__ == "__main__":
    pass
