import pandas as pd
import csv
import nltk
from pymorphy2 import MorphAnalyzer



morph = MorphAnalyzer()

df = pd.read_csv("data.csv")



def is_cyr_word(word):

    for ch in word:
        if (ch=='ё'): ch='е'
        if not('а'<= ch <= 'я'):
            return False

    return True


def process_text(text):

    lower = (word.lower() for word in nltk.wordpunct_tokenize(text))
    cyr = (word for word in lower if is_cyr_word(word))
    norm_form = (morph.parse(word)[0].normal_form for word in cyr)

    return ' '.join(norm_form)




df2 = pd.DataFrame()

df2[0] = df['text'].map(process_text)
df2[1] = df['is_positive']

df2.to_csv('words.csv', index=False)






from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split




def getMessagesAndLabels(FILENAME):
    messages = []
    labels = []

    with open(FILENAME, "r", newline="") as file:
        reader = csv.DictReader(file)
        print("Извлекаем данные...")
        for row in reader:
            messages.append(row['0'])
            labels.append(row['1'])


    return messages, labels




FILENAME = "words.csv"

messages, labels = getMessagesAndLabels(FILENAME)


ru_stopwords = [
u'я', u'а', u'да', u'но', u'она', u'он', u'тебе', u'мне', u'ты', u'и', u'у', u'на', u'ща', u'ага',
u'так', u'там', u'какие', u'который', u'какая', u'туда', u'давай', u'короче', u'кажется', u'вообще',
u'ну', u'не', u'чет', u'неа', u'свои', u'наше', u'хотя', u'такое', u'например', u'кароч', u'как-то',
u'нам', u'хм', u'всем', u'нет', u'да', u'оно', u'своем', u'про', u'вы', u'м', u'тд', u'тп',
u'вся', u'кто-то', u'что-то', u'вам', u'это', u'эта', u'эти', u'этот', u'прям', u'либо', u'как', u'мы',
u'просто', u'блин', u'очень', u'самые', u'твоем', u'ваша', u'кстати', u'вроде', u'типа', u'пока', u'ок'
]




vectorizer = TfidfVectorizer(max_features=200000, norm='l1', stop_words = ru_stopwords)
vectorizer.fit(messages)

X_train, X_test, y_train, y_test = train_test_split(messages, labels, test_size=0.25)




X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)

print("Обучение...")


clf = LinearSVC()
clf.fit(X_train, y_train)


print("Точность прогноза:")
print(clf.score(X_test, y_test))
