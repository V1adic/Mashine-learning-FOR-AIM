from sklearn.ensemble import AdaBoostClassifier
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import nltk
from nltk.corpus import stopwords
import pickle
import numpy as np
from datasets import load_dataset

def make_blending_prediction(basic_clfs, final_clf, data):
    y0 = []
    for c in basic_clfs:
        y0.append(c.predict(data))
    y0_t = np.array(y0).transpose()
    return final_clf.predict(y0_t)

def make_stacking_prediction(basic_clfs, final_clf, data):
    y0 = []
    for c in basic_clfs:
        y0.append(c.predict(data))
    y0_t = np.array(y0).transpose()
    return final_clf.predict(y0_t)


dataset = load_dataset('json', data_files={'train': 'Data/train.jsonl','test':'Data/validation.jsonl'})
nltk.download('stopwords')

vectorizer = CountVectorizer(max_features=500, min_df=4, max_df=0.7, stop_words=stopwords.words('english'))
X_train_vec = vectorizer.fit_transform(dataset['train']['text'])

print(dataset['train']['text'][:1])
vocabulary = vectorizer.get_feature_names_out()
print(vocabulary)

tfidf = TfidfTransformer()
X_train_idf = tfidf.fit_transform(X_train_vec)

X_test_vec = vectorizer.transform(dataset['test']['text'])
X_test_idf = tfidf.transform(X_test_vec)
print(X_test_idf[:1].toarray().sum())

X_train = X_train_idf.toarray()
X_test = X_test_idf.toarray()
X_train, X_test, y_train, y_test = X_train, X_test, dataset['train']['label'], dataset['test']['label']

# Глава 2, обучение на объектах, неверно классифицированных на предыдущем шаге


gbc = GradientBoostingClassifier(n_estimators = 100, max_depth = 6, random_state=42)

filename = "./models/AdaBoost_100_6.pickle"
model = pickle.load(open(filename, "rb"))
y_pred = model.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))
print("ADA")
print(model.get_params())

X_train_0, X_train_1, y_train_0, y_train_1 = train_test_split(X_train, y_train, test_size=0.33, random_state=42)

# Blending

N = 10
y_pred_1 = []
crf = []
for n in range(1,N+1):
    crf.append(RandomForestClassifier(n_estimators = 2,max_depth=2, random_state=n))
    crf[-1].fit(X_train_0,y_train_0)
    y_pred_1.append(crf[-1].predict(X_train_1).reshape(len(X_train_1),1))
    y_pred_1t = np.array(y_pred_1).transpose()[0]

y_pred_1t = np.array(y_pred_1).transpose()[0]
clf_final = RandomForestClassifier(n_estimators = 10,max_depth=6, random_state=42)
clf_final.fit(y_pred_1t,y_train_1)

y_test_pred = make_blending_prediction(crf,clf_final, X_test)
print(f"blending -> {metrics.accuracy_score(y_test, y_test_pred)}")
print(metrics.accuracy_score(y_test, crf[0].predict(X_test)))

N = 10
y_pred_1 = []
crf_stack = []
kf = KFold(n_splits=N, random_state=None, shuffle=False)

x_test_2 = []
y_test_2 = []

pre_prediction = np.zeros((len(X_train), N))

for i, (train_index, test_index) in enumerate(kf.split(X_train)):
    X_train_0, X_test_1 = np.array(X_train)[train_index], np.array(X_train)[test_index]
    y_train_0, y_test_1 = np.array(y_train)[train_index], np.array(y_train)[test_index]
    
    crf_stack.append(RandomForestClassifier(n_estimators = 2,max_depth=2, random_state=i))
    crf_stack[-1].fit(X_train_0,y_train_0)
    pre_prediction[test_index,i]=crf_stack[-1].predict(X_test_1)

clf_stack_final = RandomForestClassifier(n_estimators = 10,max_depth=6, random_state=42)
clf_stack_final.fit(pre_prediction,y_train)

y_test_pred = make_stacking_prediction(crf_stack,clf_stack_final, X_test)
print(f"stacking -> {metrics.accuracy_score(y_test, y_test_pred)}")
print(metrics.accuracy_score(y_test, crf_stack[0].predict(X_test)))

# Глава 3, обучение в направленном

filename = "./models/GBC_100_6.pickle"
model = pickle.load(open(filename, "rb"))
y_pred = model.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))
print("GBC")
print(model.get_params())

X_train_0, X_train_1, y_train_0, y_train_1 = train_test_split(X_train, y_train, test_size=0.33, random_state=42)

# Blending
N = 10
y_pred_1 = []
crf = []
for n in range(1,N+1):
    crf.append(RandomForestClassifier(n_estimators = 2,max_depth=2, random_state=n))
    crf[-1].fit(X_train_0,y_train_0)
    y_pred_1.append(crf[-1].predict(X_train_1).reshape(len(X_train_1),1))
    y_pred_1t = np.array(y_pred_1).transpose()[0]

y_pred_1t = np.array(y_pred_1).transpose()[0]

clf_final = RandomForestClassifier(n_estimators = 10,max_depth=6, random_state=42)
clf_final.fit(y_pred_1t,y_train_1)

y_test_pred = make_blending_prediction(crf,clf_final, X_test)
print(f"blending -> {metrics.accuracy_score(y_test, y_test_pred)}")
print(metrics.accuracy_score(y_test, crf[0].predict(X_test)))

N = 10
y_pred_1 = []
crf_stack = []
kf = KFold(n_splits=N, random_state=None, shuffle=False)

x_test_2 = []
y_test_2 = []

pre_prediction = np.zeros((len(X_train), N))

for i, (train_index, test_index) in enumerate(kf.split(X_train)):
    X_train_0, X_test_1 = np.array(X_train)[train_index], np.array(X_train)[test_index]
    y_train_0, y_test_1 = np.array(y_train)[train_index], np.array(y_train)[test_index]
    
    crf_stack.append(RandomForestClassifier(n_estimators = 2,max_depth=2, random_state=i))
    crf_stack[-1].fit(X_train_0,y_train_0)
    pre_prediction[test_index,i]=crf_stack[-1].predict(X_test_1)

clf_stack_final = RandomForestClassifier(n_estimators = 10,max_depth=6, random_state=42)
clf_stack_final.fit(pre_prediction,y_train)

y_test_pred = make_stacking_prediction(crf_stack,clf_stack_final, X_test)
print(f"stacking -> {metrics.accuracy_score(y_test, y_test_pred)}")
print(metrics.accuracy_score(y_test, crf_stack[0].predict(X_test)))

print("Параметры моделей выбрались идентичным образом!!!")