#!/usr/bin/env python
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import RFECV
import seaborn as sns          
import matplotlib.pyplot as plt
import warnings
from IPython.display import display
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

warnings.filterwarnings('ignore')
initial_data = pd.read_csv('output_train.csv')
fig, ax = plt.subplots()
fig.canvas.set_window_title('output_train.csv')
sns.countplot(initial_data['status'], label="Sum") #.set_title(filename)
plt.show()
display(initial_data.head())

initial_data.drop(initial_data.columns[0], axis=1, inplace=True)

for column in initial_data.columns:
    if "Unnamed" in column:
        initial_data.drop(column, axis = 1, inplace=True)

initial_data['status']=initial_data['status'].map({'complete':1, 'active':1, 'cancelled':0, 'unsuccessful':0})
initial_data['methodType'] = initial_data['methodType'].map({'belowThreshold':0, 'reporting':1, 'aboveThresholdUA':2, 'negotiation.quick':3,
                                                           'aboveThresholdEU':4,'negotiation':5, 'aboveThresholdUA.defense':6, 'competitiveDialogueUA':7, 'competitiveDialogueEU':8,
                                                           'competitiveDialogueUA.stage2':9, 'competitiveDialogueEU.stage2':10})
initial_data['complaints.status']=initial_data['complaints.status'].map({'declined':0, 'resolved':1, 'claim':0, 'stopped':1, 'accepted':1, 'ignored':0, 'stopping':1, 'satisfied':1,
                                                           'invalid':1,'cancelled':1, 'draft':1, 'pending':1, 'answered':1, '0':0, 0:0, 1:1, '1':1})

initial_data['value'] = round(initial_data['value'], 3)
initial_data['cpvs'] = initial_data['cpvs']
initial_data['region'] = initial_data['region']
text = initial_data['title']
vectorizer = TfidfVectorizer()
# tokenize and build vocab
vectorizer.fit(text)
vector = vectorizer.transform(text)
print(vector.shape)
print(type(vector))
initial_data['title']=vector.toarray()
display(initial_data.head(n=150))

X = initial_data[['title','cpvs','complaints', 'complaints.status', 'questions', 'answer', 'documents', 'methodType', 'value', 'procuringEntity', 'region']]
y = initial_data['status']   
col_labels = ['title','cpvs','complaints', 'complaints.status', 'questions', 'answer', 'documents', 'methodType', 'value', 'procuringEntity', 'status', 'region']

initial_data.columns = col_labels
display(initial_data.head(n=150))
X = preprocessing.scale(X)
for c in col_labels:
     no_missing = initial_data[c].isnull().sum()
     if no_missing > 0:
         print(c)
         print(no_missing)
     else:
         print(c)
         print("No missing values")
         print(' ')
X_train, X_test, y_train, y_test = train_test_split(X,y)

QDA = QuadraticDiscriminantAnalysis()

scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted']
scores = cross_validate(QDA, X_train, y_train, scoring=scoring, cv=20)

sorted(scores.keys())
QDA_fit_time = scores['fit_time'].mean()
QDA_score_time = scores['score_time'].mean()
QDA_accuracy = scores['test_accuracy'].mean()
QDA_precision = scores['test_precision_macro'].mean()
QDA_recall = scores['test_recall_macro'].mean()
QDA_f1 = scores['test_f1_weighted'].mean()
#DA_roc = scores['test_roc_auc'].mean()

LR = LogisticRegression()

scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted']
scores = cross_validate(LR, X_train, y_train, scoring=scoring, cv=20)

sorted(scores.keys())
LR_fit_time = scores['fit_time'].mean()
LR_score_time = scores['score_time'].mean()
LR_accuracy = scores['test_accuracy'].mean()
LR_precision = scores['test_precision_macro'].mean()
LR_recall = scores['test_recall_macro'].mean()
LR_f1 = scores['test_f1_weighted'].mean()
#LR_roc = scores['test_roc_auc'].mean()

decision_tree = DecisionTreeClassifier()

scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted']
scores = cross_validate(decision_tree, X_train, y_train, scoring=scoring, cv=20)

sorted(scores.keys())
dtree_fit_time = scores['fit_time'].mean()
dtree_score_time = scores['score_time'].mean()
dtree_accuracy = scores['test_accuracy'].mean()
dtree_precision = scores['test_precision_macro'].mean()
dtree_recall = scores['test_recall_macro'].mean()
dtree_f1 = scores['test_f1_weighted'].mean()
#dtree_roc = scores['test_roc_auc'].mean()

LDA = LinearDiscriminantAnalysis()

scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted']
scores = cross_validate(LDA, X_train, y_train, scoring=scoring, cv=20)

sorted(scores.keys())
LDA_fit_time = scores['fit_time'].mean()
LDA_score_time = scores['score_time'].mean()
LDA_accuracy = scores['test_accuracy'].mean()
LDA_precision = scores['test_precision_macro'].mean()
LDA_recall = scores['test_recall_macro'].mean()
LDA_f1 = scores['test_f1_weighted'].mean()
#LDA_roc = scores['test_roc_auc'].mean()

random_forest = RandomForestClassifier()

scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted']
scores = cross_validate(random_forest, X_train, y_train, scoring=scoring, cv=20)

sorted(scores.keys())
forest_fit_time = scores['fit_time'].mean()
forest_score_time = scores['score_time'].mean()
forest_accuracy = scores['test_accuracy'].mean()
forest_precision = scores['test_precision_macro'].mean()
forest_recall = scores['test_recall_macro'].mean()
forest_f1 = scores['test_f1_weighted'].mean()
#forest_roc = scores['test_roc_auc'].mean()

KNN = KNeighborsClassifier()

scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted']
scores = cross_validate(KNN, X_train, y_train, scoring=scoring, cv=20)

sorted(scores.keys())
KNN_fit_time = scores['fit_time'].mean()
KNN_score_time = scores['score_time'].mean()
KNN_accuracy = scores['test_accuracy'].mean()
KNN_precision = scores['test_precision_macro'].mean()
KNN_recall = scores['test_recall_macro'].mean()
KNN_f1 = scores['test_f1_weighted'].mean()
#KNN_roc = scores['test_roc_auc'].mean()

bayes = GaussianNB()

scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted']
scores = cross_validate(bayes, X_train, y_train, scoring=scoring, cv=20)

sorted(scores.keys())
bayes_fit_time = scores['fit_time'].mean()
bayes_score_time = scores['score_time'].mean()
bayes_accuracy = scores['test_accuracy'].mean()
bayes_precision = scores['test_precision_macro'].mean()
bayes_recall = scores['test_recall_macro'].mean()
bayes_f1 = scores['test_f1_weighted'].mean()
#bayes_roc = scores['test_roc_auc'].mean()


initial_data_test = pd.read_csv('output_test.csv')
fig, ax = plt.subplots()
fig.canvas.set_window_title('output_test.csv')
sns.countplot(initial_data_test['status'], label="Sum") #.set_title(filename)
plt.show()

display(initial_data_test.head(n=70))
initial_data_test.drop(initial_data_test.columns[0], axis=1, inplace=True)
initial_data_test.drop(initial_data_test.columns[10], axis=1, inplace=True)
display(initial_data_test.head(n=70))


text_test = initial_data_test['title']
vectorizer_test = TfidfVectorizer()
# tokenize and build vocab
vectorizer_test.fit((text_test).values.astype('U'))
vector_test= vectorizer_test.transform((text_test).values.astype('U'))
print(vector_test.shape)
print(type(vector_test))
initial_data_test['title']=vector_test.toarray()
display(initial_data_test.head(n=150))

initial_data_test['methodType']=initial_data_test['methodType'].map({'belowThreshold':0, 'reporting':1, 'aboveThresholdUA':2, 'negotiation.quick':3,
                                                           'aboveThresholdEU':4,'negotiation':5, 'aboveThresholdUA.defense':6, 'competitiveDialogueUA':7, 'competitiveDialogueEU':8,
                                                           'competitiveDialogueUA.stage2':9, 'competitiveDialogueEU.stage2':10})
initial_data_test['complaints.status']=initial_data_test['complaints.status'].map({'declined':0, 'resolved':1, 'claim':0, 'stopped':1, 'accepted':1, 'ignored':0, 'stopping':1, 'satisfied':1,
                                                           'invalid':1,'cancelled':1, 'draft':1, 'pending':1, 'answered':1, '0':0, 0:0, 1:1, '1':1})

X_test = initial_data_test[['title','cpvs', 'complaints', 'complaints.status', 'questions', 'answer', 'documents', 'methodType', 'value', 'procuringEntity', 'region']]
col_labelss = ['title','cpvs', 'complaints', 'complaints.status', 'questions', 'answer', 'documents', 'methodType', 'value', 'procuringEntity', 'region']

initial_data_test.columns = col_labelss
display(initial_data_test.head(n=250))
X_test = preprocessing.scale(X_test)

for c in col_labelss:
     no_missing = initial_data_test[c].isnull().sum()
     if no_missing > 0:
         print(c)
         print(no_missing)
     else:
         print(c)
         print("No missing values")
         print(' ')
            
display(initial_data_test.head(n=70))

methods=[LR, decision_tree, LDA, QDA, random_forest, KNN, bayes]
for method in methods:
    method.fit(X_train, y_train)
    target_pred = method.predict(X_test)
    print(method)
    
models_initial = pd.DataFrame({
    'Model'       : ['Logistic Regression', 'Decision Tree', 'Linear Discriminant Analysis', 'Quadratic Discriminant Analysis', 'Random Forest', 'KNeighborsClassifier', 'Bayes'],
    'Fitting time': [LR_fit_time, dtree_fit_time, LDA_fit_time, QDA_fit_time, forest_fit_time, KNN_fit_time, bayes_fit_time],
    'Scoring time': [LR_score_time, dtree_score_time, LDA_score_time, QDA_score_time, forest_score_time, KNN_score_time, bayes_score_time],
    'Accuracy'    : [LR_accuracy, dtree_accuracy, LDA_accuracy, QDA_accuracy, forest_accuracy, KNN_accuracy, bayes_accuracy],
    'Precision'   : [LR_precision, dtree_precision, LDA_precision, QDA_precision, forest_precision, KNN_precision, bayes_precision],
    'Recall'      : [LR_recall, dtree_recall, LDA_recall, QDA_recall, forest_recall, KNN_recall, bayes_recall],
    'F1_score'    : [LR_f1, dtree_f1, LDA_f1, QDA_f1, forest_f1, KNN_f1, bayes_f1],
    #'AUC_ROC'     : [LR_roc, dtree_roc, SVM_roc, LDA_roc, QDA_roc, forest_roc, KNN_roc, bayes_roc],
    }, columns = ['Model', 'Fitting time', 'Scoring time', 'Accuracy', 'Precision', 'Recall', 'F1_score'])

display(models_initial.sort_values(by='Accuracy', ascending=False))
