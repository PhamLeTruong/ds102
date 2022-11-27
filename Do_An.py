import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import plot_confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import openpyxl
from sklearn.naive_bayes import MultinomialNB
import time
import plotly.graph_objects as go
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

#Preprocessing Data
def Preprocessing(name):
  name = ' '.join(name.split())
  name = re.sub(r'([A-Z])\1+', lambda m: m.group(1).upper(), name, flags=re.I)
  name = name.lower()
  name = re.sub(r"[-()\\\"#/@;:<>{}`+=~|.!?,%/0123456789]", "", name)
  name = re.sub('\n', ' ', name)
  name = ' '.join(name.split()[1:])
  return name

def Train_model(model, option):
    if option==1:
        model.fit(X_train_cv, y_train)
        y_pred = model.predict(X_test_cv)
    else: 
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
    return y_pred

def Measure_model(y_test, y_pred):
    
    cls_report = classification_report(y_test, y_pred, output_dict=True)
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1_avg = f1_score(y_test, y_pred, average='macro')
    
    return acc, pre, recall, f1_avg

def Plot_table_measure(y_test, y_pred_cv, y_pred_tfidf):
    acc_cv, pre_cv, recall_cv, f1_cv_avg = Measure_model(y_test, y_pred_cv)
    acc_tfidf, pre_tfidf, recall_tfidf, f1_tfidf_avg = Measure_model(y_test, y_pred_tfidf)
    col_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    idx_names = ['CountVectorizer', 'TfidfVectorizer']
    values = [[acc_cv, pre_cv, recall_cv, f1_cv_avg],
            [acc_tfidf, pre_tfidf, recall_tfidf, f1_tfidf_avg]]
    table = pd.DataFrame(values, columns=col_names, index=idx_names)
    st.table(table)
    return [acc_cv, pre_cv, recall_cv, f1_cv_avg], [acc_tfidf, pre_tfidf, recall_tfidf, f1_tfidf_avg]
    
def Plot_confusion_matrix(model_cv, model_tfidf, model_name):
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(16,5))
    plot_confusion_matrix(model_cv, X_test_cv, y_test, ax=ax1,cmap='Blues')
    ax1.title.set_text(f'Count Vector + {model_name}')
    plot_confusion_matrix(model_tfidf, X_test_tfidf, y_test, ax=ax2,cmap='Blues')
    ax2.title.set_text(f'TF-IDF + {model_name}')
    st.pyplot(fig)
   
def Plot_bar_chart(measure_cv, measure_tfidf):
    x_label = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    countvector = go.Bar(x=x_label, y=measure_cv, name='CountVector')
    tfidfvector = go.Bar(x=x_label, y=measure_tfidf, name='TfidfVector')
    fig = go.Figure(data=[countvector, tfidfvector], layout=go.Layout(
            title='Measure of Model',
            template='plotly_white',
            barmode = 'group'))
    st.plotly_chart(fig)
st.title('Gender Prediction Based on Vietnamese Names with Machine Learning Techniques')

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(['Dataset', 'Naive Bayes', 'SVM', 'Logistic Regression', 'KNN', 'Decision Tree', 'RandomForest', 'VotingClassifier'])

# Dataset
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        df = pd.read_csv('Vietnamese_Names.csv', index_col=0)
        st.dataframe(df)
        
        ethnic  = pd.read_excel('UIT-ViNames/Data_ethnic.xlsx')
        data = pd.read_csv('UIT-ViNames/UIT-ViNames - Full.csv')
        
        data_female = pd.concat((data[data['Gender']==0].reset_index(drop=True), ethnic[ethnic['Gender']==0].reset_index(drop=True)), axis=0, ignore_index=True)
        data_male = pd.concat((data[data['Gender']==1].reset_index(drop=True), ethnic[ethnic['Gender']==1].reset_index(drop=True)), axis=0, ignore_index=True)
        
        X_train_female, X_test_female, y_train_female, y_test_female = train_test_split(data_female['Full_Names'], data_female['Gender'], test_size=0.3, random_state=42)
        X_train_male, X_test_male, y_train_male, y_test_male = train_test_split(data_male['Full_Names'], data_male['Gender'], test_size=0.3, random_state=42)
        
        X_train = pd.concat((X_train_female, X_train_male), ignore_index=True)
        X_test = pd.concat((X_test_female, X_test_male), ignore_index=True)
        y_train = pd.concat((y_train_female, y_train_male), ignore_index=True)
        y_test = pd.concat((y_test_female, y_test_male), ignore_index=True)
        
        # preprocessing for training set
        for i in range(len(X_train)):
            X_train[i] = Preprocessing(X_train[i])
            
        # preprocessing for test set
        for i in range(len(X_test)):
            X_test[i] = Preprocessing(X_test[i])
            
        # Use CountVectorizer to encode data
        cv = CountVectorizer()
        X_train_cv = cv.fit_transform(X_train)
        X_test_cv = cv.transform(X_test)
        
        # Use TfidfVectorizer to encode data
        tfidf = TfidfVectorizer()
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)
        
    with col2:
        # fig, ax = plt.subplots(figsize=(5,5))
        # sns.countplot(data=df, x='Gender')
        # st.pyplot(fig)
        gender_female = go.Bar(x=['Female', 'Male'], y=[len(data_female)], name='Female')
        gender_male = go.Bar(x=['Male'], y=[len(data_male)], name='Male')
        fig = go.Figure(data=[gender_female, gender_male], layout=go.Layout(barmode = 'group'))
        st.plotly_chart(fig)
        
    st.write(f'Giới tính nữ: {len(data_female)} ({round(len(data_female)/len(df) *100,2)}%)')
    st.write(f'Giới tính nam: {len(data_male)} ({round(len(data_male)/len(df) *100,2)}%)')
    st.write(f'Điểm dữ liệu: {len(df)}')

# Naive Bayes model
with tab2:
    st.header('Naive Bayes Model')
    # List parameters of Naive Bayes model
    st.subheader('Select parameter')
    alpha = st.number_input('alpha : float, default=1.0', min_value=0.0, max_value=1.0, step=0.1, value=1.0)
    fit_prior = st.radio('fit_prior : bool, default=True', ['True', 'False'])
    class_prior = st.text_input('class_prior : array-like of shape (n_classes,), default=None', value=None)
    if class_prior != 'None':
        class_prior = float(class_prior)
    else: class_prior = None
    
    NB_model_cv = MultinomialNB(alpha=alpha, fit_prior=fit_prior, class_prior=class_prior)
    NB_model_tfidf = MultinomialNB(alpha=alpha, fit_prior=fit_prior, class_prior=class_prior)

    button2 = st.button('Run')
    if button2:
        y_pred_cv = Train_model(NB_model_cv, option=1)
        y_pred_tfidf = Train_model(NB_model_tfidf, option=2)

        measure_cv, measure_tfidf = Plot_table_measure(y_test, y_pred_cv, y_pred_tfidf)
        
        Plot_confusion_matrix(NB_model_cv, NB_model_tfidf, model_name='Naive Bayes')

        Plot_bar_chart(measure_cv, measure_tfidf)
        
# SVM model
with tab3:
    st.header('Support Machine Vector Model')
    # List parameters of SVM model
    st.subheader('Select parameter')
    C = st.number_input('C : float, default=1.0', min_value=0.0, max_value=100.0, step=0.1, value=1.0)
    kernel = st.radio('kernel : {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} or callable, default=’rbf’', ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'], index=2)
    if kernel == 'poly':
        degree = st.number_input('degree : int, default=3', value=3)
    else: degree = 3
    gamma = st.radio('gamma : {‘scale’, ‘auto’} or float, default=’scale’', ['scale', 'auto', 'float'])
    if gamma == 'float':
        gamma = st.number_input('Enter value of gamma', min_value=0.0, max_value=1.0)
    max_iter = st.number_input('max_iter : int, default=-1', min_value=-1, max_value=1000, step=1, value=-1)
    random_state = st.text_input('random_state : int, RandomState instance or None, default=None', value='None')
    if random_state != 'None':
        random_state = int(random_state)
    else: random_state = None
    
    SVM_model_cv = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, max_iter=max_iter, random_state=random_state)
    SVM_model_tfidf = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, max_iter=max_iter, random_state=random_state)
        
    button = st.button('Run SVM model')
    if button: 
        y_pred_cv = Train_model(SVM_model_cv, option=1)
        y_pred_tfidf = Train_model(SVM_model_tfidf, option=2)

        measure_cv, measure_tfidf = Plot_table_measure(y_test, y_pred_cv, y_pred_tfidf)
        
        Plot_confusion_matrix(SVM_model_cv, SVM_model_tfidf, model_name='SVM')

        Plot_bar_chart(measure_cv, measure_tfidf)

# Logistic Regression model
with tab4:
    st.header('Logistic Regression Model')
    # List parameters of Logistic Regression model
    st.subheader('Select parameter')
    penalty = st.radio('penalty : {‘l1’, ‘l2’, ‘elasticnet’, ‘none’}, default=’l2’', ['l1', 'l2', 'elasticnet', 'none'], index=1)
    C = st.number_input('C : float, default=1.0', min_value=0.0, max_value=1000.0, step=0.1, value=1.0)
    fit_intercept = st.radio('fit_interceptbool, default=True', [True, False])
    random_state = st.text_input('random_state : int, RandomState instance, default=None', value='None')
    if random_state != 'None':
        random_state = int(random_state)
    else: random_state = None
    solver = st.radio('solver : {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, default=’lbfgs’', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], index=1)
    max_iter = st.number_input('max_iter : int, default=100', min_value=0, max_value=1000, step=1, value=100)
    button = st.button('Run Logistic Regression model')
    
    LR_model_cv = LogisticRegression(penalty=penalty, C=C, fit_intercept=fit_intercept, random_state=random_state, max_iter=max_iter)
    LR_model_tfidf = LogisticRegression(penalty=penalty, C=C, fit_intercept=fit_intercept, random_state=random_state, max_iter=max_iter)
    
    if button:
        y_pred_cv = Train_model(LR_model_cv, option=1)
        y_pred_tfidf = Train_model(LR_model_tfidf, option=2)

        measure_cv, measure_tfidf = Plot_table_measure(y_test, y_pred_cv, y_pred_tfidf)
        
        Plot_confusion_matrix(LR_model_cv, LR_model_tfidf, model_name='Logistic Regression')

        Plot_bar_chart(measure_cv, measure_tfidf)
        
# KNN model
with tab5:
    st.header('K-Nearest Neighbors Model')
    # List parameters of K-Nearest Neighbors model
    st.subheader('Select parameter')
    n_neighbors = st.number_input('n_neighbors : int, default=5', value=5)
    weight = st.radio('weights{‘uniform’, ‘distance’} or callable, default=’uniform’', ['uniform', 'distance'])
    algorithm = st.radio('algorithm{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’', ['auto', 'ball_tree', 'kd_tree', 'brute'])
    leaf = st.number_input('leaf_sizeint, default=30', value=30)
    p = st.number_input('p : int, default=2', value=2)
    n_job = st.text_input('n_jobs : int, default=None', value=None)
    if n_job != 'None':
        n_job = int(n_job)
    else: n_job = None
    
    KNN_model_cv = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weight, algorithm=algorithm, leaf_size=leaf, p=p, n_jobs=n_job)
    KNN_model_tfidf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weight, algorithm=algorithm, leaf_size=leaf, p=p, n_jobs=n_job)

    button = st.button('Run KNN model')
    if button:
        y_pred_cv = Train_model(KNN_model_cv, option=1)
        y_pred_tfidf = Train_model(KNN_model_tfidf, option=2)

        measure_cv, measure_tfidf = Plot_table_measure(y_test, y_pred_cv, y_pred_tfidf)
        
        Plot_confusion_matrix(KNN_model_cv, KNN_model_tfidf, model_name='Logistic Regression')

        Plot_bar_chart(measure_cv, measure_tfidf)

# Decision Tree model
with tab6:
    st.header('Decision Tree Model')
    # List parameters of Decision Tree model
    st.subheader('Select parameter')
    criterion = st.radio('criterion{“gini”, “entropy”, “log_loss”}, default=”gini”', ['gini', 'entropy', 'log_loss'])
    splitter = st.radio('splitter : {“best”, “random”}, default=”best”', ['best', 'random'])
    max_depth = st.text_input('max_dept : hint, default=None', value=None)
    if max_depth != 'None':
        max_depth = int(max_depth)
    else: max_depth = None
    min_samples_split = st.number_input('min_samples_split : int or float, default=2', value=2)
    min_samples_leaf = st.number_input('min_samples_leaf : int or float, default=1', value=1)
    min_weight_fraction_leaf = st.number_input('min_weight_fraction_leaf : float, default=0.0', min_value=0.0, max_value=1.0, value=0.0)
    max_features = st.radio('max_features : int, float or {“auto”, “sqrt”, “log2”}, default=None', ['int', 'float', 'auto', 'sqrt', 'log2', None], index=5)
    if max_features == 'int' or max_features == 'float':
        max_features = st.number_input('Enter value for max_features')
    random_state = st.text_input('random_state : int, RandomState instance or None, default=None', value=0, key='tab6_random_state')
    if random_state != 'None':
        random_state = int(random_state)
    else: random_state = None
    max_leaf_nodes = st.text_input('max_leaf_nodes : int, default=None', value=None)
    if max_leaf_nodes != 'None':
        max_leaf_nodes = int(random_state)
    else: max_leaf_nodes = None
    
    DT_model_cv = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth, 
                                              min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                              min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                              random_state=random_state, max_leaf_nodes=max_leaf_nodes)
    DT_model_tfidf = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth, 
                                              min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                              min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                              random_state=random_state, max_leaf_nodes=max_leaf_nodes)

    
    button = st.button('Run Decision Tree model')
    if button:
        y_pred_cv = Train_model(DT_model_cv, option=1)
        y_pred_tfidf = Train_model(DT_model_tfidf, option=2)

        measure_cv, measure_tfidf = Plot_table_measure(y_test, y_pred_cv, y_pred_tfidf)
        
        Plot_confusion_matrix(DT_model_cv, DT_model_tfidf, model_name='Decision Tree')

        Plot_bar_chart(measure_cv, measure_tfidf)
        
# Random Forest model
with tab7:
    haha=123
    st.header('Random Forest Model')
    # List parameters of Random Forest model
    st.subheader('Select parameter')
    n_estimators = st.number_input('n_estimators : int, default=100', value=100)
    criterion = st.radio('criterion{“gini”, “entropy”, “log_loss”}, default=”gini”', ['gini', 'entropy', 'log_loss'], key='randomforest')
    max_depth = st.text_input('max_depth : int, default=None', value=None)
    if max_depth != 'None':
        max_depth = int(max_depth)
    else: max_depth = None
    min_samples_split = st.number_input('min_samples_split : int or float, default=2', value=2, key='tab7_min_samples_split')
    min_samples_leaf = st.number_input('min_samples_leaf : int or float, default=1', value=1, key='tab7_min_samples_leaf')
    min_weight_fraction_leaf = st.number_input('min_weight_fraction_leaf : float, default=0.0', min_value=0.0, max_value=1.0, value=0.0, key='tab7_min_weight_fraction_leaf')
    max_features = st.radio('max_features : {“sqrt”, “log2”, None}, int or float, default=”sqrt”', ['sqrt', 'log2', None, 'int', 'float'])
    if max_features in ['int', 'float']:
        max_features = st.number_input('Enter value of max_features')
    max_leaf_nodes = st.text_input('max_leaf_nodes : int, default=None', value=None, key='tab7_max_leaf_nodes')
    if max_leaf_nodes != 'None':
        max_leaf_nodes = int(random_state)
    else: max_leaf_nodes = None
    bootstrap = st.radio('bootstrap : bool, default=True', [True, False])
    oob_score = st.radio('oob_score : bool, default=False', [True, False], index=1)
    n_jobs = st.text_input('n_jobs : int, default=None', value=None, key='tab7_n_jobs')
    if n_jobs != 'None':
        n_jobs = int(n_jobs)
    else: n_jobs = None
    random_state = st.text_input('random_state : int, RandomState instance or None, default=None', value=0, key='tab7_random_state')
    if random_state != 'None':
        random_state = int(random_state)
    else: random_state = None
    max_samples = st.radio('max_samples : int or float, default=None', ['int', 'float', None], index=2)
    if max_samples in ['int', 'float']:
        max_samples = st.number_input('Enter value of max_samples')
        
    RF_model_cv = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split,
                                             min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                             max_leaf_nodes=max_leaf_nodes, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, max_samples=max_samples)
    RF_model_tfidf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split,
                                             min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                             max_leaf_nodes=max_leaf_nodes, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, max_samples=max_samples)
    
    button = st.button('Run Randomforest model')
    if button:
        y_pred_cv = Train_model(RF_model_cv, option=1)
        y_pred_tfidf = Train_model(RF_model_tfidf, option=2)

        measure_cv, measure_tfidf = Plot_table_measure(y_test, y_pred_cv, y_pred_tfidf)
        
        Plot_confusion_matrix(RF_model_cv, RF_model_tfidf, model_name='RandomForest')

        Plot_bar_chart(measure_cv, measure_tfidf)
        
# Votingclassifier model
with tab8:
    st.write('Select 3 best model for VotingClassifier')
    select_model = [st.checkbox(name_model) for name_model in ['Naive Bayes', 'Logistic Regression', 'Support Vector Machine', 'K-Nearest Neighbors', 'Decision Tree', 'RandomForest']]
    
    if st.button('Run VotingClassifier'):
        name_model_cv = [NB_model_cv, LR_model_cv, SVM_model_cv, KNN_model_cv, DT_model_cv, RF_model_cv]
        name_model_tfidf = [NB_model_tfidf, LR_model_tfidf, SVM_model_tfidf, KNN_model_tfidf, DT_model_tfidf, RF_model_tfidf]
        columns = ['Naive Bayes', 'Logistic Regression', 'Support Vector Machine', 'K-Nearest Neighbors', 'Decision Tree', 'RandomForest']
        score_cv, score_tfidf, col, estimate, ii = [], [], [], [], []
        SVM_model_cv.probability=True
        SVM_model_tfidf.probability=True
        
        for i in range(len(select_model)):
            if select_model[i]:
                name_model_cv[i].fit(X_train_cv, y_train)
                name_model_tfidf[i].fit(X_train_tfidf, y_train)
                score_cv.append(round(f1_score(y_test, name_model_cv[i].predict(X_test_cv), average='macro'),4))
                score_tfidf.append(round(f1_score(y_test, name_model_tfidf[i].predict(X_test_tfidf), average='macro'),4))
                col.append(columns[i])
                
        ii = [i for i, val in enumerate(select_model) if val==True]
        weight_cv = [1+ index for index in np.argsort(score_cv)]
        weight_tfidf = [1+ index for index in np.argsort(score_tfidf)]
        
        voting_clf_cv = VotingClassifier(estimators=[('model1',name_model_cv[ii[0]]),('model2',name_model_cv[ii[1]]),('model3',name_model_cv[ii[2]])], voting='soft',weights=weight_cv)
        voting_clf_tfidf = VotingClassifier(estimators=[('model1',name_model_tfidf[ii[0]]),('model2',name_model_tfidf[ii[1]]),('model3',name_model_tfidf[ii[2]])], voting='soft',weights=weight_tfidf)
        voting_clf_cv.fit(X_train_cv, y_train)
        voting_clf_tfidf.fit(X_train_tfidf, y_train)
        col.append('VotingClassifier')
        
        score_cv.append(round(f1_score(y_test, voting_clf_cv.predict(X_test_cv), average='macro'),4))
        score_tfidf.append(round(f1_score(y_test, voting_clf_tfidf.predict(X_test_tfidf), average='macro'),4))
        score = ['F1-score (CountVector)', 'F1-score (TfidfVector)']
        st.table(pd.DataFrame([score_cv, score_tfidf], columns=col, index=score))
        
