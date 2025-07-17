import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import set_config, model_selection, preprocessing, linear_model, svm, ensemble, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
df = pd.read_csv("train.csv")
st.title("Projet de classification binaire Titanic")
st.sidebar.title("Sommaire")
pages = ["Exploration", "DataVizualization", "Modélisation"]
page = st.sidebar.radio("Aller vers", pages)
if page == pages[0]:
    st.write("### Introduction")
    st.dataframe(df.head(n = 10))
    st.write(df.shape)
    st.dataframe(df.describe())
    if st.checkbox("Afficher les NA:"):
        st.dataframe(df.isna().sum())
if page == pages[1]:
    st.write("### DataVizualization")
    fig = plt.figure()
    sns.countplot(x = df["Survived"])
    st.pyplot(fig)
    fig = plt.figure()
    sns.countplot(x = df["Sex"])
    plt.title("Répartition du genre des passagers")
    st.pyplot(fig)
    fig = plt.figure()
    sns.countplot(x = df["Pclass"])
    plt.title("Répartition des classes des passagers")
    st.pyplot(fig)
    fig = sns.displot(x = df["Age"], kind = "hist")
    plt.title("Distribution de l'âge des passagers")
    st.pyplot(fig)
    fig = plt.figure()
    sns.countplot(x = df["Survived"], hue = df["Sex"])
    st.pyplot(fig)
    fig = plt.figure()
    sns.pointplot(x = df["Pclass"], y = df["Survived"])
    st.pyplot(fig)
    fig = sns.lmplot(data = df, x = "Age", y = "Survived", hue = "Pclass")
    st.pyplot(fig)
    fig = plt.figure()
    df = df.select_dtypes(include = ["int64", "float64"])
    sns.heatmap(data = df.drop(columns = ["PassengerId"]).corr())
    st.pyplot(fig)
if page == pages[2]:
    st.write("### Modélisation")
new_df = df.drop(columns = ["PassengerId", "Name", "Cabin", "Ticket"])
y = new_df["Survived"]
X_cat = new_df[["Pclass", "Sex", "Embarked"]]
X_num = new_df[["Age", "SibSp", "Parch", "Fare"]]
X_cat = X_cat.apply(lambda col: col.fillna(col.mode()[0]), axis = 0)
X_num = X_num.apply(lambda col: col.fillna(col.median()), axis = 0)
#for col in X_num.columns:
    #X_num[col] = X_num[col].fillna(X_num[col].median())
    #for col in X_cat.columns:
    #X_cat[col] = X_cat[col].fillna(X_cat[col].mode()[0])
X_num = X_num.apply(lambda col: col.fillna(col.median()), axis = 0)
X_cat_scaled = pd.get_dummies(data = X_cat, columns = X_cat.columns, prefix = X_cat.columns)
X_clean = pd.concat([X_cat_scaled, X_num], axis = 1)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_clean, y, train_size = 0.8, test_size = 0.2, shuffle = True, random_state = 123)
scaler = preprocessing.StandardScaler()
X_train[X_num.columns] = scaler.fit_transform(X_train[X_num.columns])
X_test[X_num.columns] = scaler.transform(X_test[X_num.columns])
def prediction(classifier_name):
    if classifier_name == "Random Forest":
        clf = ensemble.RandomForestClassifier()
    elif classifier_name == "SVC":
        clf = svm.SVC()
    elif classifier_name == "Logistic Regression":
        clf = linear_model.LogisticRegression()
    clf.fit(X_train, y_train)
    return clf
def scores(clf, choice):
    if choice == "Accuracy":
        return clf.score(X_test, y_test)
    elif choice == "Confusion Matrix":
        return metrics.confusion_matrix(y_test, clf.predict(X_test))
if page == pages[2]:
    choix = ["Random Forest", "SVC", "Logistic Regression"]
    option = st.selectbox("Choix du Modèle", choix)
    st.write("Le modèle choisi est :", option)
    clf = prediction(option)
    display = st.radio("Que souhaitez-vous montrer?", ("Accuracy", "Confusion Matrix"))
    if display == "Accuracy":
        st.write(scores(clf, display))
    elif display == "Confusion Matrix":
        st.dataframe(scores(clf, display))
