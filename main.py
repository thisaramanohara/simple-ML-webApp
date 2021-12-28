import streamlit as st
from sklearn import datasets

st.title('Streamlit example')

st.write("""
# Explore different classifier
Choose the best one !!!
""")

dataset_name = st.sidebar.selectbox('Select Dataset', ('Iris', 'Breast Cancer', 'Wine Dataset'))
classifier_name = st.sidebar.selectbox('Select Classifier', ('KNN', 'SVM', 'Random Forest'))

def get_dataset(dataset_name):
    if dataset_name == 'Iris':
        data = datasets.load_iris()
    elif dataset_name == 'Breast Cancer':
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()

    x = data.data
    y = data.target

    return x, y

