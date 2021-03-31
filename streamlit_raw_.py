import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes, load_boston
import matplotlib.pyplot as plt

#---------------------------------#
# Page layout
# Page expands to full width
st.set_page_config(page_title='No BS only DS',
                   layout='wide')

#---------------------------------#
# Model building


def results(X_train, Y_train, X_test, Y_test, reg, model_name):
    st.header(model_name)
    st.markdown('**2.1. Training set**')
    Y_pred_train = reg.predict(X_train)
    if challenge == "Regression":
        st.write('Coefficient of determination ($R^2$):')
        st.info(r2_score(Y_train, Y_pred_train))
    elif challenge == "Classification":
        st.write('Accuracy:')
        st.info(metrics.accuracy_score(Y_train, Y_pred_train))

    st.write('Error (MSE or MAE):')
    st.info(mean_squared_error(Y_train, Y_pred_train))

    st.markdown('**2.2. Test set**')
    Y_pred_test = reg.predict(X_test)
    if challenge == "Regression":
        st.write('Coefficient of determination ($R^2$):')
        st.info(r2_score(Y_test, Y_pred_test))
    elif challenge == "Classification":
        st.write('Accuracy:')
        st.info(metrics.accuracy_score(Y_test, Y_pred_test))

    st.write('Error (MSE or MAE):')
    st.info(mean_squared_error(Y_test, Y_pred_test))

    st.subheader('3. Model Parameters')
    st.write(reg.get_params())
    # st.subheader('4. Predict')
    # predict = []
    # for col in X_train.columns:
    #     col
    #     predict.append(st.sidebar.number_input(f'Enter value for {col}',))
    # st.write(predict)

# Regression Algorithms


def linearReg(X_train, X_test, Y_train, Y_test, parameter_fit_intercept=True, parameter_normalize=True):
    # if "Multilinear Regression" in model:
    # with st.sidebar.subheader('2.2 Linear Regression Learning Parameters'):
    #     parameter_fit_intercept = st.sidebar.select_slider(
    #         'Fit intercept', options=[True, False])
    #     parameter_normalize = st.sidebar.select_slider(
    #         'Normalize regressors', options=[True, False])
    reg = LinearRegression(
        fit_intercept=parameter_fit_intercept, normalize=parameter_normalize)
    reg = reg.fit(X_train, Y_train)
    results(X_train, Y_train, X_test, Y_test, reg, "Multilinear Regression")


def SVR(X_train, X_test, Y_train, Y_test):
    reg = svm.SVR(kernel=parameter_kernel,
                  tol=parameter_tol, C=parameter_c)
    reg = reg.fit(X_train, Y_train)
    results(X_train, Y_train, X_test, Y_test, reg, "Support Vector Regression")


def neuralNet(X_train, X_test, Y_train, Y_test):
    reg = MLPRegressor(hidden_layer_sizes=parameter_hidden_layer_sizes, activation=parameter_activation,
                       solver=parameter_solver, alpha=parameter_alpha, learning_rate=parameter_learning_rate, random_state=1)
    reg = reg.fit(X_train, Y_train)
    results(X_train, Y_train, X_test, Y_test, reg, "Neural Network")


def randomForest(X_train, X_test, Y_train, Y_test):
    reg = RandomForestRegressor(
        n_estimators=parameter_n_estimators, random_state=1)
    reg = reg.fit(X_train, Y_train)
    results(X_train, Y_train, X_test, Y_test, reg, "Random Forest")

# Classification Algorithms


def logisticReg(X_train, X_test, Y_train, Y_test):
    # if "Multilinear Regression" in model:
    # with st.sidebar.subheader('2.2 Linear Regression Learning Parameters'):
    #     parameter_fit_intercept = st.sidebar.select_slider(
    #         'Fit intercept', options=[True, False])
    #     parameter_normalize = st.sidebar.select_slider(
    #         'Normalize regressors', options=[True, False])
    reg = LogisticRegression(C=parameter_C, random_state=1)
    reg = reg.fit(X_train, Y_train)
    results(X_train, Y_train, X_test, Y_test, reg, "Logistic Regression")


def KMeansClustering(X_train, X_test, Y_train, Y_test):
    reg = KMeans(n_clusters=parameter_n_clusters_kmean, random_state=1)
    reg = reg.fit(X_train, Y_train)
    results(X_train, Y_train, X_test, Y_test, reg, "K-Means Clustering")


def KNearestNeighbors(X_train, X_test, Y_train, Y_test):
    reg = KNeighborsClassifier(
        n_neighbors=parameter_n_neighbors, weights=parameter_weights)
    reg = reg.fit(X_train, Y_train)
    results(X_train, Y_train, X_test, Y_test, reg, "K-Nearest Neighbors")


def SVC(X_train, X_test, Y_train, Y_test):
    reg = svm.SVC(C=parameter_c_svmc, kernel=parameter_kernel, random_state=1)
    reg = reg.fit(X_train, Y_train)
    results(X_train, Y_train, X_test, Y_test,
            reg, "Support Vector Classification")


def neuralNetClassifier(X_train, X_test, Y_train, Y_test):
    reg = MLPClassifier(hidden_layer_sizes=parameter_hidden_layer_sizes_classifier, activation=parameter_activation_classifier,
                        solver=parameter_solver_classifier, alpha=parameter_alpha_classifier, learning_rate=parameter_learning_rate_classifier, random_state=1)
    reg = reg.fit(X_train, Y_train)
    results(X_train, Y_train, X_test, Y_test, reg, "Neural Network")


def randomForestClassifier(X_train, X_test, Y_train, Y_test):
    reg = RandomForestClassifier(
        n_estimators=parameter_n_estimators_classifier, random_state=1)
    reg = reg.fit(X_train, Y_train)
    results(X_train, Y_train, X_test, Y_test, reg, "Random Forest")


def XGBoostClassifier(X_train, X_test, Y_train, Y_test):
    reg = XGBClassifier()
    reg = reg.fit(X_train, Y_train)
    results(X_train, Y_train, X_test, Y_test, reg, "XGBoost")

 # Building Model


def build_model(df):
    cols = df.columns
    st.markdown('**Dependent Variable**:')
    dependent_variable = st.selectbox(
        "Choose the dependent variable", cols)
    indep_cols = []
    for col in cols:
        if col is not dependent_variable:
            indep_cols.append(col)
    st.markdown('**Independent Variables**:')
    independent_variables = st.multiselect(
        "Choose the independent variables", indep_cols)
    st.markdown('**Cross Validation**:')
    X = df[independent_variables]
    Y = df[dependent_variable]
    fig, ax = plt.subplots()
    ax.hist(Y)
    # st.pyplot(fig)

    for col in X.columns:
        if type(X[col][0]) is str:
            X[col] = pd.factorize(X[col])[0]
    if type(Y[0]) is str:
        Y = pd.factorize(Y)[0]
        Y = pd.DataFrame(Y)
    st.write(Y.head())
    st.write(X.head())
    st.markdown('**Variable details**:')
    st.write('X variable')
    st.info(list(X.columns))
    st.write('Y variable')
    st.info(dependent_variable)

    # Data splitting
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, train_size=int(float(split_size)/100 * len(X)))

    st.markdown('**Data splits**')
    st.write('Training set')
    st.info(X_train.shape)
    st.write('Test set')
    st.info(X_test.shape)
    if challenge == "Regression":
        if "Multilinear Regression" in model:
            linearReg(X_train, X_test, Y_train, Y_test)
            # linearReg(X_train, X_test, Y_train, Y_test,
            #           parameter_fit_intercept, parameter_normalize)
        if "Support Vector Regression" in model:
            SVR(X_train, X_test, Y_train, Y_test)
        if "Neural Network" in model:
            neuralNet(X_train, X_test, Y_train, Y_test)
        if "Random Forest" in model:
            randomForest(X_train, X_test, Y_train, Y_test)

    elif challenge == "Classification":
        model
        if "Logistic Regression" in model:
            logisticReg(X_train, X_test, Y_train, Y_test)
        if "K-Means Clustering" in model:
            KMeansClustering(X_train, X_test, Y_train, Y_test)
        if "K-Nearest Neighbors" in model:
            KNearestNeighbors(X_train, X_test, Y_train, Y_test)
        if "Support Vector Machine" in model:
            SVC(X_train, X_test, Y_train, Y_test)
        if "Neural Network" in model:
            neuralNetClassifier(X_train, X_test, Y_train, Y_test)
        if "Random Forest" in model:
            randomForestClassifier(X_train, X_test, Y_train, Y_test)
        if "XGBoost" in model:
            XGBoostClassifier(X_train, X_test, Y_train, Y_test)


#---------------------------------#
st.write("""
# The Easy DS Tool
Deploy data science models and analysis quickly with this tool. Start by uploading a dataset, selecting the type of challenge, choose your intended models and explore. Compare several different model performances at once to find the optimal method for solving your
problem.
""")
uploaded_file = None


#---------------------------------#
# Sidebar - Collects user input features into dataframe
uploaded_file = None
with st.sidebar.header('Pre-requisites'):
    challenge = st.selectbox(
        'What type of data science challenge is this?',
        [None, "Regression", "Classification", "Time Series Analysis", "Computer Science"])

if challenge is not None:
    with st.sidebar.header('1. Upload your CSV data'):
        uploaded_file = st.sidebar.file_uploader(
            "Upload your input CSV file", type=["csv"])
        st.sidebar.markdown("""
    [Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
    """)

    # Sidebar - Regression
    if challenge == "Regression":
        with st.sidebar.header('2.0 Models'):
            model = st.sidebar.multiselect("Which regression modeling strategy do you want to use?",
                                           ("Multilinear Regression", "Support Vector Regression", "Neural Network", "Random Forest", "All of the above"))
        if model is not None:
            with st.sidebar.header('2.1 Set Parameters'):
                split_size = st.sidebar.slider(
                    'Data split ratio (% for Training Set)', 10, 95, 80, 5)
            # Sidebar - Multilinear Regression
            with st.sidebar.header('2.2 Set Cross Validation Folds'):
                cross_valid = st.slider(
                    'Number of folds', 1, 20, 10, 1)
            if "Multilinear Regression" in model:
                with st.sidebar.subheader('Linear Regression Learning Parameters'):
                    parameter_fit_intercept = st.sidebar.select_slider(
                        'Fit intercept', options=[True, False])
                    parameter_normalize = st.sidebar.select_slider(
                        'Normalize regressors', options=[True, False])
            # Sidebar - Multilinear Regression
            if "Support Vector Regression" in model:
                with st.sidebar.subheader('SVR Learning Parameters'):
                    parameter_kernel = st.sidebar.selectbox(
                        'Select', ["linear", "poly", "rbf", "sigmoid", "precomputed"])
                    parameter_tol = st.sidebar.slider(
                        'Tolerance for stopping criterion', 0.001, .1, .001, .001)
                    parameter_c = st.sidebar.slider(
                        'Regularization parameter', .1, 1.0, .1, .1)
            # Sidebar - Multilinear Regression
            if "Neural Network" in model:
                with st.sidebar.subheader('Neural Network Learning Parameters'):
                    parameter_hidden_layer_sizes = st.sidebar.slider(
                        'Number of Hidden Layers', 10, 500, 100, 5)
                    parameter_activation = st.sidebar.selectbox(
                        'Activation Layer', ["identity", "logistic", "tanh", "relu"], index=3)
                    parameter_solver = st.sidebar.selectbox(
                        'Weight Optimizer', ["adam", "sgd", "lbfgs"], index=0)
                    parameter_alpha = st.sidebar.number_input('Enter alpha')
                    parameter_learning_rate = st.sidebar.selectbox(
                        'Learning Rate', ["constant", "invscaling", "adaptive"], index=0)
            # Sidebar - Multilinear Regression
            if "Random Forest" in model:
                with st.sidebar.subheader('Random Forest Learning Parameters'):
                    parameter_n_estimators = st.sidebar.slider(
                        'Number of Trees', 10, 500, 100, 5)

    elif challenge == "Classification":
        # Sidebar - Specify parameter settings
        # with st.sidebar.header('2. Set Parameters'):
        #     split_size = st.sidebar.slider(
        #         'Data split ratio (% for Training Set)', 10, 90, 80, 5)

        with st.sidebar.header('2.0 Models'):
            model = st.sidebar.multiselect("Which classificaiton modeling strategy do you want to use?",
                                           ("Logistic Regression", "K-Means Clustering", "K-Nearest Neighbors", "Support Vector Machine", "Neural Network", "Random Forest", "XGBoost"))
        if model is not None:
            with st.sidebar.header('2.1 Set Parameters'):
                split_size = st.sidebar.slider(
                    'Data split ratio (% for Training Set)', 10, 95, 80, 5)
            # Sidebar - Logistic Regression
            if "Logistic Regression" in model:
                with st.sidebar.subheader('Logistic Regression Learning Parameters'):
                    parameter_C = st.sidebar.slider(
                        'Regularization', 0, 10, 1, 1)
            # Sidebar - K-Means Clustering
            if "K-Means Clustering" in model:
                with st.sidebar.subheader('K-Means Clustering Learning Parameters'):
                    parameter_n_clusters_kmean = st.sidebar.slider(
                        'Number of Clusters', 1, 40, 8, 1)
            # Sidebar - K-Nearest Neighbors
            if "K-Nearest Neighbors" in model:
                with st.sidebar.subheader('K-Nearest Neighbors Learning Parameters'):
                    parameter_n_neighbors = st.sidebar.slider(
                        'Number of Neighbors', 1, 75, 5, 1)
                    parameter_weights = st.sidebar.select_slider(
                        'Weight Applied', options=["uniform", "distance"])

            # Sidebar - Support Vector Machine
            if "Support Vector Machine" in model:
                with st.sidebar.subheader('SVM Learning Parameters'):
                    parameter_kernel = st.sidebar.selectbox(
                        'Kernel', ["linear", "poly", "rbf", "sigmoid", "precomputed"])
                    parameter_c_svmc = st.sidebar.slider(
                        'Regularization Parameter', 0.0, 10.0, .1, 1.0)
            # Sidebar - Multilinear Regression
            if "Neural Network" in model:
                with st.sidebar.subheader('Neural Network Learning Parameters'):
                    parameter_hidden_layer_sizes_classifier = st.sidebar.slider(
                        'Number of Hidden Layers', 10, 500, 100, 5)
                    parameter_activation_classifier = st.sidebar.selectbox(
                        'Activation Layer', ["identity", "logistic", "tanh", "relu"], index=3)
                    parameter_solver_classifier = st.sidebar.selectbox(
                        'Weight Optimizer', ["adam", "sgd", "lbfgs"], index=0)
                    parameter_alpha_classifier = st.sidebar.number_input(
                        'Enter alpha')
                    parameter_learning_rate_classifier = st.sidebar.selectbox(
                        'Learning Rate', ["constant", "invscaling", "adaptive"], index=0)
            # Sidebar - Random Forest
            if "Random Forest" in model:
                with st.sidebar.subheader('Random Forest Learning Parameters'):
                    parameter_n_estimators_classifier = st.sidebar.slider(
                        'Number of Trees', 10, 500, 100, 5)
            # Sidebar - XGBoost
            # if "XGBoost" in model:
#---------------------------------#
# Main panel

# Displays the dataset
st.subheader('1. Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(df)
    build_model(df)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        # Diabetes dataset
        # diabetes = load_diabetes()
        # X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        # Y = pd.Series(diabetes.target, name='response')
        # df = pd.concat( [X,Y], axis=1 )

        # st.markdown('The Diabetes dataset is used as the example.')
        # st.write(df.head(5))

        # Boston housing dataset
        boston = load_boston()
        X = pd.DataFrame(boston.data, columns=boston.feature_names)
        Y = pd.Series(boston.target, name='response')
        df = pd.concat([X, Y], axis=1)

        st.markdown('The Boston housing dataset is used as the example.')
        st.write(df)

        build_model(df)
