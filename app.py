import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.title('Neural Network Graph')

df = pd.read_csv("dataset/9.random.csv")
df = df.rename(columns={'3.74E-01': 'x1', "1.08E+00": "x2", "0.00E+00": 'y'})

def generate_dataset():
    X = df[['x1', 'x2']]
    y = df['y'].iloc[X.index]
    return X.values, y.values

def plot_original_data(X, y):
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k', alpha=0.6)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Original Data Scatter Plot')
    st.pyplot(fig)

def plot_classification_results(X, y, model):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, cmap='bwr', alpha=0.6)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k', alpha=0.6)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Classification Results')
    fig.colorbar(ax.contourf(xx, yy, Z, cmap='bwr', alpha=0.6))
    st.pyplot(fig)

st.sidebar.title('Neural Network Playground')
activation_function = st.sidebar.selectbox('Activation Function', ['relu', 'sigmoid', 'tanh'])
optimizer = st.sidebar.selectbox('Optimizer', ['adam', 'sgd', 'rmsprop'])
random_state = st.sidebar.slider('Random State', min_value=0, max_value=100, value=42)
num_epochs = st.sidebar.slider('Number of Epochs', min_value=1, max_value=100, value=20)
num_dense_layers = st.sidebar.slider('Number of Dense Layers', min_value=1, max_value=5, value=3)
num_units_hidden_layers = [st.sidebar.slider(f'Units in Hidden Layer {i}', min_value=2, max_value=100, value=10) for i in range(num_dense_layers)]

if st.sidebar.button('Submit'):
    X, y = generate_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state)

    plot_original_data(X_train, y_train)

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(X_train_scaled.shape[1],)))
    for i in range(num_dense_layers):
        if i == num_dense_layers - 1:
            model.add(tf.keras.layers.Dense(num_units_hidden_layers[i], activation=activation_function, use_bias=True))
        else:
            model.add(tf.keras.layers.Dense(num_units_hidden_layers[i], activation=activation_function, use_bias=True))

    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.fit(X_train_scaled, y_train, epochs=num_epochs, verbose=0)

    # Evaluate model on test data
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    st.write(f'Test Accuracy: {test_accuracy}')

    plot_classification_results(X_train_scaled, y_train, model)
