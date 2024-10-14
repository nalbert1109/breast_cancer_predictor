import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib


def train_logistic_regression(x_train, y_train):
    
    model_lr = LogisticRegression(random_state=7) 
    model_lr.fit(x_train, y_train)
    return model_lr

def train_random_forest(x_train, y_train):
    
    model_rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=7) 
    model_rf.fit(x_train, y_train)
    return model_rf

def train_svm(x_train, y_train):
    
    model_svm = SVC(kernel='rbf', random_state=7) 
    model_svm.fit(x_train, y_train)
    return model_svm

def evaluate_model(model, x_test, y_test):
    
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    return accuracy, matrix

def save_model(model,model_name):
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, f'models/{model_name}.pkl')
    print(f'{model_name} saved sucessfully.')

def load_model(model_name):
    
    model = joblib.load(f'models/{model_name}.pkl')
    print(f'{model_name} loaded sucessfully.')
    return model
