from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data():
    
    cancer_data = load_breast_cancer()
    
    x = cancer_data.data
    y = cancer_data.target

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=7)

    return x_train, x_test, y_train, y_test