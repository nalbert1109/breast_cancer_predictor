from src.data_processing import load_and_preprocess_data
import src.model as model
from src.visualize import plot_confusion_matrix

def main():
    
    x_train, x_test, y_train, y_test = load_and_preprocess_data()

    model_lr = model.train_logistic_regression(x_train, y_train)
    model.save_model(model_lr, 'logistic_regression_model')

    model_rf = model.train_random_forest(x_train, y_train)
    model.save_model(model_rf, 'random_forest_model')

    model_svm = model.train_svm(x_train, y_train)
    model.save_model(model_svm, 'svm_model')  

    for model_name in ['logistic_regression_model', 'random_forest_model', 'svm_model']:
        loaded_model = model.load_model(model_name)
        accuracy, matrix = model.evaluate_model(loaded_model, x_test, y_test)
        print(f'{model_name} Accuracy: {accuracy * 100:.2f}%')
        plot_confusion_matrix(matrix, model_name)

if __name__ == '__main__':
    main()