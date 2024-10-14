import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(matrix,model_name):
    
    plt.figure(figsize=(8,6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap='Blues',
                xticklabels=['Malignant', 'Benign'],
                yticklabels=['Malignant', 'Benign'])
    plt.title(f'Confusion Matrix for {model_name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'confusion_matrix_{model_name}.png')
    plt.close() 