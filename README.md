# Breast Cancer Predictor

## Description
This project utilizes logistic regression , random forest classfication, and support vector machines to classify whether a tumor is malignant or benign. Each model is written to save each time and can be loaded easily. The accuracy of each of the models are compared against each other to determine which is the most fitting and successful at predicting breast cancer malignancy.

The dataset used for this model is the Breast Cancer Wisconsin dataset that is publically available in sklearn as well as UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

Data includes 30 features from 569 patients. 

According to UCI Machine Learning repository, the features were "computed from a digitized image of a fine needle aspirate(FNA) of a breast mass" and "describe characterteristics of the cell nuclei"(Wolberg et. al 1993) 

Ten of these features were computed from each cell nucleus:

	a) radius (mean of distances from center to points on the perimeter)
	b) texture (standard deviation of gray-scale values)
	c) perimeter
	d) area
	e) smoothness (local variation in radius lengths)
	f) compactness (perimeter^2 / area - 1.0)
	g) concavity (severity of concave portions of the contour)
	h) concave points (number of concave portions of the contour)
	i) symmetry 
	j) fractal dimension ("coastline approximation" - 1)


## Table of Contents
   
    - [Background](#background)
    - [Installation](#installation)
    - [Usage](#usage)
    - [Contributing](#contributing)
    - [License](#license)
    
## Background

Having worked at a biotech company that worked towards creating drugs to treat cancer, I was really interested in working on a project that could intertwine the world of biology with machine learning while being able to learn a lot about the different machine learning algorithms. With breast cancer being so prevalent and having vast accessible data, it seemed fitting to create a machine learning model to predict tumor malignancy. Cancer is often a multi-stage and complex illness, so a multi-classification problem such as identifying what stage a tumor is at (None, Stage 1, Stage 2, Stage 3, Stage 4) would probably be better suited in practicality. However, for the sake of learning and simplicity, I decided to use a binary classification to determine whether or not a tumor was malignant or benign. 

## Installation

1. Clone the repository:
   ...console
   git clone https://github.com/nalbert1109/breast_cancer_predictor.git
   ...
2. Enter project directory:
   ...console
   cd breast_cancer_predictor
   ...
3. Create a virtual environment:
   ...console
   python -m venv venv
   ...
4. Activate the virtual environment:
   ...console
   activate //on Windows, not sure what it is on MacOS/Linux
   ...
5. Install the required packages into the virtual environment:
   ...console
   pip install -r requirements.txt
   ...
## Usage

Run the application:
...console
python main.py
...

main.py imports methods from the src folder which contains data_processing.py, model.py, visualize.py
It will load the breast cancer data from sklearn, so you do not have to read a csv file. It will process the data, train it, and graph a heatmap(confusion matrix) of each of the three different models for you to visualize. The light blue square is a True Positive while the dark blue is a True Negative. An additional output will be the accuracy. The model is automatically saved and loaded as you will see in the methods of model.py.

## Contributing 

If you are interested in contributing, feel free to contact me! The project was designed to be fairly modular so that future additions could be made. Some next steps would be to try implementing CNNs since there's academic papers suggesting CNN's are the most accurate when it comes to predicting breast cancer(https://www.nature.com/articles/s41598-019-48995-4#Sec2). The SVM and random forest also do not have any hyperparameters tuned so that could be something to look into. Another idea would be to try a different data set for another disorder that has been implemented in some academic papers before such as: diabetic retinopathy or CT-scans/MRI of brain tumors. A grand final idea would be to have a suite of different models for different disorders all packaged together as one software that can be deployed as a full stack application to be used by medical professionals.

LinkedIn: linkedin.com/in/adn1109
Email: nalbert1109@gmail.com

## License

This project is licensed under the MIT license. https://www.mit.edu/~amini/LICENSE.md
