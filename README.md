## Iris Classifier Project
This project trains a Decision Tree and k-NN classifieron the Iris dataset using Python.  
It demonstrates basic machine learning workflow: data loading, training models, evaluating accuracy, and saving outputs.
## Project Structure
iris-classifier/
├── notebooks/ # Jupyter notebook with code
│ └── iris_classifier.ipynb
├── src/ # Python script
│ └── train.py
├── outputs/ # Generated model and confusion matrix 
├── requirements.txt # Python dependencies
├── README.md # Project instructions
└── .gitignore # Files to ignore (venv, outputs, etc.)
## How to Run

1. **Clone the repository:**
git clone https://github.com/Rupinder108/iris-classifier.git
cd iris-classifier
2.**Create a virtual environment:**
python -m venv venv 
3.**Activate the virtual environment:**
venv\Scripts\activate
4.**Install dependencies:**
pip install -r requirements.txt
5.**Run the training script:**
python src/train.py
##This will generate:
outputs/confusion_matrix.png → Confusion matrix
outputs/model.joblib → Trained Decision Tree model##
