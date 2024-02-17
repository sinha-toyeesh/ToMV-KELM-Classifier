# ToMV Leaf Classification

This project implements a machine learning model using image texture features to classify Tomato Mosaic Virus (ToMV) disease in tomato leaves.

## Project Highlights:

- **Accuracy:** Achieves over 99% accuracy on the test set.
- **Model:** Kernel Extreme Learning Machine (KELM) with RBF kernel.
- **Features:** Gray level co-occurrence matrix (GLCM) features and color histograms.

## Screeshots

![image](https://github.com/sinha-toyeesh/ToMV-KELM-Classifier/assets/64722289/bed66f1c-4345-42ec-b362-68b9b358d9e5)


## Requirements:

- Python 3.6+
- Streamlit
- OpenCV
- NumPy
- scikit-image
- Pickle

## Getting Started:

1. **Clone:** `git clone https://github.com/your-username/ToMV-KELM-Classifier.git`
2. **Install:** `pip install -r requirements.txt`
3. **Download:** Place pre-trained models (`finalized_train_data.pkl` and `finalized_model.pkl`) in the `Model_Files` directory.
4. **Run:** `streamlit run main.py`

## Using the Application:

1. Upload a tomato leaf image.
2. Click "Get Classification".
3. The model predicts the leaf health status (Healthy or Diseased).

## Model Description:

The KELM model utilizes an RBF kernel. It extracts texture features from images and predicts health status based on those features.

## Disclaimer:

Prototype model not intended for commercial use.
