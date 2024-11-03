![image](https://github.com/user-attachments/assets/12d98045-b18f-4243-8ee7-d720f8a95ceb)


# SIRP (Satellite Image Road Prediction) [W.I.P, Readme will be updated soon]

**SIRP** is a project aimed at detecting roads in high-resolution satellite images using an Artificial Neural Network (ANN). This project automates road detection, making it easier to support geographic applications like mapping, infrastructure planning, and Geographic Information Systems (GIS).

## Project Overview

SIRP utilizes a backpropagation-based neural network model trained to distinguish roads from non-road areas in satellite images. By feeding the model with normalized RGB values from 3x3 pixel windows, it learns to identify visual characteristics of roads across different geographic regions.

### Key Features
- **Data Preprocessing**: Normalizes RGB values and extracts 3x3 pixel windows for each pixel in satellite images.
- **ANN Model**: Simple but effective neural network for binary classification (road vs. non-road).
- **Training and Testing**: Trains on labeled satellite data where roads are marked in white (1) and non-road areas in black (0).
- **Output Visualization**: Provides overlayed road predictions on satellite images for clear visualization of detected roads.

## Getting Started

### Prerequisites
- Python 3.x
- TensorFlow/Keras
- scikit-image
- numpy
- matplotlib

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/SIRP.git
    cd SIRP
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Folder Structure

```
project_folder/
│
├── road_map_converter.py
├── data_preparation.py
├── train.py
└── predict_and_visualize.py
```


### Usage

#### 1. Data Preparation
Prepare your satellite images and reference maps (label images) where:
   - **Roads** are marked in white (1)
   - **Non-road areas** are marked in black (0)

You don't need to mark black only white is enough. To mark black all other areas, use `road_map_converter.py`

Run `data_preparation.py` to process the images and save the features and labels for model training:

```python
# Example usage in data_preparation.py
prepare_and_save_data('path_to_images/', 'path_to_reference_map.png', 'road_data.npz')
```

#### 2. Training the Model
Use `train.py` to train the model on your prepared data:

```
python train.py
```

#### Predicting and Visualizing Results
Use `predict_and_visualize.py` to apply the trained model on new satellite images and visualize the road predictions:

```
python predict_and_visualize.py
```

![image](https://github.com/user-attachments/assets/2a132bc0-1118-4399-8a7f-1b84f53bb064)


### Contributing
Contributions are welcome! If you have ideas to improve this project or encounter any issues, please open an issue or submit a pull request.

### License
This project is licensed under the MIT License

### Acknowledgments
- This project was inspired by the paper [Road detection from high-resolution satellite images using artificial neural networkss](https://www.sciencedirect.com/science/article/abs/pii/S0303243406000171) by M. Mokhtarzade.
