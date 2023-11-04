# Visual Skin Classification Using Python

## Installation and Setup
1. Clone the repository
```
git clone git@github.com:njanne19/skin-classification-cv.git
```
2. Install the required packages (preferably in a [virtual environment](https://docs.python.org/3/library/venv.html))
```
pip install -r requirements.txt
```
3. Install this repository as a local, editable python package
```
pip install -e .
```
4. Download the dataset from [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) and extract the contents into the `data` directory. The directory structure should look like this:
```
data
├── HAM10000_images_part_1
├── HAM10000_images_part_2
├── HAM10000_metadata.csv
└── hmnist_28_28_RGB.csv
```
5. Run the `test_config.py` script to verify everything has been installed correctly. 
```
python test_config.py
```

## Introduction
This project is a collection of implementations of various machine learning algorithms for skin classification. The goal is to classify images of skin lesions from the [HAM10000 dataset](https://paperswithcode.com/dataset/ham10000-1#:~:text=HAM10000%20is%20a%20dataset%20of,for%20detecting%20pigmented%20skin%20lesions.) into their assigned diagnostic labels. The dataset contains 10015 images of skin lesions, each of which is labeled with one of seven diagnostic categories. The images are in the JPEG format and have a resolution of 600x450 pixels. The dataset is available for download [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T).

