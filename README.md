# Visual Skin Classification Using Python

## Installation and Setup
1. Clone the repository
```
git clone git@github.com:njanne19/skin-classification-cv.git
```
2. Install the required packages (preferably in a [virtual environment](https://docs.python.org/3/library/venv.html))
```
pythom -m venv env
source ./env/bin/activate
pip install -r requirements.txt
```
3. Install this repository as a local, editable python package. This will allow you to run commands like `python skin-classification-cv/kmeans', for example. 
```
pip install -e .
```
4. Run the dataset installation script
```
./install_dataset.sh
```
**Note:** If you receive an error that permission is denied, retry running the command with `sudo`. If you receive an error that `./install_dataset.sh` is not a command, mark the file as executible with `sudo chmod +x ./install_dataset.sh` and try again. 

At the end of this procedure, you should have the following file structure: 

```
skin-classification-cv/
├─ datasets/
│  ├─ HAM_10000/
│  │  ├─ HAM10000_metadata.csv
│  │  ├─ ISIC_0024306.jpg
│  │  ├─ ISIC_0024307.jpg
│  │  ├─ ... (all other images) 
├─ skin-classification-cv (where our code will go)/
├─ ... (all other package files, like setup.py)
```

## Introduction
This project is a collection of implementations of various machine learning algorithms for skin classification. The goal is to classify images of skin lesions from the [HAM10000 dataset](https://paperswithcode.com/dataset/ham10000-1#:~:text=HAM10000%20is%20a%20dataset%20of,for%20detecting%20pigmented%20skin%20lesions.) into their assigned diagnostic labels. The dataset contains 10015 images of skin lesions, each of which is labeled with one of seven diagnostic categories. The images are in the JPEG format and have a resolution of 600x450 pixels. The dataset is available for download [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T).

