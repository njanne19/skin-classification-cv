#!/bin/bash 

# Immediately quit if an error occurs 
set -e

# Create parent directory for datasets
DATASET_DIR="datasets/"

# Create parent directory if it doesn't exist
if [ ! -d "$DATASET_DIR" ]; then
    # Create the directory
    mkdir "${DATASET_DIR%/}"
fi

# Create directory for individual datasets 
# (more to be added later as we develop them) 
HAM_10000_DATASET_FOLDER="HAM_10000/"


##### HAM10000 DATASET #####
# Check to see if the datset folder already exists
# If it does, do nothing. If it doesn't, create the folder 
# and download the dataset
if [ ! -d "$DATASET_DIR$HAM_10000_DATASET_FOLDER" ]; then
    # Create the directory
    echo "Creating HAM10000 dataset folder..."
    mkdir "$DATASET_DIR${HAM_10000_DATASET_FOLDER%/}"

    # Download the first part of the dataset
    echo "Downloading HAM10000_images_part_1.zip ..."
    wget --show-progress --progress=bar:force https://dataverse.harvard.edu/api/access/datafile/3172585 -O HAM10000_images_part_1.zip

    # Unpack part 1
    echo "Unpacking HAM10000_images_part_1.zip ..."
    unzip -q HAM10000_images_part_1.zip -d "$DATASET_DIR$HAM_10000_DATASET_FOLDER"

    # Download the second part of the dataset
    echo "Downloading HAM10000_images_part_2.zip ..."
    wget --show-progress --progress=bar:force https://dataverse.harvard.edu/api/access/datafile/3172584 -O HAM10000_images_part_2.zip

    # Unpack part 2
    echo "Unpacking HAM10000_images_part_2.zip ..."
    unzip -q HAM10000_images_part_2.zip -d "$DATASET_DIR$HAM_10000_DATASET_FOLDER"

    echo "Cleaning up zip files..." 
    rm HAM10000_images_part_1.zip HAM10000_images_part_2.zip

    # Download the metadata
    echo "Downloading HAM10000_metadata.csv ..."
    wget --show-progress --progress=bar:force https://dataverse.harvard.edu/api/access/datafile/3172582?format=original -O "$DATASET_DIR$HAM_10000_DATASET_FOLDER/HAM10000_metadata.csv"

    # Check to see if the images have been downloaded correctly
    # Count the number of jpgs in the folder we unzipped files to 
    # and compare it to the number of jpgs in the metadata file
    echo "Checking to see if the dataset was downloaded correctly..."
    NUM_IMAGES=$(ls "$DATASET_DIR$HAM_10000_DATASET_FOLDER" | grep -c jpg)
    echo "Found $NUM_IMAGES images in the HAM_10000 dataset folder." 
    if [ "$NUM_IMAGES" -eq 10015 ]; then
        echo "The dataset was downloaded correctly!"
    else
        echo "The dataset was not downloaded correctly. Please try again."
        exit 1
    fi

else
    echo "HAM10000 dataset folder found!" 
    echo "Skipping download..."

fi

# Check to see if the dataset has been installed already,
# if it hasn't then download it
