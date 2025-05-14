# CS437Proj2025

Trash Detection in Water Channels Using YOLOv8 and CLIP

This repository contains the code and models for detecting trash floating on water channels, rivers, and urban drainage systems. The project leverages YOLOv8 and CLIP to detect various types of waste in urban waterways. The system integrates real-time object detection with multimodal transformer-based image-text alignment to enhance the detection of small, fragmented, or occluded debris.
Project Overview

Objective: Detect trash in water channels to help with environmental monitoring and urban waterway cleanliness.

Models Used: YOLOv8, CLIP-based Vision Transformer for rescoring.

Dataset: The "Trash in Water Channels" dataset (created and annotated by Dr. Murtaza Taj, 2021), containing various types of trash, including plastic bags, bottles, and wrappers in urban waterways.

Features

Real-Time Detection: YOLOv8 provides fast and accurate detection of trash in water channels.

Multimodal Transformer Integration: Uses CLIP to refine and rescore detections, particularly for small and occluded objects.

Dataset: A comprehensive dataset of trash in water channels with object-level annotations, ideal for training and evaluating object detection models.

Installation and Setup

To use this project, clone the repository and install the required dependencies.
Clone the Repository


    git clone https://github.com/ShariqKazmi25/CS437Proj2025.git
    cd CS437Proj2025

Install Dependencies

You can install the required Python dependencies by running:
   
    pip install -r requirements.txt

Data Setup

You need to download and set up the "Trash in Water Channels" dataset. Make sure to place the dataset in the correct directory as specified in the code.
Usage
Training the Model

To train the model, use the following command:

    python train.py --data "path_to_data.yaml" --batch-size 16 --epochs 20 --img-size 416 --lr 1e-4

    
Running Inference

After training, you can run inference on a set of images using the following command:

    python inference.py --model path_to_trained_model.pt --input path_to_input_image_or_directory --output path_to_save_inference_results


Results

The results of the final model on the Trash in Water Channels dataset can be found in the results/ folder. Detailed performance metrics and comparisons are included in the corresponding paper and evaluation section.

Acknowledgments

YOLOv8: Ultralytics' YOLOv8 for state-of-the-art object detection.

CLIP: OpenAI's CLIP for multimodal learning and rescoring.

Dataset: "Trash in Water Channels" dataset introduced by Dr. Murtaza Taj.
