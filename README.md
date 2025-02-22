# Face Recognition API with FastAPI

=====================================

## Overview

This project is a Face Recognition API built using FastAPI and face_recognition. It allows users to upload images, encode faces, and match them against a database of stored face encodings. The matched results are saved in a separate folder. I have also added a ipynb file to test in console.

## Features

-> Upload multiple images to store face encodings.
-> Upload a target image to find matching faces.
-> Uses SQLite to store face encodings.
-> Automatically cleans up processed images and resets the database after each request.
-> Built with FastAPI for high performance.

## Project Structure

📁 project_root/
│── main.py # FastAPI application
│── image_encoder_and_matcher.py # Face encoding and matching logic
│── requirements.txt # Required dependencies
│── README.md # Project documentation

## API Endpoints

Endpoint: POST /process_images/
Description: Accepts a set of images and a target image, encodes faces, matches them, and returns matched images.

## Dependeicies

FastAPI (for API development)
Uvicorn (for running the FastAPI server)
Face Recognition (for face detection & encoding)
OpenCV (for image processing)
NumPy (for handling encoding data)
SQLite (for database storage)
