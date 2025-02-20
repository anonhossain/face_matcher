# main.py
from fastapi import FastAPI, UploadFile, File
import shutil
import os
import uvicorn
from image_encoder_and_matcher import ImageEncoderAndMatcher

app = FastAPI()

# Initialize the ImageEncoderAndMatcher class
image_encoder_and_matcher = ImageEncoderAndMatcher()

@app.post("/process_images/")
async def process_images(all_pictures: UploadFile = File(...), target_picture: UploadFile = File(...)):
    # Save uploaded images to directories
    all_pictures_dir = "all_pictures"
    target_pictures_dir = "target_picture"

    os.makedirs(all_pictures_dir, exist_ok=True)
    os.makedirs(target_pictures_dir, exist_ok=True)

    # Save all_pictures to 'all_pictures' directory
    with open(os.path.join(all_pictures_dir, all_pictures.filename), "wb") as buffer:
        shutil.copyfileobj(all_pictures.file, buffer)

    # Save target_picture to 'target_picture' directory
    with open(os.path.join(target_pictures_dir, target_picture.filename), "wb") as buffer:
        shutil.copyfileobj(target_picture.file, buffer)

    # Step 1: Encode faces from all_pictures
    image_encoder_and_matcher.encode_faces(all_pictures_dir)

    # Step 2: Match faces from target_picture
    result_images = image_encoder_and_matcher.match_faces(target_pictures_dir)

    # Step 3: Clean up the 'all_pictures' folder and database
    image_encoder_and_matcher.clean_up(all_pictures_dir)

    # Return the paths of matched images
    return {"result_images": result_images}

@app.on_event("shutdown")
def shutdown_event():
    image_encoder_and_matcher.close_connection()

# This condition allows you to run the app with Uvicorn directly from this script
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
