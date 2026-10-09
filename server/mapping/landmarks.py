"""Pose landmark extraction via MediaPipe PoseLandmarker.

Run: python -m server.mapping.landmarks <image_path> [--model models/pose_landmarker_full.task]
Prints JSON: {"landmarks": [{"x","y","z","visibility","presence"}, ...]}.
Pass --draw out.png to also render landmarks + skeleton onto a copy of the image.
"""
# treats type hints as plain text
from __future__ import annotations
# lets me read terminal
import argparse
#converts python dictionary from detect into json file
import json
#tells my output where to go
import sys
#helps get the path to the model and the image
from pathlib import Path

#imports mediapipe and the pose landmarker module and sets it as mp so I can use it later in the code
import mediapipe as mp
# imports the mediapipe tasks module and sets it as mp_python so I can use it later in the code basic setup
from mediapipe.tasks import python as mp_python
# importas the mediapipe vision module and sets it as mp_vision so I can use it later in the code image tools
from mediapipe.tasks.python import vision as mp_vision
#default model to use since I don't have a model trained for this specific task, I will use the default model provided by mediapipe
DEFAULT_MODEL = Path("models/pose_landmarker_full.task")

# Pairs of landmark indices that form the skeleton edges MediaPipe draws.
# one slight issue mp.solutions is from the old mediapipe solution API, but it still works for getting the pose connections
POSE_CONNECTIONS = frozenset(mp.solutions.pose.POSE_CONNECTIONS)

#detect function takes in an image path and a model path, checks if they exist, and then uses the MediaPipe PoseLandmarker to detect pose landmarks in the image. It returns a dictionary containing the detected landmarks and a boolean indicating whether any landmarks were detected.
def detect(image_path: Path, model_path: Path) -> dict: # type hinting for the function parameters and return type
    if not image_path.exists(): # checks if the image path exists, if not it raises a FileNotFoundError with a message indicating that the image was not found
        raise FileNotFoundError(f"image not found: {image_path}")
    if not model_path.exists(): # checks if the model path exists, if not it raises a FileNotFoundError with a message indicating that the model was not found
        raise FileNotFoundError(f"model not found: {model_path}")
    # starts building the detector settings by making an object of the PoselandmarkerOptions class
    options = mp_vision.PoseLandmarkerOptions(
        # sets the model path to the provided model path, converting it to a string cause mediapipe expects a string path
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        # processes a single image, so we set the running mode to IMAGE going to change to VIDEO for later experimentation
        running_mode=mp_vision.RunningMode.IMAGE,
        # number of poses to 1 because there is only one person in the image, if there were multiple people we would set this to a higher number
        num_poses=1,
        #don't need it for this one, clean white background right now but i need to impliment it for later
        output_segmentation_masks=False,
    )
    #from the settings of the object options we create another object of class PoseLandmarker that becomes the detector
    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        #looks at the image path and creates an image object from it
        image = mp.Image.create_from_file(str(image_path))
        #uses the landmarker object to detect pose landmarks in the image and then stores the land marks in a variable called result, a list of landmarks is stored in result.pose_landmarks, if no landmarks are detected then result.pose_landmarks will be empty
        result = landmarker.detect(image)
    # returns false if no landmarks were detected, otherwise it returns a dictionary containing the detected landmarks one entry per person
    if not result.pose_landmarks:
        return {"landmarks": [], "detected": False}
    #take the first persons 33 landmarks and store them in a variable called pose, then return a dictionary containing the detected landmarks and a boolean indicating whether any landmarks were detected
    pose = result.pose_landmarks[0]
    return {
        "detected": True,
        "landmarks": [
            {   # seperate dictionary for each landmark with the x, y, z coordinates and visibility and presence values
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility,
                "presence": lm.presence,
            }
            for lm in pose
        ],
    }


def draw(image_path: Path, landmarks: list[dict], out_path: Path) -> None:
    from PIL import Image, ImageDraw
    #opens the image at the given path and converts it to RGB format, then gets the width and height of the image and creates a drawing canvas on top of the image
    img = Image.open(image_path).convert("RGB") #rgb conversion
    w, h = img.size #width and heigh of the image
    canvas = ImageDraw.Draw(img) #drawing tool that draws on img

    pts = [(lm["x"] * w, lm["y"] * h) for lm in landmarks] #convert landmark coordinates to image coordinates also known as normalizing
    # grabs the pairs of landmark indicies 
    for a, b in POSE_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            #draws a line between the two landmarks on the canvas, with a green color and a width that is proportional to the size of the image
            canvas.line([pts[a], pts[b]], fill=(0, 255, 0), width=max(2, min(w, h) // 300))
    #draws a red circle at each landmark point on the canvas, with a radius that is proportional to the size of the image
    r = max(3, min(w, h) // 200)
    for x, y in pts:
        canvas.ellipse((x - r, y - r, x + r, y + r), fill=(255, 0, 0))
    #saves the annotated image to the specified output path
    img.save(out_path)

# accepts command line arguments for the image path, model path, and output path for the annotated image. 
def main(argv: list[str] | None = None) -> int:
    # sets up an argument parser to handle command line arguments
    parser = argparse.ArgumentParser(description="Extract pose landmarks from an image.")
    parser.add_argument("image", type=Path, help="path to input image") # converts what i type to input image path to a Path object
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="path to .task file") # converts what i type to model path to a Path object, if no model is provided it uses the default model
    parser.add_argument("--draw", type=Path, help="write an annotated image to this path") # converts what i type to output image path to a Path object, if no output path is provided it does not draw the annotated image
    args = parser.parse_args(argv) # parses the command line arguments and stores them in the args variable

    out = detect(args.image, args.model) # calls the detect function with the provided image and model paths, and stores the result in the out variable
    json.dump(out, sys.stdout, indent=2) # converts the out dictionary to a JSON string and writes it to standard output with an indentation of 2 spaces for readability
    sys.stdout.write("\n") # writes a newline character to standard output to ensure that the output is properly formatted

    if args.draw and out["detected"]: # checks if the --draw argument was provided and if any landmarks were detected
        draw(args.image, out["landmarks"], args.draw) # calls the draw function with the provided image path, detected landmarks, and output path to create an annotated image
        print(f"wrote annotated image: {args.draw}", file=sys.stderr)# writes a message to standard error indicating that the annotated image was successfully created and saved to the specified output path

    return 0 if out["detected"] else 1 # returns 0 if landmarks were detected, otherwise returns 1 to indicate that no landmarks were found in the image


if __name__ == "__main__": # checks if the script is being run as the main program and not being imported as a module, and if so, it calls the main function and exits the program with the return code from main
    raise SystemExit(main()) 
