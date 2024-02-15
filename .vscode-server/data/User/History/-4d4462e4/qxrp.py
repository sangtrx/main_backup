#!/usr/bin/env python

import csv
import json
import requests
import io
import base64
import os
import sys
import random
from PIL import Image

##################################################
# Config settings
##################################################
csv_path = "/home/tqsang/x2_lab/image.csv"
output_dir = "/mnt/tqsang/x2_lab/insta"
url = "http://127.0.0.1:7860"
img_width = 540
img_height = 540
restore_faces = "true"

negative_prompt = "EasyNegative, bad-hands-5, (((nude, naked,child, child face))), un-detailed skin, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, ugly eyes, (out of frame:1.3), worst quality, low quality, jpeg artifacts, cgi, sketch, cartoon, drawing, (out of frame:1.1), worst quality, low quality, jpeg artifacts, poorly drawn, (((word, words, letter, text, signature, watermark)))"
steps = 80
hires = "true"

##################################################
# Function to generate images via API calls
##################################################
def generate_image(prompt):
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "batch_size": 1,
        "steps": steps,
        "seed": random.randrange(sys.maxsize),
        "enable_hr": hires,
        "width": img_width,
        "height": img_height,
        "restore_faces": restore_faces,
    }

    response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)

    try:
        r = response.json()
    except ValueError as e:
        print("NOT JSON RESPONSE")
        exit()

    # Print the response to debug the issue
    print("API Response:", r)

    img_data = r['images'][0]
    image = Image.open(io.BytesIO(base64.b64decode(img_data.split(",", 1)[0])))
    return image


##################################################
# Read CSV file, generate images, and save them to the output directory
##################################################
with open(csv_path, mode="r") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        prompt = row[0]
        print(f"Generating image for prompt: {prompt}")
        image = generate_image(prompt)
        output_filename = f"{output_dir}/{prompt}.jpg"
        image.save(output_filename, "JPEG")
        print(f"Image saved to {output_filename}")
