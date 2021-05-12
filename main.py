import io
import os
from google.cloud import vision

TYPES = ["audi", "bmw", "ferrari", "opel"]


def car_image_processor(image_uri):
    print(f'VISION API --input {image_uri}')

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "car-dataflow-96f066f92d10.json"

    client = vision.ImageAnnotatorClient()

    image = read_image_local(image_uri)
    response = client.label_detection(image=image, max_results=100)
    labels = response.label_annotations

    print(labels)

    for label in labels:
        if label.description.lower() in TYPES:
            print(f'VISION API CAR BRAND FOUND: {label.description.lower()} WITH SCORE: {label.score}')
            return [label.description.lower()]


def read_image_local(image_uri):
    # Loads the image into memory
    with io.open(os.path.abspath(image_uri), 'rb') as image_file:
        content = image_file.read()
    return vision.Image(content=content)


def read_image_gcp(image_uri):
    image = vision.Image()
    image.source.image_uri = image_uri

    return image


if __name__ == '__main__':

    print(car_image_processor('opel.jpg'))
