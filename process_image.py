import cv2
import tensorflow as tf
import numpy as np
import json
import os
import shutil
import csv
from datetime import datetime
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input

# Ensure GPU memory growth and get available device
def get_available_device():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            return '/GPU:0'
        except RuntimeError as e:
            print(e)
            return '/CPU:0'
    return '/CPU:0'

device = get_available_device()

# Load the object detection model
detect_model = tf.saved_model.load('./models/SL_modelv3/')

# Load the genus classification model
genus_model = tf.keras.models.load_model('./models/GSC_mod')
with open('/home/spr/SDv4/GSC_label.json') as f:
    genus_labels = json.load(f)

# Dictionary to cache species models
species_model_dict = {}

# Excluded genera with single species
excluded_genera = {
    'Carcharodon': 'Carcharodon_carcharias',
    'Galeocerdo': 'Galeocerdo_cuvier',
    'Rhincodon': 'Rhincodon_typus',
    'Prionace': 'Prionace_glauca',
    'Carcharias': 'Carcharias_taurus',
    'Triaenodon': 'Triaenodon_obesus'
}

# Load species model function
def load_species_model(genus_name):
    if genus_name in species_model_dict:
        return species_model_dict[genus_name]
    model_path = f'./models/{genus_name}_model'
    label_path = f'./models/{genus_name}_label.json'
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        with open(label_path) as f:
            labels = json.load(f)
        reversed_labels = {v: k for k, v in labels.items()}
        species_model_dict[genus_name] = (model, reversed_labels)
        return model, reversed_labels
    return None, None

def overwrite_directory(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    os.makedirs(directory_path)

def process_image(img_path, save_path):   
    frame = cv2.imread(img_path)
    
    input_tensor = tf.convert_to_tensor(frame, dtype=tf.uint8)
    input_tensor = tf.expand_dims(input_tensor, axis=0)

    detections = detect_model(input_tensor)

    detection_boxes = detections['detection_boxes'].numpy()[0]
    detection_scores = detections['detection_scores'].numpy()[0]

    high_score_indices = detection_scores > 0.95
    detection_boxes = detection_boxes[high_score_indices]
    detection_scores = detection_scores[high_score_indices]

    results = []

    if len(detection_scores) > 0:
        best_box_index = detection_scores.argmax()
        best_prob = detection_scores[best_box_index]
        best_box = detection_boxes[best_box_index]
        ymin, xmin, ymax, xmax = best_box
        (im_height, im_width) = frame.shape[:2]
        (x, y, w, h) = (int(xmin * im_width), int(ymin * im_height), int((xmax - xmin) * im_width), int((ymax - ymin) * im_height))
        cropped_img = frame[y:y+h, x:x+w]

        cropped_img_resized = cv2.resize(cropped_img, (224, 224))
        cropped_img_preprocessed = preprocess_input(cropped_img_resized)

        genus_pred = genus_model.predict(tf.expand_dims(cropped_img_preprocessed, axis=0))
        genus_index = genus_pred.argmax()
        genus_prob = genus_pred[0][genus_index]
        genus = genus_labels[genus_index]

        if genus in excluded_genera:
            species = excluded_genera[genus]
        else:
            species_model, species_labels = load_species_model(genus)
            if species_model:
                if genus == "Carcharhinus":
                    cropped_img_preprocessed = vgg19_preprocess_input(cropped_img_resized)
                species_pred = species_model.predict(tf.expand_dims(cropped_img_preprocessed, axis=0))
                species_index = species_pred.argmax()
                species = species_labels.get(species_index, "Unknown")
            else:
                species = "Unknown"
        spec_name = species.replace('_', ' ')

        annotation_text = f"{spec_name}"
        cv2.putText(frame, annotation_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imwrite(save_path, frame)
        img_name = os.path.basename(save_path)
        results.append({
            "image_path": img_name, 
            "species": spec_name,
            "detection_prob": best_prob,
            "class_prob": genus_prob
        })
    return results

def save_results_to_csv(results, csv_path):
    keys = results[0].keys()
    with open(csv_path, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)

if __name__ == "__main__":
    input_dir = './images/'
    output_dir = './output/images'
    overwrite_directory(output_dir)  # Overwrite the output directory
    
    all_results = []

    for img_file in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_file)
        save_path = os.path.join(output_dir, img_file)
        results = process_image(img_path, save_path)
        all_results.extend(results)
        shutil.move(img_path, save_path)

    current_time = datetime.now().strftime("m%d%H%M")
    csv_output_path = os.path.join('./output/', f"images_{current_time}_results.csv")
    save_results_to_csv(all_results, csv_output_path)
    print(f"Results saved to {csv_output_path}")
    print(all_results)
