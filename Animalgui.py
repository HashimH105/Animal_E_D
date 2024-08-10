import tkinter as tk
from tkinter import filedialog, Label, Button
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import cv2

# Function to load a Keras model from JSON and weights file
def load_model(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Load both models (replace with your actual model files)
animal_model = load_model("animal_classification_model.json", "animal_classification_model.weights.h5")
emotion_model = load_model("animal_emotion_model.json", "animal_emotion_model.weights.h5")

# List of animal classes and emotions
ANIMAL_CLASSES = ["Bear", "Brown bear", "Bull", "Butterfly", "Camel", 
                  "Canary", "Caterpillar", "Horse", "Centipede", "Cheetah", 
                  "Chicken", "Crab", "Crocodile", "Deer", "Duck", 
                  "Eagle", "Elephant", "Cat", "Fox", "Frog", 
                  "Giraffe", "Goat", "Goldfish", "Goose", "Hamster", 
                  "Harbor seal", "Hedgehog", "Hippopotamus", "Cattle", "Jaguar", 
                  "Jellyfish", "Kangaroo", "Koala", "Ladybug", "Leopard", 
                  "Penguin", "Lizard", "Lynx", "Magpie", "Monkey", 
                  "Moths and butterflies", "Mouse", "Mule", "Ostrich", "Otter", 
                  "Owl", "Panda", "Parrot", "Starfish", "Pig", 
                  "Polar bear", "Rabbit", "Raccoon", "Dog", "Red panda", 
                  "Rhinoceros", "Scorpion", "Sea lion", "Sea turtle", "Seahorse", 
                  "Shark", "Sheep", "Shrimp", "Snail", "Snake", 
                  "Sparrow", "Raven", "Squid", "Squirrel", "Lion", 
                  "Swan", "Tick", "Tiger", "Tortoise", "Turkey", 
                  "Turtle", "Whale", "Woodpecker", "Worm", "Zebra"]  # Replace with actual animal classes

EMOTIONS_LIST = ["Angry", "Happy", "Hungry", "Sad"]

# Initialize the main window
top = tk.Tk()
top.geometry('800x600')
top.title('Animal Classifier and Emotion Detector')
top.configure(background='#CDCDCD')

# Initialize labels and buttons
result_label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

# Load the Haar cascade for face detection (if needed)
# facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def preprocess_animal_image(file_path):
    try:
        image = cv2.imread(file_path)
        image_resized = cv2.resize(image, (48,48))
        image_array = np.array(image_resized).astype('float32') / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        print(f"Error preprocessing animal image: {e}")
        return None

def preprocess_emotion_image(file_path):
    try:
        image = cv2.imread(file_path)
        # Convert to RGB if not already (assuming BGR format from cv2)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[-1] == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image_resized = cv2.resize(image_rgb, (48, 48))
        image_array = np.array(image_resized).astype('float32') / 255.0
        image_array = np.expand_dims(image_array, axis=0)   # Add batch dimension
        return image_array
    except Exception as e:
        print(f"Error preprocessing emotion image: {e}")
        return None

def detect_animal(file_path):
    try:
        image_array = preprocess_animal_image(file_path)
        if image_array is not None:
            pred = animal_model.predict(image_array)
            animal_name = ANIMAL_CLASSES[np.argmax(pred)]
            return animal_name
        else:
            return "Error"
    except Exception as e:
        print(f"Error detecting animal: {e}")
        return "Error"

def detect_emotion(file_path):
    try:
        image_array = preprocess_emotion_image(file_path)
        if image_array is not None:
            pred = emotion_model.predict(image_array)
            emotion = EMOTIONS_LIST[np.argmax(pred)]
            return emotion
        else:
            return "Error"
    except Exception as e:
        print(f"Error detecting emotion: {e}")
        return "Error"

def detect_both(file_path):
    try:
        animal_name = detect_animal(file_path)
        emotion = detect_emotion(file_path)
        if animal_name != "Error" and emotion != "Error":
            result_label.configure(foreground="#011638", text=f"Animal Name: {animal_name} | Emotion: {emotion}")
        else:
            result_label.configure(foreground="#011638", text=f"Error detecting. Please try again.")
    except Exception as e:
        print(f"Error detecting both: {e}")

def show_detect_button(file_path):
    detect_b = Button(top, text="Detect", command=lambda: detect_both(file_path), padx=10, pady=5)
    detect_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    detect_b.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        if file_path:
            uploaded = Image.open(file_path)
            uploaded.thumbnail(((top.winfo_width()/2.25), (top.winfo_height()/2.25)))
            im = ImageTk.PhotoImage(uploaded)

            sign_image.configure(image=im)
            sign_image.image = im
            result_label.configure(text='')
            show_detect_button(file_path)
    except Exception as e:
        print(f"Error uploading image: {e}")

upload = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 20, 'bold'))
upload.pack(side='bottom', pady=50)
sign_image.pack(side='bottom', expand=True)
result_label.pack(side='bottom', expand=True)
heading = Label(top, text='Animal Classifier and Emotion Detector', pady=20, font=('arial', 25, 'bold'))
heading.configure(background='#CDCDCD', foreground="#364156")
heading.pack()

top.mainloop()
