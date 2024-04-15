import os
import cv2 
from PIL import ImageGrab
import pymsgbox
import threading
import numpy as np
import math
import sys
import csv
import random
import tkinter as tk
from tkinter import filedialog, simpledialog
from tkinter import messagebox
from PIL import Image, ImageDraw
import pyautogui
import time
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from keras.models import load_model
#from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
#from tensorflow.keras.models import Model




k1=0
k2=0
k3=0
k4=0

directory_path = '/home/sourav/Desktop/CC_diag/training_data'
csv_file_path = '/home/sourav/Desktop/CC_diag/dataset/dataset.csv'

class DrawingCanvas:
    def __init__(self, canvas_width=800, canvas_height=800):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.canvas_closed = False
        self.button_clicked = False
        self.drawing_straight_line = False
        self.drawing_arrow = False
        self.select_item = False
        self.selection_rectangle = False
        self.line_counter=0
        self.selected_items = set()

    def create_canvas(self):
        self.root = tk.Tk()
        self.top_window = tk.Toplevel(self.root)
        self.canvas = tk.Canvas(self.top_window, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack()

        self.button1 = tk.Button(self.root, text="straight line", command=self.straight_line, font=("Helvetica", 16), padx=30, pady=30)
        self.canvas.pack(side=tk.RIGHT)
        self.button1.pack(anchor=tk.NW)

        self.button2 = tk.Button(self.root, text="arrow head", command=self.arrow_head, font=("Helvetica", 16), padx=30, pady=30)
        self.canvas.pack(side=tk.RIGHT)
        self.button2.pack(anchor=tk.NW)

        self.button5 = tk.Button(self.root, text="New canvas", command=self.new_canvas, font=("Helvetica", 16), padx=30, pady=30)
        self.canvas.pack(side=tk.RIGHT)
        self.button5.pack(anchor=tk.NW)

        self.button3 = tk.Button(self.root, text="quit", command=self.on_window_close, font=("Helvetica", 14), padx=25, pady=25)
        self.canvas.pack(side=tk.RIGHT)
        self.button3.pack(anchor=tk.NW)

        self.button4 = tk.Button(self.root, text="save", command=self.save_image, font=("Helvetica", 14), padx=25, pady=25)
        self.canvas.pack(side=tk.LEFT)
        self.button4.pack(anchor=tk.NE)

#        self.button4 = tk.Button(self.root, text="select", command=lambda event=None: self.select_image(event), font=("Helvetica", 14), padx=25, pady=25)
        self.button4 = tk.Button(self.root, text="select", command=self.select_image, font=("Helvetica", 14), padx=25, pady=25)
        self.canvas.pack(side=tk.LEFT)
        self.button4.pack(anchor=tk.NE)

        self.button4 = tk.Button(self.root, text="delete", command=self.delete_image, font=("Helvetica", 14), padx=25, pady=25)
        self.canvas.pack(side=tk.LEFT)
        self.button4.pack(anchor=tk.NE)

        self.button4 = tk.Button(self.root, text="select all", command=self.select_all, font=("Helvetica", 14), padx=25, pady=25)
        self.canvas.pack(side=tk.LEFT)
        self.button4.pack(anchor=tk.NE)

        self.button_verify = tk.Button(self.root, text="Verify", command=self.verify_image, padx=25, pady=25)
        self.button_verify.pack(side=tk.BOTTOM)

        self.root.protocol("WM_DELETE_WINDOW", self.on_window_close)
        self.root.bind("<Escape>", self.on_window_close)
        self.root.bind("<c>", self.on_window_close)
        self.root.bind("d", self.delete_image)



        self.root.mainloop()

    def select_all(self):
        for item in self.canvas.find_all():
            if item not in self.selected_items:
                self.canvas.itemconfig(item, fill="red")  # Change outline color to indicate selection
                self.selected_items.add(item)


    def select_image(self):
        self.select_item = True
        self.canvas.bind("<Button-1>", self.selection)
    
    def selection(self, event):
        if self.select_item:
            item = self.canvas.find_closest(event.x, event.y)
            self.canvas.itemconfig(item, fill="red")  # Change outline color to indicate selection
            self.selected_items.add(item[0])

    def delete_image(self):
        print("Delete key pressed")
        for item in self.selected_items:
            self.canvas.delete(item)  # Delete the selected item
        self.selected_items.clear()  # Clear the set of selected items



    def arrow_head(self):
        self.drawing_arrow = True
        self.drawing_straight_line = False
        self.select_item = False
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)



    def straight_line(self):
        print('drawing a straight line')
        self.drawing_straight_line = True
        self.drawing_arrow = False
        self.select_item = False
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)

    def on_canvas_click(self, event):
        if self.drawing_straight_line or self.drawing_arrow:
            print('clicked')
            self.start_point=(event.x,event.y)
            self.line_tag = "line{}".format(self.line_counter)
            self.line_counter += 1
            print(event.x,event.y)

    def on_canvas_drag(self,event):
        if self.drawing_straight_line and self.start_point:
            print('dragging',self.start_point[0], self.start_point[1])
            # Check if any items with the specified tag exist on the canvas
            self.canvas.delete(self.line_tag)  # Delete previous line
            self.canvas.create_line(self.start_point[0], self.start_point[1], event.x, event.y, tags=(self.line_tag,))

        if self.drawing_arrow and self.start_point:
            print('dragging',self.start_point[0], self.start_point[1])
            # Check if any items with the specified tag exist on the canvas
            self.canvas.delete(self.line_tag)  # Delete previous line
            x1=-self.start_point[1] + event.y + event.x
            y1=self.start_point[0] - event.x + event.y
            x2=self.start_point[1] - event.y + event.x
            y2=-self.start_point[0] + event.x + event.y
            self.canvas.create_line(self.start_point[0], self.start_point[1], x1, y1, tags=(self.line_tag,))
            self.canvas.create_line(self.start_point[0], self.start_point[1], x2, y2, tags=(self.line_tag,))


    def on_canvas_release(self,event):
            self.start_point = None
            self.drawing_arrow = False


    def new_canvas(self):
        print("Button1 clicked!")



    def on_canvas_release(self,event):
        if self.drawing_straight_line and self.start_point:
#            self.drawing_straight_line = False
            self.start_point = None


    def save_image(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".jpeg", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpeg")])
        if file_path:
            # Determine the file extension chosen by the user
            extension = os.path.splitext(file_path)[1].lower()

            # Create an empty image with the same size as the canvas
            image = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
            draw = ImageDraw.Draw(image)

            # Copy the contents of the canvas onto the image
            self.canvas.postscript(file="tmp.ps", colormode="color")
            image = Image.open("tmp.ps")

            # Save the image based on the chosen file format
            if extension == ".jpeg" or extension == ".jpg":
                image.save(file_path, "JPEG")
            else:  # Default to PNG if extension not recognized or chosen
                image.save(file_path, "PNG")

            # Clean up temporary postscript file
            os.remove("tmp.ps")


    def on_window_close(self, event=None):
        print("Window closed!")
        self.canvas_closed = True
        cv2.destroyAllWindows()
        self.root.destroy()
        sys.exit()

    def verify_image(self):
        test_image_path='/home/sourav/Desktop/CC_diag/test/image1.jpeg'
        # Get the drawn image from the canvas
#        img = Image.open(test_image_path)
        img = cv2.imread(test_image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        threshold_value=100
        _, binary_image = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

        dark_pixels = np.where(binary_image == 0)
        min_row = np.min(dark_pixels[0])
        max_row = np.max(dark_pixels[0])
        min_col = np.min(dark_pixels[1])
        max_col = np.max(dark_pixels[1])
        
        # Crop the image using the bounding box
        cropped_image = binary_image[min_row:max_row+4, min_col:max_col+4]
        
        # Display or save the cropped image
#        cv2.imshow('Cropped Image', cropped_image)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
#        self.canvas.postscript(file=filename, colormode='color')
#        img = Image.open(filename)
        cropped_np = np.array(cropped_image)  # Convert to NumPy array
        img = Image.fromarray(cropped_np).resize((28, 28))
#        img = cropped_image.resize((28, 28))  # Resize to match model input shape
#        img = img.convert('L')  # Convert to grayscale
#        img = np.array(img)
        img = img.astype('float32') / 255.0
        img = img.reshape(1, 28, 28, 1)  # Reshape for model input
    
        # Normalize pixel values
#        cv2.imshow('image', img_np[0])  # Display the first image in the batch
#        cv2.waitKey(100)
    
        # Make prediction
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)
    
        # Show prediction result
        messagebox.showinfo("Prediction Result", f"Predicted Class: {predicted_class}")







class cropping:
    def __init__(self,original_image,csv_path):
        self.csv_path=csv_path
        self.original_image=original_image
        self.image_dict = {}
        self.dark_pixels_list = []

    def improc(self):
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        threshold_value=100
        _, binary_image = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        
        dark_pixels = np.where(binary_image == 0)
#        self.dark_pixels_list = [dark_pixels[1], dark_pixels[0]]
        self.dark_pixels_list = [(dark_pixels[1][i], dark_pixels[0][i]) for i in range(len(dark_pixels[0]))]
#        print(self.dark_pixels_list)

    def calculate_distance(self,x1, y1, x2, y2):
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


    def crop(self):
        nn=20
        k=0
        l=len(self.dark_pixels_list)
#        print('lllllllllll',l)
        for i in range (l-1):
            near_neighbours=[]
#            print('self.dark_pixels_list[i]',self.dark_pixels_list[i])
            x,y=self.dark_pixels_list[i]
            if x != 0 and y != 0:
                near_neighbours.append((x,y))
        #    print('dark_pixels',dark_pixels)
                for (a,b) in near_neighbours:
                    for j in range (i+1,l):
                        x1,y1=self.dark_pixels_list[j]
                        d=self.calculate_distance(a, b, x1, y1)
                #        print('iiii',i,x,y,x1,y1)
                #        print('ddd',d)
                        if d < 10: # d is the distance between two co-ordinates 
                            #need to adjust according to the diffence between images on the page
        
                            near_neighbours.append((x1,y1))
                            self.dark_pixels_list[j]=(0,0)
                k=k+1
                list_name = f"image_{k}"
                self.image_dict[list_name] = near_neighbours
 #               print(self.image_dict)


#    def random_even_number(self):
#        # Generate a random integer
#        num = random.randint(1, 1000)
#    
#        # If the number is odd, add 1 to make it even
#        if num % 2 != 0:
#            num += 1
#    
#        return num
#    
#    def random_odd_number(self):
#        # Generate a random integer
#        num = random.randint(1, 1000)
#    
#        # If the number is odd, add 1 to make it even
#        if num % 2 != 1:
#            num += 1
#    
#        return num
 
    def type1(self,kk,cropped_image,csv_path):
        # particle creation +e
        image = cropped_image
        output_dir = '/home/sourav/Desktop/CC_diag/dataset/type1'
        file_name=f"image_{kk}"
        output_path = os.path.join(output_dir,f"{file_name}.jpeg")
        cv2.imwrite(output_path, image)
        label= '0'
        data=[
        [output_path,label]
        ]
        with open(csv_path, mode='a', newline='')as file:
            writer = csv.writer(file)
            writer.writerows (data)
    
    def type2(self,kk,cropped_image,csv_path):
        # hole creation +O
        image = cropped_image
        output_dir = '/home/sourav/Desktop/CC_diag/dataset/type2'
        file_name=f"image_{kk}"
        output_path = os.path.join(output_dir,f"{file_name}.jpeg")
        cv2.imwrite(output_path, image)
        label= '1'
        data=[
        [output_path,label]
        ]
        with open(csv_path, mode='a', newline='')as file:
            writer = csv.writer(file)
            writer.writerows (data)
    
    def type3(self,kk,cropped_image,csv_path):
        # particle annihilation -e
        image = cropped_image
        output_dir = '/home/sourav/Desktop/CC_diag/dataset/type3'
        file_name=f"image_{kk}"
        output_path = os.path.join(output_dir,f"{file_name}.jpeg")
        cv2.imwrite(output_path, image)
        label= '2'
        data=[
        [output_path,label]
        ]
        with open(csv_path, mode='a', newline='')as file:
            writer = csv.writer(file)
            writer.writerows (data)
    
    def type4(self,kk,cropped_image,csv_path):
        # particle annihilation -o
        image = cropped_image
        output_dir = '/home/sourav/Desktop/CC_diag/dataset/type4'
        file_name=f"image_{kk}"
        output_path = os.path.join(output_dir,f"{file_name}.jpeg")
        cv2.imwrite(output_path, image)
        label= '3'
        data=[
        [output_path,label]
        ]
        with open(csv_path, mode='a', newline='')as file:
            writer = csv.writer(file)
            writer.writerows (data)

    def show_popup(self, message):
        def display_message():
            messagebox.showinfo("Popup", message)

        # Create a thread for displaying the message
        popup_thread = threading.Thread(target=display_message)
        popup_thread.daemon = True  # Set the thread as a daemon so it terminates when the main program ends
        popup_thread.start()


    
    def recons_image(self):
        global k1, k2, k3, k4
#        for key in self.image_dict.items():
        for key, coordinates in self.image_dict.items():
            # Extract the index 'i' from the key
            i = int(key.split('_')[-1])
            # Find the extreme coordinates for each image_i
            max_x = max(coord[0] for coord in coordinates)
            min_x = min(coord[0] for coord in coordinates)
            max_y = max(coord[1] for coord in coordinates)
            min_y = min(coord[1] for coord in coordinates)
            # Calculate the reconstructed dimensions with margin
            margin = 10  # Adjust this margin as needed
            x1= max_x + margin
            x2= min_x - margin
            y1= max_y + margin
            y2= min_y - margin

            cropped_image = self.original_image[y2:y1, x2:x1]
            constant_x = x2
            constant_y = y2
            # Apply operations to each coordinate
            modified_coordinates = []
            dark_pixel_coordinates = []
            dark_threshold = 50
            for x, y in coordinates:
                modified_x = x - constant_x  # Add constant to x coordinate
                modified_y = y - constant_y  # Subtract constant from y coordinate
                modified_coordinates.append((modified_x, modified_y))
#            print('coordinates',coordinates)
#            print('mod_coordinates',modified_coordinates)
#            print('x1,x2,y1,y2',x1,x2,y1,y2)
#            if cropped_image is not None:
#                try:
#                    cv2.imshow(f'image_{i}', cropped_image)
#                except Exception as e:
#                    print(f"Error displaying cropped image: {e}")
#                    continue

            # Convert black pixels to white based on coordinates
            height, width = cropped_image.shape[:2]
 #           print('cropped_image',cropped_image)
 #           print(width,height)
            for x in range(height):
                for y in range(width):
                    if (y, x) not in modified_coordinates:
                        cropped_image[x, y] = 255  # Set pixel to white
            if cropped_image is not None:
                try:
                    cv2.imshow(f'image_{i}', cropped_image)
                except Exception as e:
                    print(f"Error displaying cropped image: {e}")
                    continue
#            cv2.imshow(f'image_{i}', cropped_image)
            while True:  #it is a infinite loop wait until the correct integer key has been pressed
                key_press = cv2.waitKey(0)  # Wait indefinitely for a key press

                # Convert the key press to an integer
                key_press_int = int(chr(key_press & 0xFF))

                # Check if the input matches the desired input
                if key_press_int in [3, 4, 5, 6, 0]:
                    print("Input received:", key_press_int)
                    if key_press_int == 3:
                        k1=k1+1
                        self.type1(k1,cropped_image,csv_file_path)
                    if key_press_int == 4:
                        k2=k2+1
                        self.type2(k2,cropped_image,csv_file_path)
                    if key_press_int == 5:
                        k3=k3+1
                        self.type3(k3,cropped_image,csv_file_path)
                    if key_press_int == 6:
                        k4=k4+1
                        self.type4(k4,cropped_image,csv_file_path)
                    if key_press_int == 0:
                        print('this picture is rejected')
                    break  # Exit the loop if desired input is received

            cv2.destroyAllWindows()  # Close all OpenCV windows after input

class running_model:
    def __init__(self,csv_path):
        self.csv_path=csv_path
        self.input_shape = (28, 28, 1)
        self.df = pd.DataFrame()
        self.image_paths = []
        self.labels = []
        self.X_train, self.X_val, self.y_train, self.y_val = None, None, None, None

    def read_csv(self):
        self.df = pd.read_csv(self.csv_path)
        self.image_paths = self.df['path'].tolist()
        self.labels = self.df['label'].tolist()
        print(self.df.head())



    def load_images(self):
        images = []
        for path in self.image_paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale image
            img = cv2.resize(img, (28, 28))  # Resize to input shape
            img = img.astype('float32') / 255.0  # Normalize pixel values
            img = np.expand_dims(img, axis=-1)  # Add channel dimension
            images.append(img)
        return np.array(images)

    def split_data(self):
#        images = []
        self.read_csv()
        images = self.load_images()
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(images, self.labels, test_size=0.2, random_state=42)
        self.y_train = np.array(self.y_train)
        self.y_val = np.array(self.y_val)

        # Convert image data to NumPy arrays
        self.X_train = np.array(self.X_train)
        self.X_val = np.array(self.X_val)

#        print("X_train shape:", self.X_train.shape)
#        print("X_val shape:", self.X_val.shape)
#        print("y_train shape:", self.y_train.shape)
#        print("y_val shape:", self.y_val.shape)
#
#        print("X_train data type:", self.X_train.dtype)
#        print("X_val data type:", self.X_val.dtype)
#        print("y_train data type:", self.y_train.dtype)
#        print("y_val data type:", self.y_val.dtype)

    def create_cnn_model(self):
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(4, activation='softmax')  # Assuming 4 classes
        ])
        return model

    def run_model(self):
        self.split_data()
        model = self.create_cnn_model()
        model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        print(self.X_train)
        model.fit(self.X_train, self.y_train, epochs=10, validation_data=(self.X_val, self.y_val))
        est_loss, test_acc = model.evaluate(self.X_val, self.y_val)
        print(f'Test accuracy: {test_acc}')
        model.save('diag_model_1.keras')
        print("Model saved as 'diag_model_1.keras'")




def read_image(directory_path):
    file_names = [f for f in os.listdir(directory_path) if\
            os.path.isfile(os.path.join(directory_path, f))]# Get a list of file names
    file_names.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    print(file_names)
    return(file_names)

def create_csv(csv_path):
    data = [
    ['image_name', 'label']
    ]   
    
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)


opt=input("DRAWING IMAGE : 1 \n READING IMAGE AND PRODUCE DATASET : 2 \n RUNNING MODEL : 3 \n TESTING MODEL : 4 ")

if opt == '1':
    canvas = DrawingCanvas()
    canvas.create_canvas()
if opt == '2':
    file_names = read_image(directory_path)
    create_csv(csv_file_path)
        
    for file_name in file_names:
        image_path = os.path.join(directory_path,file_name)
        image=cv2.imread(image_path)
        cv2.imshow('Image',image)
        cv2.waitKey(1000)
        aa = 'press 3 for particle creation (+e), press 4 for hole creation (+O), press 5 for particle annihilation (-e), press 6 for hole annihilation -O, press 0 to reject the picture'
        
        b=cropping(image,csv_file_path)
        b.improc()
        b.crop()
        b.show_popup(aa)
        b.recons_image()
if opt == '3':
    c=running_model(csv_file_path)
    c.read_csv()
    c.split_data()
    c.run_model()

if opt == '4':
    model = load_model('/home/sourav/Desktop/CC_diag/Saved_Model/diag_model_1.keras')
    d = DrawingCanvas()
    d.create_canvas()
    d.verify_image()

