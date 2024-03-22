import os
import cv2 
from PIL import ImageGrab
import numpy as np
import math
import sys
import csv
import random
import tkinter as tk
from tkinter import filedialog, simpledialog
from PIL import Image, ImageDraw
import pyautogui



k1=0
k2=0
k3=0
k4=0

dark_pixel=[]
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



opt=input("If you want to draw your own data write 'y' or if you want to read data from file write 'n'")
if opt == 'y':
    canvas = DrawingCanvas()
    canvas.create_canvas()


def create_csv(path):
    data = [
    ['image_name', 'label']
    ]   

    with open(path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

create_csv(csv_file_path)

file_names = [f for f in os.listdir(directory_path) if\
        os.path.isfile(os.path.join(directory_path, f))]# Get a list of file names
file_names.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
print(file_names)

for file_name in file_names:
    image_path = os.path.join(directory_path,file_name)
    image=cv2.imread(image_path)
    cv2.imshow('Image',image)
    cv2.waitKey(1000)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

threshold_value=100
_, binary_image = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)


dark_pixels = np.where(binary_image == 0)
dark_pixels_list = list(zip(dark_pixels[1], dark_pixels[0]))

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)



def random_even_number():
    # Generate a random integer
    num = random.randint(1, 1000)

    # If the number is odd, add 1 to make it even
    if num % 2 != 0:
        num += 1

    return num

def random_odd_number():
    # Generate a random integer
    num = random.randint(1, 1000)

    # If the number is odd, add 1 to make it even
    if num % 2 != 1:
        num += 1

    return num


def type1(kk,cropped_image,csv_path):
    # particle creation +e
    image = cropped_image
    output_dir = '/home/sourav/Desktop/CC_diag/dataset/type1'
    file_name=f"image_{kk}"
    output_path = os.path.join(output_dir,f"{file_name}.jpeg")
    cv2.imwrite(output_path, image)
    label= + random_even_number()
    data=[
    [output_path,label]
    ]
    with open(csv_path, mode='a', newline='')as file:
        writer = csv.writer(file)
        writer.writerows (data)

def type2(kk,cropped_image,csv_path):
    # hole creation +O
    image = cropped_image
    output_dir = '/home/sourav/Desktop/CC_diag/dataset/type2'
    file_name=f"image_{kk}"
    output_path = os.path.join(output_dir,f"{file_name}.jpeg")
    cv2.imwrite(output_path, image)
    label= + random_odd_number()
    data=[
    [output_path,label]
    ]
    with open(csv_path, mode='a', newline='')as file:
        writer = csv.writer(file)
        writer.writerows (data)

def type3(kk,cropped_image,csv_path):
    # particle annihilation -e
    image = cropped_image
    output_dir = '/home/sourav/Desktop/CC_diag/dataset/type3'
    file_name=f"image_{kk}"
    output_path = os.path.join(output_dir,f"{file_name}.jpeg")
    cv2.imwrite(output_path, image)
    label= - random_even_number()
    data=[
    [output_path,label]
    ]
    with open(csv_path, mode='a', newline='')as file:
        writer = csv.writer(file)
        writer.writerows (data)

def type4(kk,cropped_image,csv_path):
    # particle annihilation -o
    image = cropped_image
    output_dir = '/home/sourav/Desktop/CC_diag/dataset/type4'
    file_name=f"image_{kk}"
    output_path = os.path.join(output_dir, f"{file_name}.jpeg")
    cv2.imwrite(output_path, image)
    label= - random_odd_number()
    data=[
    [output_path,label]
    ]
    with open(csv_path, mode='a', newline='')as file:
        writer = csv.writer(file)
        writer.writerows (data)


class cropping:
    def __init__(self,original_image,image_dict):
        self.image_dict=image_dict
        self.original_image=original_image
    def recons_image(self):
        global k1, k2, k3, k4
#        for key in self.image_dict.items():
        for key, coordinates in self.image_dict.items():
            # Extract the index 'i' from the key
            i = int(key.split('_')[-1])
            
#            print(f"Coordinates for {key}: {coordinates}")
            
            # Find the extreme coordinates for each image_i
            max_x = max(coord[0] for coord in coordinates)
            min_x = min(coord[0] for coord in coordinates)
            max_y = max(coord[1] for coord in coordinates)
            min_y = min(coord[1] for coord in coordinates)
#            print('extreem_coord',max_x, min_x, max_y, min_y)
            
            # Calculate the reconstructed dimensions with margin
            margin = 10  # Adjust this margin as needed
            x1= max_x + margin
            x2= min_x - margin
            y1= max_y + margin
            y2= min_y - margin

            cropped_image = self.original_image[y2:y1, x2:x1]
            cv2.imshow('original_image',self.original_image)
            cv2.waitKey(1000)

            # Display or save the reconstructed image
            cv2.imshow(f'image_{i}', cropped_image)
            while True:  #it is a infinite loop wait until the correct integer key has been pressed
                key_press = cv2.waitKey(0)  # Wait indefinitely for a key press

                # Convert the key press to an integer
                key_press_int = int(chr(key_press & 0xFF))

                # Check if the input matches the desired input
                if key_press_int in [1, 2, 3, 4]:
                    print("Input received:", key_press_int)
                    if key_press_int == 1:
                        k1=k1+1
                        type1(k1,cropped_image,csv_file_path)
                    if key_press_int == 2:
                        k2=k2+1
                        type2(k2,cropped_image,csv_file_path)
                    if key_press_int == 3:
                        k3=k3+1
                        type3(k3,cropped_image,csv_file_path)
                    if key_press_int == 4:
                        k4=k4+1
                        type4(k4,cropped_image,csv_file_path)
                    break  # Exit the loop if desired input is received

            cv2.destroyAllWindows()  # Close all OpenCV windows after input



# nn is the number of images to be separated 
image_dict = {}
nn=20
k=0
l=len(dark_pixels_list)
for i in range (l-1):
    near_neighbours=[]
    x,y=dark_pixels_list[i]
    if x != 0 and y != 0:
        near_neighbours.append((x,y))
#    print('dark_pixels',dark_pixels)
    #l=500
        for (a,b) in near_neighbours:
            for j in range (i+1,l):
                x1,y1=dark_pixels_list[j]
                d=calculate_distance(a, b, x1, y1)
        #        print('iiii',i,x,y,x1,y1)
        #        print('ddd',d)
                if d < 10: # d is the distance between two co-ordinates 
                    #need to adjust according to the diffence between images on the page
#                    new_row=[(x,y),(x1,y1),d]

                    near_neighbours.append((x1,y1))
                    dark_pixels_list[j]=(0,0)
        k=k+1
        list_name = f"image_{k}"
        image_dict[list_name] = near_neighbours
#print('list_name',new_)
#print(near_neighbours)
#sys.exit()
#print(near_neighbours.shape)
#a=Isolate(near_neighbours)
#image_dict=a.isolate_coord()
#print(len(image_dict))
#print(image_dict)

print('press 1 for particle creation (+e), press 2 for hole creation (+O), press 3 for particle annihilation (-e),&
press 4 for hole annihilation -O')
b=cropping(binary_image,image_dict)
b.recons_image()


##for i in range (4):
##    print(dark_pixels[i])
