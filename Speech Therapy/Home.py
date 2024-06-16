from tkinter import Tk, Button, PhotoImage, Label
from PIL import Image, ImageTk
import subprocess

# Modern theme colors (adjust to your preference)
background_color = "#f2f2f2"
button_color = "#d1c4e9"
button_hover_color = "#b39ddb"
text_color = "#333"
custom_font_path = "LuckiestGuy-Regular.ttf"


root = Tk()
root.title("REVIVE")  # Using your chosen title
root.geometry("1200x700")
root.configure(bg=background_color)  # Set background color

image1_path = "speak.png"
image2_path = "mouth.png"
image3_path = "sign.png"
logo_path = "thinking.png"  # Add the path to your logo image

image1 = Image.open(image1_path)
image2 = Image.open(image2_path)
image3 = Image.open(image3_path)
logo_image = Image.open(logo_path)  # Load the logo image

# Resize and convert images to PhotoImage (adjust size as needed)
original_width, original_height = image1.size
new_width = int(original_width * 0.3)
new_height = int(original_height * 0.3)
image1 = ImageTk.PhotoImage(image1.resize((new_width, new_height), Image.ANTIALIAS))
image2 = ImageTk.PhotoImage(image2.resize((new_width, new_height), Image.ANTIALIAS))
image3 = ImageTk.PhotoImage(image3.resize((new_width, new_height), Image.ANTIALIAS))
logo_image = ImageTk.PhotoImage(
    logo_image.resize((70, 70), Image.ANTIALIAS)
)  # Resize the logo image


# Define button click functions
def button_click_1():
    subprocess.run(["python", "speech.py"])


def button_click_2():
    subprocess.run(["python", "test.py"])


def button_click_3():
    subprocess.run(["python", "write/main.py"])


button_width = 15
button_height = 2

button_style = {
    "bg": button_color,
    "fg": "black",  # Black text color
    "font": ("LuckiestGuy", 12),  # Luckiest Guy font, smaller size
    "activebackground": button_hover_color,
    "activeforeground": "black",  # Active text color
    "border": 2,  # Border width
    "relief": "solid",  # Solid border style
    "padx": 10,  # Horizontal padding
    "pady": 5,  # Vertical padding
}

# Create widgets
button1 = Button(
    root,
    text="PLAY",
    command=button_click_1,
    width=button_width,
    height=button_height,
    **button_style,
)
button2 = Button(
    root,
    text="PLAY",
    command=button_click_2,
    width=button_width,
    height=button_height,
    **button_style,
)
button3 = Button(
    root,
    text="PLAY",
    command=button_click_3,
    width=button_width,
    height=button_height,
    **button_style,
)

title_label = Label(
    root,
    text="REVIVE",
    font=("LuckiestGuy", 60, "bold"),  # Luckiest Guy font
    anchor="center",
    fg=text_color,
)
logo_label = Label(
    root, image=logo_image, bg=background_color
)  # Create label for the logo image

image_label1 = Label(root, image=image1, bg=background_color)
image_label2 = Label(root, image=image2, bg=background_color)
image_label3 = Label(root, image=image3, bg=background_color)

# Grid manager for layout (adjust as needed)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_rowconfigure(2, weight=1)

# Place widgets on the grid
title_label.grid(
    row=0, column=0, columnspan=2, pady=(5, 0), padx=(0, 70)
)  # Adjusting top padding only
logo_label.grid(
    row=0, column=1, sticky="w", padx=(120, 5)
)  # Align logo to the right and adjust right padding only

image_label1.grid(row=1, column=0, sticky="nsew")
image_label2.grid(row=1, column=1, sticky="nsew")

button1.grid(row=2, column=0, pady=10, padx=100)
button2.grid(row=2, column=1, pady=10, padx=100)

image_label3.grid(row=3, columnspan=2, sticky="nsew")
button3.grid(row=4, columnspan=2, pady=20, padx=100)

root.mainloop()
