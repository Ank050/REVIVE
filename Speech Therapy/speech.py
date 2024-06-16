import random
import speech_recognition as sr
from difflib import SequenceMatcher
import tkinter as tk
from tkinter import ttk
import pygame

# Initialize Pygame for sound effects
pygame.init()
pygame.mixer.init()

# Load sound effects
game_over_sound = pygame.mixer.Sound("winner.mp3")  # Ensure this file exists


def get_difficulty_words(difficulty):

    if difficulty == "short sounds":
        return [
            "up",
            "down",
            "in",
            "out",
            "on",
            "off",
            "at",
            "it",
            "is",
            "as",
            "go",
            "to",
            "me",
            "we",
            "he",
            "she",
            "hi",
            "bye",
            "can",
            "you",
        ]
    elif difficulty == "word":
        return [
            "bed",
            "sit",
            "stand",
            "walk",
            "run",
            "hop",
            "jump",
            "clap",
            "grab",
            "hold",
            "clap",
            "stop",
            "spin",
            "jump",
            "trip",
            "flag",
            "slip",
            "skin",
            "train",
            "spoon",
            "day",
            "see",
            "high",
            "boat",
            "moon",
            "face",
            "bike",
            "house",
            "soup",
            "juice",
            "boy",
            "out",
            "cow",
            "toy",
            "house",
            "mouse",
            "loud",
            "proud",
            "sound",
            "found",
            "star",
            "her",
            "bird",
            "fork",
            "burn",
            "shirt",
            "church",
            "work",
            "nurse",
            "first",
            "safe",
            "fish",
            "shop",
            "bath",
            "vest",
            "zip",
            "face",
            "nose",
            "shoe",
            "teeth",
            "cup",
            "bag",
            "top",
            "dog",
            "cat",
            "go",
            "pig",
            "bed",
            "key",
            "gum",
        ]
    elif difficulty == "sentence":
        return [
            "How are you feeling today",
            "Can you please open the window",
            "Would you like some water",
            "Let's try to take a few steps",
            "I can help you with that",
            "Don't hesitate to ask for help if you need it",
            "Squeeze my hand if you understand",
            "Blink once for yes twice for no",
            "It's a beautiful day outside",
            "The weather is nice today",
            "You can see the blue sky",
            "I brought you your favorite juice",
            "Would you like to watch some television",
            "Do you feel any pain right now",
            "Let's practice your speech therapy exercises",
            "You're doing a great job",
            "Rest when you feel tired",
            "Is there anything you need",
            "I'm here for you",
            "Take a deep breath and relax",
            "Your therapy session is almost over",
            "We can practice again tomorrow",
            "Let's work on your balance today",
            "Can you stand up with some help",
            "Great job You're getting stronger every day",
            "Can you try to walk a few steps",
            "Slow and steady wins the race",
            "Don't give up you can do it",
            "We believe in you",
            "Let's celebrate your progress",
        ]
    else:
        raise ValueError(
            "Invalid difficulty level. Choose 'short word', 'word', or 'sentence'."
        )


def recognize_speech():
    recognizer = sr.Recognizer()
    while True:
        try:
            with sr.Microphone() as source:
                recognized_label.config(
                    text="Adjusting for ambient noise...",
                    font=("Poppins", 15, "normal", "italic"),
                    fg="grey",
                )
                root.update()
                recognizer.adjust_for_ambient_noise(source, duration=1)
                recognized_label.config(text="Recording...")
                root.update()
                recorded_audio = recognizer.listen(source, timeout=2)
                recognized_label.config(text="Recognizing...")
                root.update()
            text = recognizer.recognize_google(recorded_audio, language="en-US")
            return text
        except sr.WaitTimeoutError:
            recognized_label.config(text="Timeout occurred. Recording again...")
            root.update()
            continue
        except Exception as ex:
            return "---"


def is_similar(text1, text2):
    return SequenceMatcher(None, text1, text2).ratio() > 0.7


score = 0
rounds = 2
current_round = 0


def play_game():
    global score, current_round, word, words
    difficulty = difficulty_var.get()
    words = get_difficulty_words(difficulty)
    word = random.choice(words)
    score_label.config(text=f"Score: {score}")
    start_button.place_forget()

    def countdown(count, font_size=30, font_color="#000"):
        if count > 0:
            word_label.config(
                text=str(count), font=("Poppins", font_size, "bold"), fg="red"
            )
            root.after(1000, countdown, count - 1, font_size, font_color)
        else:
            word_label.config(text=word)
            root.after(1000, recognize_and_check)

    def recognize_and_check():
        global score, current_round, word
        recognized_text = recognize_speech()
        recognized_label.config(
            text="You said: " + recognized_text,
            font=("Poppins", 30, "bold"),
            fg="Black",
        )

        if is_similar(recognized_text.lower(), word.lower()):
            score += 1
            current_round += 1
            score_label.config(text=f"Score: {score}")
            if current_round < rounds:
                recognized_label.config(
                    text=f"Correct! Your score: {score}\n\nYou said: {recognized_text}",
                    font=("Poppins", 30, "normal"),
                    fg="green",
                )
                root.after(2500, start_next_round)
            else:
                pygame.mixer.Sound.play(game_over_sound)
                recognized_label.config(
                    text=f"Game Over! Final score: {score}\nYou said: {recognized_text}",
                    font=("Poppins", 30, "bold"),
                    fg="green",
                )
                root.after(1500, display_win_message)
        else:
            recognized_label.config(
                text=f"Incorrect. Try again.\nYou said: {recognized_text}"
            )
            root.after(2500, recognize_and_check)

    def start_next_round():
        global word
        word = random.choice(words)
        word_label.config(text="")
        recognized_label.config(text="")
        countdown(3)

    start_next_round()


def display_win_message():
    word_label.config(text="You Won!", font=("Poppins", 40, "bold"), fg="gold")
    recognized_label.config(text="")
    home_button.place(relx=0.5, rely=0.7, anchor="center")


def go_to_home():
    global score, current_round
    score = 0
    current_round = 0
    word_label.config(text="")
    recognized_label.config(text="")
    score_label.config(text=f"Score: {score}")
    home_button.place_forget()
    start_button.place(relx=0.5, rely=0.5, anchor="center")


root = tk.Tk()
root.title("Speech Recognition Game")
root.geometry("800x600")

background_image = tk.PhotoImage(file="bg4.png")
background_label = tk.Label(root, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

difficulty_var = tk.StringVar(root)
difficulty_var.set("word")
style = ttk.Style(root)
style.configure(
    "difficulty.TCombobox",
    foreground="black",
    background="lightgray",
    fieldbackground="white",
    selectforeground="black",
    borderwidth=2,
    arrowcolor="black",
    arrowsize=10,
    padding=5,
    justify="center",
    font=("Arial", 30),
    width=40,
    height=40,
)
difficulty_combobox = ttk.Combobox(
    root, textvariable=difficulty_var, state="readonly", style="difficulty.TCombobox"
)
difficulty_combobox["values"] = ["short sounds", "word", "sentence"]
difficulty_combobox.pack(pady=30)
difficulty_combobox.current(2)

score_label = tk.Label(root, text="Score: 0", font=("Helvetica", 24))
score_label.pack(pady=10)
word_label = tk.Label(root, text="", font=("Helvetica", 48, "bold"))
word_label.pack(pady=20)
recognized_label = tk.Label(root, text="", font=("Helvetica", 24))
recognized_label.pack(pady=20)

start_button = tk.Button(
    root,
    text="Start Game",
    command=play_game,
    font=("Poppins", 24),
    bg="#4CAF50",
    fg="white",
    bd=0,
    padx=20,
    pady=10,
)
start_button.place(relx=0.5, rely=0.5, anchor="center")

home_button = tk.Button(
    root,
    text="Home",
    command=go_to_home,
    font=("Poppins", 24),
    bg="#FFA500",
    fg="white",
    bd=0,
    padx=20,
    pady=10,
)

root.mainloop()
