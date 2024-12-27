import tkinter as tk
import customtkinter
from PIL import ImageTk, Image
import compare_faces
import re_train
import collect_passport

#import compare_faces
customtkinter.set_appearance_mode("dark")
root = customtkinter.CTk()
root.attributes("-fullscreen", True)
#root.attributes("-fullscreen", True)
root.geometry("500x500")

#configures the rows, 0,1 and 2 so that they will all stick to the corners. Then configures the column to do the same
root.rowconfigure(0, weight=1)
root.rowconfigure(1, weight=1)
root.rowconfigure(2, weight=1)
root.columnconfigure(0, weight=1)

def compare_faces_button():
  root.destroy()
  compare_faces.compare_faces()
  
def re_train_button():
  re_train.re_train()

def collect_passport_button():
  collect_passport.collect_passport()

#class for creating the buttons
class buttons:
  def __init__(self, name, row, commands):
    self.name = name
    self.row = row
    self.commands = commands
  
  def button_create(self):
    self.name = customtkinter.CTkButton(root, text=self.name, corner_radius=0, border_width=10, fg_color="#808080", text_font=("Helvetica",60), command=self.commands)
    self.name.grid(row=self.row, column=0, sticky=tk.NSEW)
  
collect_button = buttons("Collect Passport", 0, collect_passport_button)
collect_button.button_create()

compare_button = buttons("Compare faces", 1, compare_faces_button)
compare_button.button_create()

retrain_button = buttons("Re-train", 2, re_train_button)
retrain_button.button_create()

#ends the loop
root.mainloop()