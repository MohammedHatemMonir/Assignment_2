import pandas as pd
import tkinter as tk
from tkinter import scrolledtext

# Load the dataset
df = pd.read_csv('dataset1_merged.csv')

# Get unique habit values and their counts
habit_counts = df['habit'].value_counts()

# Create the main window
root = tk.Tk()
root.title("Unique Habit Values from Dataset1")
root.geometry("500x400")

# Create a scrollable text widget
text_widget = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=20, font=("Arial", 11))
text_widget.pack(expand=True, fill='both', padx=10, pady=10)

# Prepare the content
content = "=" * 40 + "\n"
content += "UNIQUE HABIT VALUES FROM DATASET1\n"
content += "=" * 40 + "\n\n"

# Add each habit with its count
for habit, count in habit_counts.items():
    if pd.isna(habit):
        content += f"Empty/Missing values: {count}\n"
    else:
        content += f"{habit}: {count}\n"

content += "\n" + "-" * 40 + "\n"
content += f"Total number of unique habits: {len(habit_counts)}\n"
content += f"Total records in dataset: {len(df)}\n"
content += "-" * 40

# Insert content into the text widget
text_widget.insert(tk.END, content)
text_widget.config(state=tk.DISABLED)  # Make it read-only

# Start the GUI
root.mainloop()