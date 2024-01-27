import pandas as pd
import tkinter as tk
from tkinter import Button, Entry

class LabelingApp:
    def __init__(self, root, dataframe):
        self.root = root
        self.dataframe = dataframe
        self.index = 0
        self.colors = colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 'gray', 'white', 'cyan', 'magenta', 'lime', 'teal', 'lavender', 'turquoise', 'tan', 'salmon', 'gold', 'lightcoral']
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        left_frame = tk.Frame(root)
        left_frame.pack(side=tk.LEFT, padx=10, pady=5)

        self.record_label = tk.Label(left_frame, text=f"Record: {self.index + 1}/{len(self.dataframe)}", pady=10)
        self.record_label.pack()

        predicted_label = dataframe['high_label'][self.index]
        proposed_label = dataframe['proposed_label'][self.index]

        self.headline_label = tk.Label(left_frame, text=dataframe['headline'][self.index], font=('Arial', 16, 'bold'))
        self.predicted_label = tk.Label(left_frame, text=f'Predicted label: {predicted_label}', font=('Arial', 14, 'italic'))
        self.proposed_label = tk.Label(left_frame, text=f'Proposed label: {proposed_label}', font=('Arial', 14, 'italic'), fg='green')
      
        self.headline_label.pack(pady=1)
        self.predicted_label.pack(pady=1)
        self.proposed_label.pack(pady=1)

        self.text_label = tk.Label(left_frame, text=dataframe['text'][self.index], wraplength=500, font=('Arial', 10, 'normal'))
        self.text_label.pack(pady=10)

        right_frame = tk.Frame(root)
        right_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        self.class_buttons = {}  # Dictionary to store button objects

        for i, label_class in enumerate(self.dataframe['high_label'].unique()):
            button = Button(right_frame, text=label_class, command=lambda x=label_class: self.label_record(x), height=2, width=25, bg=colors[i], font=('Arial', 10, 'bold'))
            button.pack(pady=1, fill=tk.X)
            self.class_buttons[label_class] = button

        self.reset_button = Button(right_frame, text="RESET", command=self.reset_labels, height=2, width=15, bg='red', font=('Arial', 10, 'bold'))
        self.reset_button.pack(pady=2)

        self.next_button = Button(right_frame, text="NEXT", command=self.next_record, height=2, width=15, bg='green', font=('Arial', 10, 'bold'))
        self.next_button.pack(pady=2)

        jump_label = tk.Label(right_frame, text="Jump to Record:")
        jump_label.pack()

        self.record_entry = Entry(right_frame)
        self.record_entry.pack(pady=3)

        jump_button = Button(right_frame, text="Jump", command=self.jump_to_record, height=2, width=15, bg='cyan', font=('Arial', 10, 'bold'))
        jump_button.pack(pady=2)

    def label_record(self, label_class):
        if not isinstance(self.dataframe.at[self.index, 'proposed_label'], list):
            self.dataframe.at[self.index, 'proposed_label'] = []

        # Append the chosen class to the list
        self.dataframe.at[self.index, 'proposed_label'].append(label_class)

        self.headline_label.config(text=self.dataframe['headline'][self.index])
        self.proposed_label.config(text=f'Proposed label: {self.dataframe["proposed_label"][self.index]}')
        self.predicted_label.config(text=f'Predicted label: {self.dataframe["high_label"][self.index]}')

        if self.index < len(self.dataframe):
            self.text_label.config(text=self.dataframe['text'][self.index])

        self.root.update_idletasks()

    def next_record(self):
        self.index += 1
        if self.index < len(self.dataframe):
            self.record_label.config(text=f"Record: {self.index + 1}/{len(self.dataframe)}")
            self.headline_label.config(text=self.dataframe['headline'][self.index])
            self.proposed_label.config(text=f'Proposed label: {self.dataframe["proposed_label"][self.index]}')
            self.predicted_label.config(text=f'Predicted label: {self.dataframe["high_label"][self.index]}')
            self.text_label.config(text=self.dataframe['text'][self.index])
        else:
            self.text_label.config(text="All records labeled!")

    def reset_labels(self):
        self.dataframe.at[self.index, 'proposed_label'] = []
        self.proposed_label.config(text=f'Proposed label: {self.dataframe["proposed_label"][self.index]}')

    def jump_to_record(self):
        try:
            record_number = int(self.record_entry.get()) - 1  # Adjust to 0-based index
            if 0 <= record_number < len(self.dataframe):
                self.index = record_number
                self.record_label.config(text=f"Record: {self.index + 1}/{len(self.dataframe)}")
                self.headline_label.config(text=self.dataframe['headline'][self.index])
                self.proposed_label.config(text=f'Proposed label: {self.dataframe["proposed_label"][self.index]}')
                self.predicted_label.config(text=f'Predicted label: {self.dataframe["high_label"][self.index]}')
                self.text_label.config(text=self.dataframe['text'][self.index])
            else:
                print("Invalid record number.")
        except ValueError:
            print("Please enter a valid record number.")

    def on_close(self):
        self.root.destroy()
        self.dataframe.to_csv('labeled_data.csv', index=False)

if __name__ == "__main__":
    df = pd.read_csv('labeled_data.csv')
    if 'proposed_label' not in df.columns:
        df['proposed_label'] = ''
    root = tk.Tk()
    app = LabelingApp(root, df)
    root.mainloop()
