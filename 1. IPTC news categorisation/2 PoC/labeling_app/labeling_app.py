import pandas as pd
import tkinter as tk
from tkinter import Button, Entry
import sys

sys.path.append('../')


class LabelingApp:
    '''
    GUI for labeling records in a DataFrame.
    '''

    def __init__(self, root, dataframe):
        self.root = root
        self.dataframe = dataframe
        self.index = 0
        self.colors = colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 'gray', 'white', 'cyan', 'magenta', 'lime', 'teal', 'lavender', 'turquoise', 'tan', 'salmon', 'gold', 'lightcoral']
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        left_frame = tk.Frame(root)
        left_frame.pack(side=tk.LEFT, padx=10, pady=5)

        # Display the current record number
        self.record_label = tk.Label(left_frame, text=f"Record: {self.index + 1}/{len(self.dataframe)}", pady=10)
        self.record_label.pack()


        predicted_label = dataframe['high_label'][self.index]
        proposed_label = dataframe['label'][self.index]

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

        # Create buttons for each class in one column on the right side
        self.class_buttons = []

        classes = self.dataframe['high_label'].unique()
        colors = self.colors[:len(classes)]

        for i, label_class in enumerate(classes):
            button = Button(right_frame, text=label_class, command=lambda x=label_class: self.label_record(x), height=2, width=25, bg=colors[i], font=('Arial', 10, 'bold'))
            button.pack(pady=1, fill=tk.X)
            self.class_buttons.append(button)

        # Entry widget for inputting a record number
        self.record_entry = Entry(right_frame)
        self.record_entry.pack(side=tk.TOP, pady=3)

        # Button to jump to the specified record number
        jump_button = Button(right_frame, text="Jump to Record", command=self.jump_to_record, height=2, width=15)
        jump_button.pack(pady=2)

        # Button to save the DataFrame to a CSV file (create this button under everything else in center)
        save_button = Button(right_frame, text="Save DataFrame", command=self.save_dataframe, height=2, width=15)
        save_button.pack(pady=2)


    def label_record(self, label_class):
        '''
        Label the current record with the chosen class.
        '''

        self.dataframe.at[self.index, 'label'] = label_class
        self.index += 1
        self.record_label.config(text=f"Record: {self.index + 1}/{len(self.dataframe)}")

        self.headline_label.config(text=self.dataframe['headline'][self.index])
        self.proposed_label.config(text=f'Proposed label: {self.dataframe["label"][self.index]}')
        self.predicted_label.config(text=f'Predicted label: {self.dataframe["high_label"][self.index]}')

        if self.index < len(self.dataframe):
            self.text_label.config(text=self.dataframe['text'][self.index])
        else:
            self.text_label.config(text="All records labeled!")

        self.root.update_idletasks()

    def jump_to_record(self):
        '''
        Jump to the record number specified in the entry widget.
        '''
        try:
            record_number = int(self.record_entry.get()) - 1  # Adjust to 0-based index
            if 0 <= record_number < len(self.dataframe):
                self.index = record_number
                self.record_label.config(text=f"Record: {self.index + 1}/{len(self.dataframe)}")
                self.headline_label.config(text=self.dataframe['headline'][self.index])
                self.proposed_label.config(text=f'Proposed label: {self.dataframe["label"][self.index]}')
                self.predicted_label.config(text=f'Predicted label: {self.dataframe["high_label"][self.index]}')
                self.text_label.config(text=self.dataframe['text'][self.index])
            else:
                print("Invalid record number.")
        except ValueError:
            print("Please enter a valid record number.")

    def save_dataframe(self):
        '''
        Save the DataFrame to a CSV file. Note that when closing the GUI, the DataFrame will be saved automatically.
        '''
   
        self.dataframe.to_csv('labeled_data.csv', index=False)
        self.root.update_idletasks()

    def on_close(self):
        '''
        Close the GUI and save the DataFrame to a CSV file.
        '''

        self.save_dataframe()
        self.root.destroy()



if __name__ == "__main__":
    
    df = pd.read_csv('labeled_data.csv')
    root = tk.Tk()
    app = LabelingApp(root, df)
    root.mainloop()
