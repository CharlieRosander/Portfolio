import os
import random
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tkinter import (
    filedialog,
    Tk,
    Button,
    Label,
    Listbox,
    MULTIPLE,
    Checkbutton,
    IntVar,
    Toplevel,
    messagebox,
)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Classifier")

        self.model = tf.keras.models.load_model("saved_models/model_categorical.keras")
        self.target_size = (96, 96)
        self.image_files = []
        self.classified_images = {}

        self.folder_label = Label(root, text="Ingen mapp vald")
        self.folder_label.pack()

        self.select_folder_btn = Button(
            root, text="Välj mapp", command=self.select_folder
        )
        self.select_folder_btn.pack()

        self.classify_btn = Button(
            root, text="Klassificera bilder", command=self.classify_images
        )
        self.classify_btn.pack()

        self.listbox = Listbox(root, selectmode=MULTIPLE)
        self.listbox.pack()

        self.copy_btn = Button(
            root, text="Kopiera valda bilder", command=self.copy_selected_images
        )
        self.copy_btn.pack()

    def select_folder(self):
        folder_path = filedialog.askdirectory(title="Välj en mapp med bilder")
        if folder_path:
            self.folder_label.config(text=folder_path)
            self.image_files = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith(("png", "jpg", "jpeg"))
            ]
            if len(self.image_files) > 10:
                self.image_files = random.sample(self.image_files, 10)

    def prepare_image(self, file_path):
        img = image.load_img(file_path, target_size=self.target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalisera bilden
        return img_array

    def classify_images(self):
        if not self.image_files:
            messagebox.showerror("Fel", "Ingen mapp vald eller inga bilder hittade.")
            return

        self.classified_images.clear()
        self.listbox.delete(0, "end")

        for img_file in self.image_files:
            img_array = self.prepare_image(img_file)
            predictions = self.model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            if predicted_class not in self.classified_images:
                self.classified_images[predicted_class] = []
            self.classified_images[predicted_class].append(img_file)
            self.listbox.insert(
                "end", f"Klass: {predicted_class} - {os.path.basename(img_file)}"
            )

        self.show_images()

    def show_images(self):
        top = Toplevel(self.root)
        top.title("Klassificerade bilder")
        fig = plt.figure(figsize=(15, 10))

        for i, img_file in enumerate(self.image_files):
            img_array = self.prepare_image(img_file)
            predictions = self.model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]

            ax = fig.add_subplot(2, 5, i + 1)
            img = image.load_img(img_file, target_size=self.target_size)
            ax.imshow(img)
            ax.set_title(f"Klass: {predicted_class}")
            ax.axis("off")

        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def copy_selected_images(self):
        selected_indices = self.listbox.curselection()
        selected_classes = {
            self.listbox.get(i).split()[1]: [] for i in selected_indices
        }

        for i in selected_indices:
            class_label = self.listbox.get(i).split()[1]
            img_file = " ".join(self.listbox.get(i).split()[3:])
            selected_classes[class_label].append(img_file)

        dest_folder = filedialog.askdirectory(title="Välj en mål mapp")
        if not dest_folder:
            messagebox.showerror("Fel", "Ingen mål mapp vald.")
            return

        for class_label, files in selected_classes.items():
            class_folder = os.path.join(dest_folder, f"Class_{class_label}")
            os.makedirs(class_folder, exist_ok=True)
            for file in files:
                shutil.copy(file, class_folder)

        messagebox.showinfo("Klar", "Bilderna har kopierats.")


if __name__ == "__main__":
    root = Tk()
    app = ImageClassifierApp(root)
    root.mainloop()
