import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV2
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk

# Setting random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


class ModelTrainerApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Transfer Learning with MobileNetV2")
        self.geometry("800x600")

        # Variables
        self.train_dir_var = tk.StringVar()
        self.test_dir_var = tk.StringVar()
        self.model_save_var = tk.StringVar(
            value=os.path.join(os.getcwd(), "saved_models", "model.keras")
        )
        self.epochs_var = tk.StringVar(value="10")
        self.lr_var = tk.StringVar(value="0.0001")

        # Storage for history and model
        self.history = None
        self.model = None

        # Create tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill="both")

        self.create_training_tab()
        self.create_testing_tab()
        self.create_plots_tab()

    def create_training_tab(self):
        training_frame = ttk.Frame(self.notebook)
        self.notebook.add(training_frame, text="Training")

        ttk.Label(
            training_frame, text="Select the directory containing the training images"
        ).pack(pady=5)
        train_dir_frame = ttk.Frame(training_frame)
        train_dir_frame.pack(pady=5)
        ttk.Label(train_dir_frame, text="Train Directory:").pack(side=tk.LEFT)
        ttk.Entry(train_dir_frame, textvariable=self.train_dir_var, width=50).pack(
            side=tk.LEFT, padx=10
        )
        ttk.Button(train_dir_frame, text="Browse", command=self.browse_train_dir).pack(
            side=tk.LEFT, padx=10
        )

        ttk.Label(
            training_frame,
            text="Specify the path to save the trained model\n(Default is root/saved_models/model.keras)",
        ).pack(pady=5)
        model_save_frame = ttk.Frame(training_frame)
        model_save_frame.pack(pady=5)
        ttk.Label(model_save_frame, text="Model Save Path:").pack(side=tk.LEFT)
        ttk.Entry(model_save_frame, textvariable=self.model_save_var, width=50).pack(
            side=tk.LEFT, padx=10
        )
        ttk.Button(
            model_save_frame, text="Browse", command=self.browse_model_save
        ).pack(side=tk.LEFT, padx=10)

        ttk.Label(
            training_frame, text="Specify the number of epochs for training"
        ).pack(pady=5)
        epochs_frame = ttk.Frame(training_frame)
        epochs_frame.pack(pady=5)
        ttk.Label(epochs_frame, text="Epochs:").pack(side=tk.LEFT)
        ttk.Entry(epochs_frame, textvariable=self.epochs_var).pack(
            side=tk.LEFT, padx=10
        )

        ttk.Label(training_frame, text="Specify the learning rate for training").pack(
            pady=5
        )
        lr_frame = ttk.Frame(training_frame)
        lr_frame.pack(pady=5)
        ttk.Label(lr_frame, text="Learning Rate:").pack(side=tk.LEFT)
        ttk.Entry(lr_frame, textvariable=self.lr_var).pack(side=tk.LEFT, padx=10)

        ttk.Button(training_frame, text="Train Model", command=self.train_model).pack(
            pady=20
        )
        ttk.Button(training_frame, text="Quit", command=self.quit).pack(pady=5)

    def create_testing_tab(self):
        testing_frame = ttk.Frame(self.notebook)
        self.notebook.add(testing_frame, text="Testing & Validation")

        ttk.Label(
            testing_frame, text="Select the directory containing the testing images"
        ).pack(pady=5)
        test_dir_frame = ttk.Frame(testing_frame)
        test_dir_frame.pack(pady=5)
        ttk.Label(test_dir_frame, text="Test Directory:").pack(side=tk.LEFT)
        ttk.Entry(test_dir_frame, textvariable=self.test_dir_var, width=50).pack(
            side=tk.LEFT, padx=10
        )
        ttk.Button(test_dir_frame, text="Browse", command=self.browse_test_dir).pack(
            side=tk.LEFT, padx=10
        )

        ttk.Button(
            testing_frame, text="Evaluate Model", command=self.evaluate_model
        ).pack(pady=20)
        self.test_results = ttk.Label(testing_frame, text="")
        self.test_results.pack(pady=20)
        ttk.Button(testing_frame, text="Quit", command=self.quit).pack(pady=5)

    def create_plots_tab(self):
        plots_frame = ttk.Frame(self.notebook)
        self.notebook.add(plots_frame, text="Plots/Graphs")

        ttk.Label(
            plots_frame,
            text="Select an option below to view the respective plots or predictions",
        ).pack(pady=5)
        ttk.Button(
            plots_frame, text="Show Training Plots", command=self.show_training_plots
        ).pack(pady=20)
        ttk.Button(
            plots_frame,
            text="Show Random Predictions",
            command=self.show_random_predictions,
        ).pack(pady=20)
        ttk.Button(plots_frame, text="Quit", command=self.quit).pack(pady=5)

    def browse_train_dir(self):
        train_dir = filedialog.askdirectory()
        self.train_dir_var.set(train_dir)

    def browse_test_dir(self):
        test_dir = filedialog.askdirectory()
        self.test_dir_var.set(test_dir)

    def browse_model_save(self):
        model_save_path = filedialog.asksaveasfilename(
            defaultextension=".keras",
            initialdir=os.path.join(os.getcwd(), "saved_models"),
            filetypes=[("Keras model", "*.keras")],
        )
        self.model_save_var.set(model_save_path)

    def train_model(self):
        train_dir = self.train_dir_var.get()
        model_save_path = self.model_save_var.get()
        epochs = int(self.epochs_var.get())
        learning_rate = float(self.lr_var.get())

        if not train_dir or not model_save_path:
            messagebox.showerror(
                "Error", "Please specify all directories and model save path."
            )
            return

        BATCH_SIZE = 32
        IMG_SIZE = (96, 96)

        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input, validation_split=0.2
        )

        train_gen = train_datagen.flow_from_directory(
            train_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="categorical",  # Change to 'categorical' for 3 classes
            subset="training",
        )

        val_gen = train_datagen.flow_from_directory(
            train_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="categorical",  # Change to 'categorical' for 3 classes
            subset="validation",
        )

        IMG_SHAPE = IMG_SIZE + (3,)
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=IMG_SHAPE, include_top=False, weights="imagenet"
        )

        base_model.trainable = False

        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

        prediction_layer = tf.keras.layers.Dense(
            3, activation="softmax"
        )  # 3 classes: cat, dog, other

        data_augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.2),
            ]
        )

        inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
        x = data_augmentation(inputs)
        x = base_model(x, training=False)
        x = global_average_layer(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = prediction_layer(x)
        self.model = tf.keras.Model(inputs, outputs)

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy,
            metrics=["accuracy"],
        )

        self.history = self.model.fit(train_gen, epochs=epochs, validation_data=val_gen)

        if not os.path.exists(os.path.dirname(model_save_path)):
            os.makedirs(os.path.dirname(model_save_path))
        self.model.save(model_save_path)

        messagebox.showinfo("Success", "Model trained and saved successfully.")

    def evaluate_model(self):
        test_dir = self.test_dir_var.get()

        if not test_dir or not self.model:
            messagebox.showerror(
                "Error",
                "Please specify the test directory and ensure the model is trained.",
            )
            return

        BATCH_SIZE = 32
        IMG_SIZE = (96, 96)

        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

        test_gen = test_datagen.flow_from_directory(
            test_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="categorical",  # Change to 'categorical' for 3 classes
            shuffle=False,
        )

        test_loss, test_acc = self.model.evaluate(test_gen, verbose=2)
        self.test_results.config(
            text=f"Test accuracy: {test_acc}\nTest loss: {test_loss}"
        )

    def show_training_plots(self):
        if not self.history:
            messagebox.showerror(
                "Error", "No training history found. Train the model first."
            )
            return

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history["accuracy"])
        plt.plot(self.history.history["val_accuracy"])
        plt.title("Model Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(["Train", "Validation"], loc="upper left")

        plt.subplot(1, 2, 2)
        plt.plot(self.history.history["loss"])
        plt.plot(self.history.history["val_loss"])
        plt.title("Model Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(["Train", "Validation"], loc="upper left")

        plt.tight_layout()
        plt.show()

    def show_random_predictions(self):
        test_dir = self.test_dir_var.get()

        if not test_dir or not self.model:
            messagebox.showerror(
                "Error",
                "Please specify the test directory and ensure the model is trained.",
            )
            return

        BATCH_SIZE = 32
        IMG_SIZE = (96, 96)

        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

        test_gen = test_datagen.flow_from_directory(
            test_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="categorical",  # Change to 'categorical' for 3 classes
            shuffle=True,
        )

        test_images, test_labels = next(test_gen)

        predictions = self.model.predict(test_images)

        predicted_probs = predictions  # No need to apply sigmoid for softmax output
        predicted_labels = np.argmax(predictions, axis=1)

        rand_indices = random.sample(range(len(test_images)), 6)

        class_labels = {0: "Cat", 1: "Dog", 2: "Other"}

        plt.figure(figsize=(10, 10), facecolor="white")

        for i, idx in enumerate(rand_indices):
            ax = plt.subplot(3, 3, i + 1)

            plt.imshow((test_images[idx] + 1) / 2.0)

            true_label = class_labels[np.argmax(test_labels[idx])]
            pred_label = class_labels[predicted_labels[idx]]
            pred_prob = predicted_probs[idx][
                predicted_labels[idx]
            ]  # Extract scalar value

            plt.title(f"True: {true_label}, Pred: {pred_label} ({pred_prob:.2f})")
            plt.axis("off")

        plt.show()


if __name__ == "__main__":
    app = ModelTrainerApp()
    app.mainloop()
