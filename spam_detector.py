"""
Spam Detector with Graphical Interface
Built with Tkinter
"""

import tkinter as tk
from tkinter import messagebox
import re
import pickle
import os

# Try importing libraries
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import pandas as pd
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class SpamDetectorGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("📧 Spam Message Detector")
        self.root.geometry("520x480")
        self.root.resizable(False, False)

        self.vectorizer = None
        self.model = None

        self.load_or_train_model()
        self.create_widgets()

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()

    def create_sample_data(self):

        ham = [
            "Hey, we have a meeting at 3pm today",
            "Want to grab lunch tomorrow?",
            "Your package has arrived",
            "Have you finished homework?",
            "Mom says come home for dinner",
            "Happy Birthday!",
            "Thanks for your help",
            "The weather is beautiful today",
            "I've sent the file to your email",
            "Remember to bring your textbook tomorrow",
            "Are we meeting tomorrow?",
            "See you in class"
        ]

        spam = [
            "Congratulations! You won 1 million! Click now",
            "Free iPhone! Claim today",
            "Your account is blocked verify immediately",
            "Earn money fast add WeChat",
            "You have refund pending click here",
            "Win cash prizes now",
            "Free coupons click link",
            "Guaranteed profit investment",
            "Cheap game items buy now",
            "Limited offer act fast",
            "Claim your reward now",
            "Lottery winner contact us"
        ]

        data = [(msg, "ham") for msg in ham] + \
               [(msg, "spam") for msg in spam]

        return pd.DataFrame(
            data,
            columns=["message", "label"]
        )

    def load_or_train_model(self):

        model_path = "model.pkl"

        if os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    self.vectorizer, self.model = pickle.load(f)

                print("Loaded saved model")
                return

            except:
                pass

        if not HAS_SKLEARN:
            messagebox.showerror(
                "Error",
                "Install sklearn and pandas first"
            )
            return

        df = self.create_sample_data()

        df["clean"] = df["message"].apply(
            self.clean_text
        )

        self.vectorizer = TfidfVectorizer(
            max_features=500
        )

        X = self.vectorizer.fit_transform(
            df["clean"]
        )

        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        temp_model = MultinomialNB()

        temp_model.fit(
            X_train,
            y_train
        )

        predictions = temp_model.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        print(
            f"Model Accuracy: {accuracy*100:.2f}%"
        )

        self.model = MultinomialNB()

        self.model.fit(
            X,
            y
        )

        with open(model_path, "wb") as f:
            pickle.dump(
                (
                    self.vectorizer,
                    self.model
                ),
                f
            )

    def create_widgets(self):

        title = tk.Label(
            self.root,
            text="📧 Spam Message Detector",
            font=("Arial", 18, "bold")
        )

        title.pack(pady=10)

        self.text_input = tk.Text(
            self.root,
            height=8,
            width=50,
            font=("Arial", 11)
        )

        self.text_input.pack(pady=10)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack()

        detect_btn = tk.Button(
            btn_frame,
            text="Detect",
            bg="blue",
            fg="white",
            width=15,
            command=self.detect
        )

        detect_btn.pack(
            side=tk.LEFT,
            padx=10
        )

        clear_btn = tk.Button(
            btn_frame,
            text="Clear",
            width=15,
            command=self.clear
        )

        clear_btn.pack(
            side=tk.LEFT,
            padx=10
        )

        self.result_label = tk.Label(
            self.root,
            text="Enter message and click detect",
            font=("Arial", 14)
        )

        self.result_label.pack(
            pady=20
        )

        self.prob_label = tk.Label(
            self.root,
            text="",
            font=("Arial", 11)
        )

        self.prob_label.pack()

    def detect(self):

        text = self.text_input.get(
            "1.0",
            tk.END
        ).strip()

        if not text:
            messagebox.showwarning(
                "Warning",
                "Enter a message"
            )
            return

        cleaned = self.clean_text(
            text
        )

        vec = self.vectorizer.transform(
            [cleaned]
        )

        prediction = self.model.predict(
            vec
        )[0]

        prob = self.model.predict_proba(
            vec
        )[0]

        spam_index = list(
            self.model.classes_
        ).index("spam")

        spam_prob = prob[
            spam_index
        ]

        if prediction == "spam":

            self.result_label.config(
                text="🚨 SPAM MESSAGE",
                fg="red"
            )

            self.prob_label.config(
                text=f"Spam Probability: {spam_prob*100:.1f}%"
            )

        else:

            self.result_label.config(
                text="✅ NORMAL MESSAGE",
                fg="green"
            )

            self.prob_label.config(
                text=f"Ham Probability: {(1-spam_prob)*100:.1f}%"
            )

    def clear(self):

        self.text_input.delete(
            "1.0",
            tk.END
        )

        self.result_label.config(
            text="Enter message and click detect"
        )

        self.prob_label.config(
            text=""
        )


def main():

    root = tk.Tk()

    app = SpamDetectorGUI(
        root
    )

    root.mainloop()


if __name__ == "__main__":
    main()