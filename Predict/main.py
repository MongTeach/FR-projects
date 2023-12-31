import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from keras.models import load_model

class FaceClassificationApp:
    def __init__(self):
    
        self.model = load_model('MOBILENETV1-Jan-2024_best.hdf5')#ganti dengan nama model yang di miliki
        self.class_labels = ['Putro', 'aser', 'ezrah', 'ferel', 'frits', 'ichal', 'jorgi', 'karen', 'mong', 'reki', 'riki']#sesuaikan dengan class yang di gunakan
        
        self.root = tk.Tk()
        self.root.title("Face Classification")
        self.show_gui = True
        
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        window_width = int(screen_width * 0.2)
        window_height = int(screen_height * 0.4)

        self.root.geometry(f"{window_width}x{window_height}")

        self.label = tk.Label(self.root, text="Face Classification", font=("Helvetica", 16), justify="left")
        self.label.pack()

        self.probabilities = [tk.Label(self.root, text=f"{self.class_labels[i]}: ", font=("Helvetica", 14), justify="left") for i in range(len(self.class_labels))]
        for prob_label in self.probabilities:
            prob_label.pack(anchor='w')

        self.root.protocol("WM_DELETE_WINDOW", self.prevent_close)
        self.root.bind('<KeyPress>', self.handle_key)  # Menambahkan event handler untuk semua tombol keyboard
        
    def handle_key(self, event):
        if event.char == 'q':
            self.show_gui = False
            self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)

    def preprocess_image(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img_gray, (48, 48))
        img_resized = np.expand_dims(img_resized, axis=-1)
        img_resized = np.expand_dims(img_resized, axis=0)
        return img_resized

    def predict_with_webcam(self):
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        while self.show_gui:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                img = self.preprocess_image(face_img)
                preds = self.model.predict(img)
                
                for i, prob_label in enumerate(self.probabilities):
                    prob_label.config(text=f"{self.class_labels[i]}: {preds[0][i]:.4f}")

                label_id = np.argmax(preds[0])
                label = self.class_labels[label_id]
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f'Prediction: {label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            
            cv2.imshow('Webcam Feed', frame)

            self.root.update_idletasks()
            self.root.update()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.show_gui = False
                self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)

        cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()

    def prevent_close(self):
        pass  # Implementasi logika untuk mencegah penutupan GUI jika diperlukan

    def run(self):
        self.predict_with_webcam()

app = FaceClassificationApp()
app.run()
