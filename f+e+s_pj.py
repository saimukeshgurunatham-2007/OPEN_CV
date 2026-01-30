
import cv2
import time
import tkinter as tk

# Load Haar cascades from OpenCV's built-in data folder
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")


def run_detector():
    capture = cv2.VideoCapture(0)

    # For FPS calculation (Frames Per Second)
    prev_time = 0

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # Region of interest (ROI)
            region_gray = gray[y:y + h, x:x + w]
            region_color = frame[y:y + h, x:x + w]

            # Detect eyes
            eyes = eye_cascade.detectMultiScale(region_gray, 1.1, 8)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(region_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

            # Detect smiles
            smiles = smile_cascade.detectMultiScale(region_gray, 1.7, 18)
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(region_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 255), 2)

        # FPS counter
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show window
        cv2.imshow("Smart Face Detector", frame)

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):   # Quit
            break
        elif key == ord('s'): # Save snapshot
            filename = f"snapshot_{int(time.time())}.png"
            cv2.imwrite(filename, frame)
            print(f"Snapshot saved as {filename}")

    capture.release()
    cv2.destroyAllWindows()


# GUI
root = tk.Tk()
root.geometry('190x280')
root.title("Smart Face Detector GUI")

start_btn = tk.Button(root, text="Start Detector",background='lightgreen', command=run_detector, width=20, height=2)
start_btn.pack(pady=10)

quit_btn = tk.Button(root, text="Quit",background='red',command=root.quit, width=20, height=2)
quit_btn.pack(pady=10)

root.mainloop()
