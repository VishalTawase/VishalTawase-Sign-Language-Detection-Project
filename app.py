# from flask import Flask, render_template, Response
# import cv2
# from cvzone.HandTrackingModule import HandDetector
# from cvzone.ClassificationModule import Classifier
# import numpy as np
# import math
# import webbrowser
# import threading

# app = Flask(__name__)

# cap = cv2.VideoCapture(0)
# detector = HandDetector(maxHands=1)
# classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# offset = 20
# imgSize = 300
# labels = ["Hello", "Okay", "Please", "Thank you", "Yes", "No"]

# def generate_frames():
#     while True:
#         success, img = cap.read()
#         if not success:
#             break
#         else:
#             imgOutput = img.copy()
#             hands, img = detector.findHands(img)
#             if hands:
#                 hand = hands[0]
#                 x, y, w, h = hand['bbox']

#                 imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
#                 imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]

#                 aspectRatio = h / w

#                 if aspectRatio > 1:
#                     k = imgSize / h
#                     wCal = math.ceil(k * w)
#                     imgResize = cv2.resize(imgCrop, (wCal, imgSize))
#                     wGap = math.ceil((imgSize - wCal) / 2)
#                     imgWhite[:, wGap: wCal + wGap] = imgResize
#                 else:
#                     k = imgSize / w
#                     hCal = math.ceil(k * h)
#                     imgResize = cv2.resize(imgCrop, (imgSize, hCal))
#                     hGap = math.ceil((imgSize - hCal) / 2)
#                     imgWhite[hGap: hCal + hGap, :] = imgResize

#                 prediction, index = classifier.getPrediction(imgWhite, draw=False)
#                 label = labels[index]

#                 cv2.rectangle(imgOutput, (x-offset, y-offset-70), (x -offset+400, y - offset+60-50), (0,255,0), cv2.FILLED)  
#                 cv2.putText(imgOutput, label, (x, y-30), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,0), 2) 
#                 cv2.rectangle(imgOutput, (x-offset, y-offset), (x + w + offset, y+h + offset), (0,255,0), 4)

#             _, buffer = cv2.imencode('.jpg', imgOutput)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video')
# def video():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# def open_browser():
#     webbrowser.open_new("http://127.0.0.1:5000/")

# if __name__ == '__main__':
#     threading.Timer(1.25, open_browser).start()  # Open browser after 1.25 seconds
#     app.run(debug=True)


# from flask import Flask, render_template, Response
# import cv2
# from cvzone.HandTrackingModule import HandDetector
# from cvzone.ClassificationModule import Classifier
# import numpy as np
# import math
# import webbrowser
# import threading
# import atexit

# app = Flask(__name__)

# # Initialize Video Capture
# cap = cv2.VideoCapture(0)
# detector = HandDetector(maxHands=1)
# classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# offset = 20
# imgSize = 300
# labels = ["Hello", "Okay", "Please", "Thank you", "Yes", "No"]
# last_frame = None  # Store last detected frame

# def generate_frames():
#     global last_frame
#     while True:
#         success, img = cap.read()
#         if not success:
#             break
#         else:
#             imgOutput = img.copy()
#             hands, img = detector.findHands(img)

#             if hands:
#                 hand = hands[0]
#                 x, y, w, h = hand['bbox']

#                 # Ensure valid cropping
#                 y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
#                 x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
#                 imgCrop = img[y1:y2, x1:x2]

#                 aspectRatio = h / w

#                 if imgCrop.size != 0:
#                     imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

#                     if aspectRatio > 1:
#                         k = imgSize / h
#                         wCal = math.ceil(k * w)
#                         imgResize = cv2.resize(imgCrop, (wCal, imgSize))
#                         wGap = math.ceil((imgSize - wCal) / 2)
#                         imgWhite[:, wGap: wCal + wGap] = imgResize
#                     else:
#                         k = imgSize / w
#                         hCal = math.ceil(k * h)
#                         imgResize = cv2.resize(imgCrop, (imgSize, hCal))
#                         hGap = math.ceil((imgSize - hCal) / 2)
#                         imgWhite[hGap: hCal + hGap, :] = imgResize

#                     prediction, index = classifier.getPrediction(imgWhite, draw=False)
#                     label = labels[index]

#                     # Draw detection box and label
#                     cv2.rectangle(imgOutput, (x-offset, y-offset-70), (x -offset+400, y - offset+60-50), (0,255,0), cv2.FILLED)  
#                     cv2.putText(imgOutput, label, (x, y-30), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,0), 2) 
#                     cv2.rectangle(imgOutput, (x-offset, y-offset), (x + w + offset, y+h + offset), (0,255,0), 4)

#                     last_frame = imgOutput  # Save last detected frame

#             else:
#                 # If no hand is detected, show last valid frame
#                 if last_frame is not None:
#                     imgOutput = last_frame.copy()
#                 cv2.putText(imgOutput, "No Hand Detected", (50, 50), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

#             _, buffer = cv2.imencode('.jpg', imgOutput)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# # Sign language meanings
# # sign_notes = {
# #     "Hello": "Used as a greeting or to call attention.",
# #     "Okay": "Indicates agreement or that everything is fine.",
# #     "Please": "A polite request or expression of courtesy.",
# #     "Thank you": "Used to express gratitude.",
# #     "Yes": "Indicates agreement or affirmation.",
# #     "No": "Indicates disagreement or refusal."
# # }

# sign_notes = {
#     "Hello": {
#         "meaning": "Used as a greeting or to call attention.",
#         "image": "hello.png"
#     },
#     "Okay": {
#         "meaning": "Indicates agreement or that everything is fine.",
#         "image": "okay.png"
#     },
#     "Please": {
#         "meaning": "A polite request or expression of courtesy.",
#         "image": "please.png"
#     },
#     "Thank you": {
#         "meaning": "Used to express gratitude.",
#         "image": "thankyou.png"
#     },
#     "Yes": {
#         "meaning": "Indicates agreement or affirmation.",
#         "image": "yes.png"
#     },
#     "No": {
#         "meaning": "Indicates disagreement or refusal.",
#         "image": "no.png"
#     }
# }


# @app.route('/')
# def index():
#     return render_template('index.html', sign_notes=sign_notes)

# @app.route('/video')
# def video():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# def open_browser():
#     """Open the web browser only once"""
#     threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:5000/")).start()

# def cleanup():
#     """Cleanup resources before exiting"""
#     print("\nExecution Completed! Cleaning up resources...")
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     atexit.register(cleanup)  # Ensure cleanup when the script exits
#     open_browser()  # Open browser only once
#     app.run(debug=True)





# from flask import Flask, render_template, Response
# import cv2
# from cvzone.HandTrackingModule import HandDetector
# from cvzone.ClassificationModule import Classifier
# import numpy as np
# import math
# import webbrowser
# import threading
# import atexit

# app = Flask(__name__)

# # Initialize Video Capture
# cap = cv2.VideoCapture(0)
# detector = HandDetector(maxHands=1)
# classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# offset = 20
# imgSize = 300
# labels = ["Hello", "Okay", "Please", "Thank you", "Yes", "No"]
# last_frame = None  # Store last detected frame

# def generate_frames():
#     global last_frame
#     while True:
#         success, img = cap.read()
#         if not success:
#             break
#         else:
#             imgOutput = img.copy()
#             hands, img = detector.findHands(img)

#             if hands:
#                 hand = hands[0]
#                 x, y, w, h = hand['bbox']

#                 # Ensure valid cropping
#                 y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
#                 x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
#                 imgCrop = img[y1:y2, x1:x2]

#                 aspectRatio = h / w

#                 if imgCrop.size != 0:
#                     imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

#                     if aspectRatio > 1:
#                         k = imgSize / h
#                         wCal = math.ceil(k * w)
#                         imgResize = cv2.resize(imgCrop, (wCal, imgSize))
#                         wGap = math.ceil((imgSize - wCal) / 2)
#                         imgWhite[:, wGap: wCal + wGap] = imgResize
#                     else:
#                         k = imgSize / w
#                         hCal = math.ceil(k * h)
#                         imgResize = cv2.resize(imgCrop, (imgSize, hCal))
#                         hGap = math.ceil((imgSize - hCal) / 2)
#                         imgWhite[hGap: hCal + hGap, :] = imgResize

#                     prediction, index = classifier.getPrediction(imgWhite, draw=False)
#                     label = labels[index]

#                     # Draw detection box and label
#                     cv2.rectangle(imgOutput, (x-offset, y-offset-70), (x -offset+400, y - offset+60-50), (0,255,0), cv2.FILLED)  
#                     cv2.putText(imgOutput, label, (x, y-30), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,0), 2) 
#                     cv2.rectangle(imgOutput, (x-offset, y-offset), (x + w + offset, y+h + offset), (0,255,0), 4)

#                     last_frame = imgOutput  # Save last detected frame

#             else:
#                 # If no hand is detected, show last valid frame
#                 if last_frame is not None:
#                     imgOutput = last_frame.copy()
#                 cv2.putText(imgOutput, "No Hand Detected", (50, 50), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

#             _, buffer = cv2.imencode('.jpg', imgOutput)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# # Sign language meanings
# sign_notes = {
#     "Hello": {"meaning": "Used as a greeting or to call attention.", "image": "hello.png"},
#     "Okay": {"meaning": "Indicates agreement or that everything is fine.", "image": "okay.png"},
#     "Please": {"meaning": "A polite request or expression of courtesy.", "image": "please.png"},
#     "Thank you": {"meaning": "Used to express gratitude.", "image": "thankyou.png"},
#     "Yes": {"meaning": "Indicates agreement or affirmation.", "image": "yes.png"},
#     "No": {"meaning": "Indicates disagreement or refusal.", "image": "no.png"}
# }

# @app.route('/')
# def home():
#     return render_template('home.html')

# @app.route('/demo')
# def index():
#     return render_template('index.html', sign_notes=sign_notes)

# @app.route('/video')
# def video():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# def open_browser():
#     """Open the web browser only once"""
#     threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:5000/")).start()

# def cleanup():
#     """Cleanup resources before exiting"""
#     print("\nExecution Completed! Cleaning up resources...")
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     atexit.register(cleanup)  # Ensure cleanup when the script exits
#     open_browser()  # Open browser only once
#     app.run(debug=True)



import cv2
import numpy as np
import base64
from flask import Flask, render_template, request, jsonify
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import math
import webbrowser
import threading

app = Flask(__name__)

detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300
labels = ["Hello", "Okay", "Please", "Thank you", "Yes", "No"]
last_frame = None

# Sign language meanings
sign_notes = {
    "Hello": {"meaning": "Used as a greeting or to call attention.", "image": "hello.png"},
    "Okay": {"meaning": "Indicates agreement or that everything is fine.", "image": "okay.png"},
    "Please": {"meaning": "A polite request or expression of courtesy.", "image": "please.png"},
    "Thank you": {"meaning": "Used to express gratitude.", "image": "thankyou.png"},
    "Yes": {"meaning": "Indicates agreement or affirmation.", "image": "yes.png"},
    "No": {"meaning": "Indicates disagreement or refusal.", "image": "no.png"}
}

def open_browser():
    """Open the web browser only once"""
    threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:5000/")).start()

def preprocess_image(image_data):
    """Decode and preprocess the image for hand detection."""
    image_data = image_data.split(",")[1]  # Remove metadata
    image_bytes = base64.b64decode(image_data)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    global last_frame
    
    # Detect hand
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]

        aspectRatio = h / w
        if imgCrop.size != 0:
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap: wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap: hCal + hGap, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            label = labels[index]
        
            # Draw bounding box and label
            cv2.rectangle(imgOutput, (x-offset, y-offset-70), (x -offset+400, y - offset+60-50), (0,255,0), cv2.FILLED)  
            cv2.putText(imgOutput, label, (x, y-30), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,0), 2) 
            cv2.rectangle(imgOutput, (x-offset, y-offset), (x + w + offset, y+h + offset), (0,255,0), 4)

            last_frame = imgOutput  
        else:  
            if last_frame is not None:
                    imgOutput = last_frame.copy()
            cv2.putText(imgOutput, "No Hand Detected", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
    else:  
        if last_frame is not None:
            imgOutput = last_frame.copy()
        cv2.putText(imgOutput, "No Hand Detected", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    return imgOutput, label if hands else "No Hand Detected"


@app.route("/")
def home():
    """Render the homepage."""
    return render_template("home.html")

@app.route("/demo")
def index():
    """Render the main sign detection page."""
    return render_template("index.html", sign_notes=sign_notes)

@app.route("/video")
def video():
    """This route is no longer needed since we switched to image-based predictions."""
    return jsonify({"message": "Video feed disabled. Use /predict for image-based detection."})

@app.route("/predict", methods=["POST"])
def predict():
    """Handle image predictions from the frontend."""
    data = request.json
    if "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    imgOutput, result_label = preprocess_image(data["image"])
    
    if imgOutput is not None:
        _, buffer = cv2.imencode('.jpg', imgOutput)
        img_base64 = base64.b64encode(buffer).decode("utf-8")
        return jsonify({"prediction": result_label, "image": img_base64})
    else:
        return jsonify({"error": "Processing failed"}), 500


if __name__ == "__main__":
    open_browser()  # Open browser only once
    app.run(debug=True)

