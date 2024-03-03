from django.shortcuts import render
from django.http import HttpResponse 
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.views.decorators.http import require_POST
from .sign_language_recognizer import SignLanguageRecognizer
import tensorflow as tf
import numpy as np
import cv2
import base64
import json
from keras.models import model_from_json
from string import ascii_uppercase
import operator
from spellchecker import SpellChecker
# Create your views here.


def home(request):
    return render(request, 'home.html')

def about(request):
    # return HttpResponse('This is about page')
    return render(request, 'about.html')

def register(request):
    return render(request, 'register.html')

def login(request):
    return render(request, 'login.html')

class SignLanguageRecognizer:
    def __init__(self):
        self.directory = 'model/'
        self.loaded_model = self.load_model()
        self.spell_checker = SpellChecker()

    def load_model(self):
        # Load the model architecture from the JSON file
        with open('D:/Final Year project/Project/unitalk/student/model/model-bw.json', 'r') as json_file:
            model_json = json_file.read()

        # Recreate the model from the JSON string
        loaded_model = tf.keras.models.model_from_json(model_json)

        # Load the weights into the model
        loaded_model.load_weights('D:/Final Year project/Project/unitalk/student/model/model-bw.h5')

        return loaded_model

    def preprocess_frame(self, frame):
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 2)
        # Apply adaptive thresholding
        th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        # Apply Otsu's thresholding
        _, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Resize frame to match the input size of the CNN model
        resized_frame = cv2.resize(res, (128, 128))
        # Normalize frame
        resized_frame = resized_frame / 255.0
        return resized_frame

    def recognize_sign_language(self, frame):
        # Preprocess the frame
        resized_frame = self.preprocess_frame(frame)
        
        # Predict the hand sign using the loaded CNN model
        result = self.loaded_model.predict(resized_frame.reshape(1, 128, 128, 1))
        
        # Convert the result to a symbol
        prediction = {}
        prediction['blank'] = result[0][0]
        inde = 1
        for i in ascii_uppercase:
            prediction[i] = result[0][inde]
            inde += 1
        # Sort the predictions
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        current_symbol = prediction[0][0]
        
        # Suggest alternative words using a spell checker
        word_suggestions = self.spell_checker.correction(current_symbol)

        return current_symbol, word_suggestions

# Initialize the SignLanguageRecognizer object
sign_language_recognizer = SignLanguageRecognizer()

# Instantiate SpellChecker
spell_checker = SpellChecker()

@require_http_methods(["GET", "POST"])
def process_image(request):
    if request.method == 'GET':
        return render(request, 'recognize_sign_language.html')
    elif request.method == 'POST':
        # Decode and preprocess the posted image data
        image_data = request.POST.get('image_data')
        image_data = image_data.split(",")[1]  # Remove the data URI prefix
        decoded_data = base64.b64decode(image_data)
        np_data = np.frombuffer(decoded_data, dtype=np.uint8)
        frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

        # Use SignLanguageRecognizer to recognize sign language
        current_symbol, word_suggestions = sign_language_recognizer.recognize_sign_language(frame)

        # Print the predicted text to the terminal
        print("Predicted Text:", current_symbol)

        # Keep track of recognized symbols over multiple frames to form a sentence
        # (You may need to implement more sophisticated logic depending on your requirements)
        # For simplicity, let's assume each recognized symbol is a word
        if 'sentence' not in request.session:
            request.session['sentence'] = []
        request.session['sentence'].append(current_symbol)

        # Get the predicted symbol
        predicted_symbol = current_symbol  # You can adjust this based on your implementation

        # Return the current sentence, predicted symbol, and word suggestions as a JSON response
        current_sentence = ' '.join(request.session['sentence'])
        return JsonResponse({'current_sentence': current_sentence, 'predicted_symbol': predicted_symbol, 'word_suggestions': word_suggestions})
    else:
        # Return an error response for other types of requests
        return JsonResponse({'error': 'Invalid Request'}, status=400)



#1
# @require_http_methods(["GET", "POST"])
# def process_image(request):
#     if request.method == 'GET':
#         return render(request, 'recognize_sign_language.html')
#     elif request.method == 'POST':
#         # Decode and preprocess the posted image data
#         image_data = request.POST.get('image_data')
#         image_data = image_data.split(",")[1]  # Remove the data URI prefix
#         decoded_data = base64.b64decode(image_data)
#         np_data = np.frombuffer(decoded_data, dtype=np.uint8)
#         frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

#         # Use SignLanguageRecognizer to recognize sign language
#         current_symbol, word_suggestions = sign_language_recognizer.recognize_sign_language(frame)

#         # Keep track of recognized symbols over multiple frames to form a sentence
#         # (You may need to implement more sophisticated logic depending on your requirements)
#         # For simplicity, let's assume each recognized symbol is a word
#         if 'sentence' not in request.session:
#             request.session['sentence'] = []
#         request.session['sentence'].append(current_symbol)

#         # Return the current sentence as a JSON response
#         current_sentence = ' '.join(request.session['sentence'])
#         return JsonResponse({'current_sentence': current_sentence, 'word_suggestions': word_suggestions})
#     else:
#         # Return an error response for other types of requests
#         return JsonResponse({'error': 'Invalid Request'}, status=400)


#3
# @require_http_methods(["GET", "POST"])
# def process_image(request):
#     if request.method == 'GET':
#         return render(request, 'recognize_sign_language.html')
#     elif request.method == 'POST':
#         try:
#             # Decode and preprocess the posted image data
#             image_data = request.POST.get('image_data')
#             image_data = image_data.split(",")[1]  # Remove the data URI prefix
#             decoded_data = base64.b64decode(image_data)
#             np_data = np.frombuffer(decoded_data, dtype=np.uint8)
#             frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
            
#             # Use SignLanguageRecognizer to recognize sign language
#             current_symbol, word_suggestions = sign_language_recognizer.recognize_sign_language(frame)

#             # Keep track of recognized symbols over multiple frames to form a sentence
#             if 'sentence' not in request.session:
#                 request.session['sentence'] = []
#             request.session['sentence'].append(current_symbol)

#             # Return the current sentence as a JSON response
#             current_sentence = ' '.join(request.session['sentence'])
#             return JsonResponse({'current_sentence': current_sentence, 'word_suggestions': word_suggestions})
#         except Exception as e:
#             return JsonResponse({'error': str(e)}, status=400)
#     else:
#         # Return an error response for other types of requests
#         return JsonResponse({'error': 'Invalid Request'}, status=400)


#2
# @require_http_methods(["GET", "POST"])
# def process_image(request):
#     if request.method == 'GET':
#         return render(request, 'recognize_sign_language.html')
#     elif request.method == 'POST':
#         # Decode and preprocess the posted image data
#         image_data = request.POST.get('image_data')
#         image_data = image_data.split(",")[1]  # Remove the data URI prefix
#         decoded_data = base64.b64decode(image_data)
#         np_data = np.frombuffer(decoded_data, dtype=np.uint8)
#         frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        
#         # Convert frame to grayscale
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
#         # Preprocess the frame
#         resized_frame = sign_language_recognizer.preprocess_frame(gray_frame)

#         # Use SignLanguageRecognizer to recognize sign language
#         current_symbol, word_suggestions = sign_language_recognizer.recognize_sign_language(resized_frame)

#         # Return the predicted text as JSON response
#         return JsonResponse({'predicted_text': current_symbol, 'word_sequence': word_suggestions})
#     else:
#         # Return an error response for other types of requests
#         return JsonResponse({'error': 'Invalid Request'}, status=400)
