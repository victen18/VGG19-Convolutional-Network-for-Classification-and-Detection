from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import cv2

img_path = 'images/test8.jpg'
img = load_img(img_path)

img = img.resize((224, 224))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = imagenet_utils.preprocess_input(img_array)
pre_trained_model = VGG19(weights='imagenet')
prediction = pre_trained_model.predict(img_array)
actual_prediction = imagenet_utils.decode_predictions(prediction)
print(
    'Predicted object is: {} with accuracy of {}'.format(actual_prediction[0][0][1], actual_prediction[0][0][2] * 100))

disp_img = cv2.imread(img_path)
cv2.putText(disp_img, actual_prediction[0][0][1], (20, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 0))
cv2.imshow('Prediction', disp_img)
cv2.waitKey(0)
