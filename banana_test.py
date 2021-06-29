import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from tensorflow.keras.models import load_model
import imutils
image = cv2.imread('./69.jpg')
orig = image.copy()

#image = cv2.resize(image, (28 ,28))
image = cv2.resize(image, (224 ,224))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

print("[INFO] loading network...")
model = load_model('./vgg16_banana2.model', compile=False)

(immature, ripe, rotten) = model.predict(image)[0]
if immature > ripe and immature >rotten:
    label = "immature"
    proba = immature
elif ripe > immature and ripe >rotten:
    label = "ripe"
    proba = ripe
else:
    label = "rotten"
    proba = rotten
label = "{}: {:.2f}%".format(label, proba * 100)

output = imutils.resize(orig, width=400)
cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.imshow("Output", output)
cv2.imwrite('./output.png',output)
cv2.waitKey(0)