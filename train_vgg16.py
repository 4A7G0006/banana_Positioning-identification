from tensorflow.keras.layers import Input
from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
import os, cv2
import random
import numpy as np
from imutils import paths
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

num_classes = 3  # 分三類
vgg16_fine_tune = tf.keras.models.Sequential()  # 初始化模型 名稱為vgg16_fine_tune

# 跟LeNet一樣  .add()  不同在於之前我們一層一層加  這邊一次加進去整個vgg16
# include_top=False  代表我們不包含他的全連接層,我們要自己訓練
# input的形狀  要在第一層定義好我們接收的size
#imagenet才是遷移式學習的重點
vgg16_fine_tune.add(VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224,224,3))))

# 代表Dense層  跟 LeNet的一樣
# num_classes=分類數
# activation代表激活函數的選哪個,這邊選softmax
# kernel_initializer代表初始值  我們設定為0.001
new_output = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax, kernel_initializer=tf.initializers.Constant(0.001))
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()  # 對所有空數據也做最大池化

# 接著  我們要把剛剛我們定義的global_average_layer和new_output 加到模型裡
vgg16_fine_tune.add(global_average_layer)
vgg16_fine_tune.add(new_output)

l_layer = len(vgg16_fine_tune.layers)
# 這一塊其實是用loop設定vgg每一層可不可以被訓練
# 我們設定為不可被訓練
# 因為我們就是故意要用他訓練好的特徵提取的能力
for i in range(l_layer - 1):
    vgg16_fine_tune.layers[i].trainable = False

sgd = SGD(lr=1e-3, momentum=0.9)  # 代表設定learning rate = 0.01,momentum=0.9
vgg16_fine_tune.summary()

# 以下程式與LeNet相同 除了模型要改作vgg
print("[INFO] loading images......")
data = []
labels = []
imagePaths = paths.list_images('./banana_images')
imagePaths = list(imagePaths)
imagePaths = sorted(imagePaths)
random.seed(42)
random.shuffle(imagePaths)  # 打散圖片
classes_num = 3  # 總類別數 三類
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    data.append(image)
    label = imagePath.split(os.path.sep)[-2]  # sep判斷系統的'\' '/' 問題  #假如路徑 '/images/santa/000001.jpg' 分割 '/' 取[-2] santa

    if label == "immature":
        label = 0
    elif label == "ripe":
        label = 1
    elif label == "rotten":  # 三類
        label = 2

    labels.append(label)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)  # 資料分類 訓練資料與測試資料 0.75 0.25這樣
# trainY=[0,1,0,1,0,0,0,1]
# =>[[1,0],[0,1],[1,0],[1,0],[1,0]] # ONE HOT encoding
trainY = to_categorical(trainY, num_classes=classes_num)
testY = to_categorical(testY, num_classes=classes_num)

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")  # 圖像翻轉 加強學習

print("[INFO] compiling model...")

EPOCHS = 200  # 訓練參數設定 #訓練次數25次
INIT_LR = 1e-3  # 訓練率 靠體感
BS = 32  # 每次圖檔抓取32張 疊代ㄧ次32張

vgg16_fine_tune.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=sgd)  # 給定模型的loss function為categorical_crossentropy 三分類損失 評估指標選用accuracy
# 因為剛剛使用資料增強 所以使用fit訓練 同時避免記憶體不足 如果不使用就使用ㄧ般generator就好
# aug.flow設定資料流 傳入圖像訓練集X 與 訓練標籤 Y  #validation_data傳入測試資料集與標籤  如同前面訓練資料ㄧ般 #steps_per_epoch訓練資料數據除以批次訓練大小就會達到一個新的EPOCH   #EPOCH跑25遍資料 #verbose打印資料訓練log
print("[INFO] training network")
H = vgg16_fine_tune.fit_generator(aug.flow(trainX, trainY, batch_size=BS), validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
                                  epochs=EPOCHS, verbose=1)
print("[INFO] serializing network")
vgg16_fine_tune.save('./vgg16_banana2.model', save_format="h5")

plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on banana")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('./plot100.png')
