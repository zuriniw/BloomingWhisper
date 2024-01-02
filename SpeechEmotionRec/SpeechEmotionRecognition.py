##############################################
# Crop 64*64 images from Mel spectrogram
##############################################
with open('mel_and_label.pkl','rb') as f:
    mel_and_label = pickle.load(f)
# After generating the Mel spectrogram, crop it. The paper uses a 315ms sliding step, but that doesn't generate enough data.
# Using a stride of 20 columns of pixels for sliding
crop_stride = 20
# The labels and 64*64 Mel images are stored in a list, with labels named like 'W_1_1', indicating the emotion type, the 1st audio segment, and the 1st segment
train_and_label_64 = []
# Crop the image
for i in range(len(mel_and_label)):
    temp = mel_and_label[i][1] # Mel image
    h, w, c = np.shape(mel_and_label[i][1]) # h=64, c=3
    # Calculate how many 64*64 images can be cropped from the Mel image
    num_crop = int(np.floor((w-64)/crop_stride)) + 1
    for j in range(num_crop):
        # Crop a 64*64*3 area
        temp1 = temp[:, j*20:j*20+64, :]
        # Data structure of train_and_label_64 is a list of length 5944
        # Each element is a 2D list
        # The 1st dimension of the 2D list is the label, e.g., 'W_1_1'
        # The 2nd dimension is a 64×64×3 image
        train_and_label_64.append([mel_and_label[i][0]+'_'+str(j), temp1])

# Resize to 227*227*3
# Be careful with memory usage, execute in segments
train_and_label_227 = []
for i in range(len(train_and_label_64)):
    temp = cv2.resize(train_and_label_64[i][1], (227,227)).astype('float16')
    train_and_label_227.append([train_and_label_64[i][0], temp])
# Data structure of train_and_label_227 is a list of length 5944
# Each element is a 2D list
# The 1st dimension of the 2D list is the label, e.g., 'W_1_1'
# The 2nd dimension is a 227×227×3 image
# Training data
train_227 = []
for i in range(len(train_and_label_227)):
    train_227.append(train_and_label_227[i][1])
train_227 = np.array(train_227) # Convert training data to ndarray format
# Define the labels
# A ==> 0
# B ==> 1
# C ==> 2
# D ==> 3
# E ==> 4
label_227 = []
for i in range(len(train_and_label_227)):
    if train_and_label_227[i][0][0] == 'A':
        label_227.append(0)
    if train_and_label_227[i][0][0] == 'B':
        label_227.append(1)
    if train_and_label_227[i][0][0] == 'C':
        label_227.append(2)
    if train_and_label_227[i][0][0] == 'D':
        label_227.append(3)
    if train_and_label_227[i][0][0] == 'E':
        label_227.append(4)
# Convert 0-4 labels to one-hot encoding
from keras.utils.np_utils import to_categorical
label_227_one_hot = to_categorical(label_227)
import keras
from keras.models import Sequential
from keras.layers import Input,Dense,Conv2D,MaxPooling2D,UpSampling2D,Dropout,Flatten
from keras.layers import BatchNormalization,AveragePooling2D,concatenate
from keras.layers import ZeroPadding2D,add
from keras.layers import Dropout, Activation
from keras.models import Model,load_model
from keras.utils.np_utils to_categorical
from keras.callbacks TensorBoard
from keras import optimizers, regularizers # Optimizers, regularization
from keras.optimizers import SGD, Adam
# Parameter settings
batch_size = 32
epochs = 50
img_rows, img_cols = 227, 227 # Input image dimensions
input_shape = (img_rows, img_cols, 3)
# Path to save the model
model_name = './model/alexnet_50epoch_5class.h5'  # Modify the model name to reflect the number of classes

X_train = train_227
y_train = label_227_one_hot  # Make sure this is the one-hot encoding for 5 classes

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')

# AlexNet model
model = Sequential()
model.add(Conv2D(96,(11,11),strides=(4,4),input_shape=(227,227,3),padding='valid',activation='relu',kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
model.add(Conv2D(256,(5,5),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
model.add(Flatten())
model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))  # Modify the output layer to have 5 neurons
model.summary()

# Compile the model
sgd = optimizers.SGD(momentum=0.9, lr=0.001, decay=1e-6)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(X_train, y_train,
          epochs=epochs,
          batch_size=batch_size,
          shuffle=True, verbose=1)

model.save(model_name)

# Evaluate the model
score = model.evaluate(X_train, y_train, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])
# Extract the 4096-dimension fully connected layer as the output segment vector
segment_feature_model = Model(inputs=model.input, outputs=model.layers[11].output)
# Use the prediction of this model as output
segment_feature = segment_feature_model.predict(train_227)
# Record the number of segments for each audio
count = np.zeros((535), np.int16)
for i in range(len(train_and_label_227)):
    count[int(train_and_label_227[i][0].split('_')[1])] += 1
# Adjust the number of segments for each audio to be a multiple of 4
for i in range(len(count)):
    count[i] -= count[i] % 4
# Create a list to record which audio each segment belongs to
segment_belong = []
for i in range(len(train_and_label_227)):
    segment_belong.append(int(train_and_label_227[i][0].split('_')[1]))
# Create a list to record the position of the first segment of each audio in 5944 (for memory addressing)
first_segment_dir = []
flag = 0
for i in range(len(segment_belong)):
    if segment_belong[i] == flag:
        first_segment_dir.append(i)
        flag += 1
# Store the multiple of 4 4096-dimension features for each utterance and the corresponding classification label
utterance_segment_feature_and_label = []
# Iterate over all utterances, 535 times
for i in range(len(count)):
    if i == 247: # count[247] == 0
        # Concatenate only one 4096-dimension vector
        utterance_segment_feature_and_label.append([mel_and_label[i][0][0], [segment_feature[first_segment_dir[i]]]])
    if i != 247:
        temp = []  # Contains two elements, label and several 4096-dimension vectors
        temp1 = [] # Stores several 4096-dimension vectors for an utterance
        temp.append(mel_and_label[i][0][0]) # Letter WFLEATN, emotional label
        # Address of the first segment of the ith utterance in segment_feature
        first_dir = first_segment_dir[i]
        # Address offset
        add_dir = 0
        for j in range(count[i]): # count[i] stores how many segments each utterance corresponds to
            temp1.append(segment_feature[first_dir + add_dir])
            add_dir += 1 # Pointer points to the next segment of the ith utterance
        temp.append(temp1)
        utterance_segment_feature_and_label.append(temp)
    if i % 10 == 0:
        print(i)

##########################################
# Feature Pooling
##########################################
with open('utterance_segment_feature_and_label.pkl', 'rb') as f:
    utterance_segment_feature_and_label = pickle.load(f)
# Mean pooling
layer1 = [] # Pyramid layer 1 feature, 535 items
layer2 = [] # Pyramid layer 2 feature, 535*2 items
layer3 = [] # Pyramid layer 3 feature, 535*4 items

for i in range(len(utterance_segment_feature_and_label)):
    # Average pooling for the first layer
    avg_layer1 = np.zeros((4096))
    for j in range(len(utterance_segment_feature_and_label[i][1])):
        avg_layer1 += utterance_segment_feature_and_label[i][1][j]
    avg_layer1 /= len(utterance_segment_feature_and_label[i][1])
    layer1.append(avg_layer1)

    # Average pooling for the second layer
    avg_layer2 = np.zeros((2,4096))
    for j in range(int(len(utterance_segment_feature_and_label[i][1])/2)):
        avg_layer2[0] += utterance_segment_feature_and_label[i][1][j]
    for j in range(int(len(utterance_segment_feature_and_label[i][1])/2)):
        avg_layer2[1] += utterance_segment_feature_and_label[i][1][j + int(len(utterance_segment_feature_and_label[i][1])/2)]
    if i != 247:
        avg_layer2 /= int(len(utterance_segment_feature_and_label[i][1])/2)
    if i == 247:
        avg_layer2 /= len(utterance_segment_feature_and_label[i][1])
    layer2.append(avg_layer2)

    # Average pooling for the third layer
    avg_layer3 = np.zeros((4,4096))
    for j in range(int(len(utterance_segment_feature_and_label[i][1])/4)):
        avg_layer3[0] += utterance_segment_feature_and_label[i][1][j]
    for j in range(int(len(utterance_segment_feature_and_label[i][1])/4)):
        avg_layer3[1] += utterance_segment_feature_and_label[i][1][j + int(len(utterance_segment_feature_and_label[i][1])/4)]
    for j in range(int(len(utterance_segment_feature_and_label[i][1])/4)):
        avg_layer3[2] += utterance_segment_feature_and_label[i][1][j + 2 * int(len(utterance_segment_feature_and_label[i][1])/4)]
    for j in range(int(len(utterance_segment_feature_and_label[i][1])/4)):
        avg_layer3[3] += utterance_segment_feature_and_label[i][1][j + 3 * int(len(utterance_segment_feature_and_label[i][1])/4)]
    if i != 247:
        avg_layer3 /= int(len(utterance_segment_feature_and_label[i][1])/4)
    if i == 247:
        avg_layer3 /= len(utterance_segment_feature_and_label[i][1])
    layer3.append(avg_layer3)

# Utterance-level feature pooling
# Mean pooling
utterance_feature = np.zeros((535,4096))
for i in range(len(utterance_feature)):
    utterance_feature[i] += 0.25 * layer1[i] + 0.5 * layer2[i][0] + 0.5 * layer2[i][1] + layer3[i][0] + layer3[i][1] + layer3[i][2] + layer3[i][3]
    utterance_feature[i] /= 7
utterance_label = np.zeros((535))
for i in range(len(utterance_label)):
    if utterance_segment_feature_and_label[i][0] == 'A':
        utterance_label[i] = 0
    if utterance_segment_feature_and_label[i][0] == 'B':
        utterance_label[i] = 1
    if utterance_segment_feature_and_label[i][0] == 'C':
        utterance_label[i] = 2
    if utterance_segment_feature_and_label[i][0] == 'D':
        utterance_label[i] = 3
    if utterance_segment_feature_and_label[i][0] == 'E':
        utterance_label[i] = 4

# Shuffle the data thoroughly
utterance_feature_label = []
for i in range(len(utterance_feature)):
    utterance_feature_label.append([utterance_label[i], utterance_feature[i]])
for i in range(100):
    random.shuffle(utterance_feature_label)
with open('utterance_feature_label_shuffle.pkl', 'wb') as f:
    pickle.dump(utterance_feature_label, f)

###############################################################################
# Draw ROC Curve
###############################################################################
# Load the data
with open('utterance_feature_label_shuffle.pkl', 'rb') as f:
    utterance_feature_label = pickle.load(f)
utterance_label = np.zeros((len(utterance_feature_label)))
utterance_feature = np.zeros((len(utterance_feature_label), len(utterance_feature_label[0][1])))
for i in range(len(utterance_feature_label)):
    utterance_label[i] = utterance_feature_label[i][0]
    utterance_feature[i] = utterance_feature_label[i][1]

# Set random seed
random_state = np.random.RandomState(0)
# Use n-fold cross-validation and draw ROC curve
# cv contains the indices of samples divided into n folds, separating into training and testing samples
cv = StratifiedKFold(utterance_label, n_folds=5)
classifier = svm.SVC(kernel='rbf', random_state=random_state)

mean_tpr = 0.0  # True positive rate
mean_fpr = np.linspace(0, 1, 100000)  # False positive rate
all_tpr = []

for i, (train, test) in enumerate(cv):
    # Train the model using the training data and test on the testing set
    predict_result = classifier.fit(utterance_feature[train], utterance_label[train]).predict(utterance_feature[test])
    # Convert the predicted results to one-hot encoding
    predict_result_one_hot = to_categorical(predict_result)
    # Convert the true labels to one-hot encoding
    true_label_one_hot = to_categorical(utterance_label[test])
    # Treat multi-classification as n binary classifications and draw n ROC curves
    fpr_avg = 0
    tpr_avg = 0
    for j in range(len(set(utterance_label))):
        # Compute the ROC curve and area under the curve
        fpr, tpr, thresholds = roc_curve(true_label_one_hot[:, j], predict_result_one_hot[:, j])
        fpr_avg += fpr
        tpr_avg += tpr
    fpr_avg /= len(set(utterance_label))
    tpr_avg /= len(set(utterance_label))
    # Interpolate mean_tpr at mean_fpr
    mean_tpr += interp(mean_fpr, fpr_avg, tpr_avg)
    mean_tpr[0] = 0.0  # Set the initial point to 0
    roc_auc = auc(fpr_avg, tpr_avg)
    # Plot the ROC curve
    plt.plot(fpr_avg, tpr_avg, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    plt.xscale('log')
    plt.xlim((1e-5, 1))

# Plot the diagonal line
#plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
plt.xscale('log')
plt.xlim((1e-5, 1))
mean_tpr /= len(cv)  # Calculate the mean at each point
mean_tpr[-1] = 1.0  # Set the last coordinate point to (1,1)
mean_auc = auc(mean_fpr, mean_tpr)  # Calculate the mean AUC value
# Plot the mean ROC curve
plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=1)

#plt.xlim([-0.05, 1.05])
plt.ylim([0, 1])
plt.grid(True)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig('ROC.png')
plt.show()

###############################################################################
# Draw Confusion Matrix
###############################################################################
# Load the data
with open('utterance_feature_label_shuffle.pkl', 'rb') as f:
    utterance_feature_label = pickle.load(f)
utterance_label = np.zeros((len(utterance_feature_label)))
utterance_feature = np.zeros((len(utterance_feature_label), len(utterance_feature_label[0][1])))
for i in range(len(utterance_feature_label)):
    utterance_label[i] = utterance_feature_label[i][0]
    utterance_feature[i] = utterance_feature_label[i][1]

classifier = svm.SVC(kernel='rbf', gamma='auto')
predict_result = classifier.fit(utterance_feature, utterance_label).predict(utterance_feature)
labels = list(set(utterance_label))
conf_mat = confusion_matrix(utterance_label, predict_result, labels = labels)
drawCM(conf_mat, 'confusion_matrix.png')

# Function to draw the confusion matrix
from __future__ import division
import numpy as np
from skimage import io, color
from PIL import Image, ImageDraw, ImageFont
import os
def drawCM(matrix, savname):
    # Display different color for different elements
    lines, cols = matrix.shape
    sumline = matrix.sum(axis=1).reshape(lines, 1)
    ratiomat = matrix / sumline
    toplot0 = 1 - ratiomat
    toplot = toplot0.repeat(100).reshape(lines, -1).repeat(100, axis=0)
    io.imsave(savname, color.gray2rgb(toplot))
    # Draw values on every block
    image = Image.open(savname)
    draw = ImageDraw.Draw(image)
    #font = ImageFont.truetype(os.path.join(os.getcwd(), "draw/ARIAL.TTF"), 15)
    for i in range(lines):
        for j in range(cols):
            dig = str(matrix[i, j])
            if i == j:
                filled = (255, 181, 197)
            else:
                filled = (46, 139, 87)
            draw.text((100 * j + 20, 100 * i + 20), dig, fill=filled)
    image.save(savname, 'jpeg')
