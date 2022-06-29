# moduel import
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import os

import nibabel as nib
import cv2

from skimage.transform import resize
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras.layers import concatenate
from tensorflow.keras.utils import normalize
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

from sklearn.utils import class_weight
import tensorflow.keras.optimizers as optim
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
import numpy as np
from keras import backend as K
import tensorflow as tf
import dill

# Function
def resizing(img:np.ndarray, img_size=256):
    """
    resize image into img_size by slice
    default = 256 x 256
    """
    return cv2.resize(img, dsize=(img_size, img_size), interpolation=cv2.INTER_AREA)

def img_normal_typetrans(img:np.array):
    """
    normalization & type transformation
    float 64 > uint 8 
    -> removal bg, clahe
    """
    xmax, xmin = img.max(), img.min()
    img = (img - xmin)/(xmax - xmin)
    return np.uint8(img*255)
    # return np.uint8(img)

def typetrans(img:np.array):
    """
    type transformation
    float 64 > uint 8 
    -> removal bg, clahe
    """
    return np.uint8(img)

def removal_bg(img:np.array):
    """
    remove ct bed & background
    using cv2
    def : removal_bg is inside def : transform 
    """
    abscontour = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    abscontour = cv2.cvtColor(abscontour, cv2.COLOR_BGR2Lab)
    blur_k = int((abscontour.mean()*0.5)//2)*2+1
    abscontour = cv2.medianBlur(abscontour, blur_k)
    abscontour = cv2.cvtColor(abscontour, cv2.COLOR_Lab2BGR)
    abscontour = cv2.cvtColor(abscontour, cv2.COLOR_BGR2GRAY)
    if abscontour.mean() > 100 : th = abscontour.mean()*0.99
    else : th = abscontour.mean() 
    ret, abscontour = cv2.threshold(abscontour, th, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_and(img, abscontour)

def clahe_apply(img:np.array):
    """
    default : clipLimit = 7.0
    """
    clahe = cv2.createCLAHE(clipLimit=5.0)
    return clahe.apply(img)

def clip_hu(img, img_min=-150, img_max=200):
    return np.clip(img, img_min, img_max)


def interval_mapping(image, from_min=-150, from_max=200, to_min=0, to_max=1):
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)

def just_normal(image):
    xmax, xmin = image.max(), image.min()
    return (image - xmin)/(xmax - xmin)

def preprocessing(img_:list, mask_:list, img_size=256):
    import warnings
    warnings.filterwarnings('ignore')
    """
    img_path = nifti file path
    read nifti file, resizing, normalizing, uint8 transform, remove bg, clahe
        - resizing :resize to img_size
        - normal_typetrans : normalization, type transformation by float to uint8
        - removal_bg : remove ct bed and background
        - clahe_apply : make image get contrast
    """
    all_modalities = []
    for img_path, mask_path in zip(img_,mask_):
        img = nib.load(img_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()
        img_size = 256
        
        img_ = []
        mask_ = []
        count = 0
        casename = os.path.basename(os.path.dirname(img_path))

        # for i, mask_image in tqdm(enumerate(mask), desc=f'{casename}[PREPROCESSING]', unit='[BY SLICE]'):
        for i, mask_image in enumerate(mask):
            if mask_image.max() >= 0.0 :

                #image
                i_transformed = resizing(img[i])
                # i_transformed = img_normal_typetrans(i_transformed)
                # i_transformed = removal_bg(i_transformed)
                # i_transformed = clahe_apply(i_transformed)
                i_transformed = clip_hu(i_transformed,-150,200)
                i_transformed = interval_mapping(i_transformed,-150,200,0,1)
                i_transformed = img_normal_typetrans(i_transformed)
                i_transformed = removal_bg(i_transformed)
                # i_transformed = clahe_apply(i_transformed)   
                i_transformed = just_normal(i_transformed)             
                img_.append(i_transformed)        

                # mask
                m_transformed = resizing(mask[i])
                m_transformed = typetrans(m_transformed)
                mask_.append(m_transformed)

        # print(f'slices : {len(img_)}')
        
        if len(img_) != 0 :
            img, mask =  np.stack(img_), np.stack(mask_)

            all_modalities.append(img)
            all_modalities.append(mask)
            all_modalities = np.array(all_modalities)
            # all_modalities = np.rint(all_modalities).astype(np.uint8)
            return all_modalities.transpose((1,2,3,0))
            # return all_modalities
        else:
            pass
        
def numpy_concat(input_data):
    c = 0
    for i in tqdm(input_data, desc = 'Numpy Concatenate', unit = 'Images / Masks'):
        imgs = i[:,:,:,0]
        masks = i[:,:,:,1]

        if c == 0 :
            img = imgs
            mask = masks
            c += 1
        else:
            img = np.concatenate((img,imgs), axis=0)
            mask = np.concatenate((mask,masks), axis=0)
            c += 1        
        # print(f'images : {imgs.shape} , masks : {masks.shape}')
    return img, mask


# model
def multi_unet_model(n_classes=4, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = inputs

    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
     
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
     
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
     
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
     
    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model

def get_model():
    return multi_unet_model(n_classes=n_classes,IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)

# loss



def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        y_true = tf.cast(y_true, tf.float32)
        # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        # Calculate alpha_t
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        # Calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.mean(K.sum(loss, axis=1))
        return loss

    return binary_focal_loss_fixed


def categorical_focal_loss(alpha, gamma=2.):
    """
    Softmax version of focal loss.
    When there is a skew between different categories/labels in your data set, you can try to apply this function as a
    loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1 
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
      categories/labels, the size of the array needs to be consistent with the number of classes.
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    alpha = np.array(alpha, dtype=np.float32)

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss, axis=-1))

    return categorical_focal_loss_fixed


if __name__ == '__main__':

    # Test serialization of nested functions
    bin_inner = dill.loads(dill.dumps(binary_focal_loss(gamma=2., alpha=.25)))
    print(bin_inner)

    cat_inner = dill.loads(dill.dumps(categorical_focal_loss(gamma=2., alpha=.25)))
    print(cat_inner)
    
    
    
print('Multiclass Semgentation')

path = '/root/project/kits/kits21/kits21/data'
n_classes = 4
p = sorted(os.listdir(path))
p.remove('kits.json')



# train, test = train_test_split(p, test_size=0.2, random_state=25,shuffle=True)
# print(f'Train : {len(train)}')
# print(f'Test  : {len(test)}')

# train = sorted(train)
# print(f'Train[-1] : {train[-1]}')
# test = sorted(test)
# tdf = pd.DataFrame({'train':train})
# tdf.to_csv('/root/project/code/kits/MUL_FINAL/train_MULTI.csv',index=0, encoding='utf-8-sig')

# input_data = []
# img = []
# mask = []
# # for i,tr in enumerate(train):
# for i,tr in enumerate(tqdm(train[:30], desc = 'Preprocessing', unit = 'Train_item')):
#     img_ = sorted(glob(f'{path}/{tr}/imaging.nii.gz'))
#     mask_ = sorted(glob(f'{path}/{tr}/aggregated_AND_seg.nii.gz'))
#     print(f'[{i}]  :  {img_[0]}')
#     all_modalities = preprocessing(img_,mask_)
#     if all_modalities is not None:
#         # print('Not none')
#         input_data.append(all_modalities)
#     else:
#         pass

# train_images, train_masks = numpy_concat(input_data)
                                                            
                                                            
import pickle

for i in tqdm(range(1),desc = 'Train_images'):
    with open('code/kits/Trials/multisegmentation/pkls/train_images.pkl','rb') as images:
        train_images = pickle.load(images)
for i in tqdm(range(1),desc = 'Train_masks'):   
    with open('code/kits/Trials/multisegmentation/pkls/train_masks.pkl','rb') as masks:
        train_masks = pickle.load(masks)
    
print(f'Train_images shape : {train_images.shape}')
print(f'Train_masks  shape : {train_masks.shape}')

print("Label Encoding")

labelencoder = LabelEncoder()
n, h, w = train_masks.shape
train_masks_reshaped = train_masks.reshape(-1,1)
train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

print(f'train_masks Unique : {np.unique(train_masks)}')
n_classes = 4

print('#######  Expanding...   ########' )
train_images = np.expand_dims(train_images, axis=3)

# plt.title(f'{train_images[100].max()}')
print(train_images[100].min(), train_images[100].max())
print(train_images.shape)
print(train_images.dtype)

# train_images = normalize(train_images)
print('Train_image : ',train_images.shape)
print(train_images.min(), train_images.max())
print(train_images.dtype)
print(train_images.shape)

print(train_masks_encoded_original_shape.shape)
train_masks = np.expand_dims(train_masks_encoded_original_shape, axis=3)
print(train_masks.shape)


print('#######  splitting...   ########' )
X_train, X_test, y_train, y_test = train_test_split(train_images, train_masks, test_size=0.2, random_state=32)



print('y_train shape : ',y_train.shape)
train_masks_cat = to_categorical(y_train, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))
print('train_masks_cat shape : ',train_masks_cat.shape)
print('y_train_cat shape : ',y_train_cat.shape)

test_masks_cat = to_categorical(y_test, num_classes=n_classes)
print('y_test shape : ',y_test.shape)
y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))
print('test_masks_cat shape : ',test_masks_cat.shape)
print('y_test_cat shape : ',y_test_cat.shape)

IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = X_train.shape[1],X_train.shape[2],X_train.shape[3]

print('IMG_HEIGHT : ', IMG_HEIGHT)
print('IMG_WIDTH  : ', IMG_WIDTH)
print('IMG_CHANNELS : ',IMG_CHANNELS)
print('n_classes : ',n_classes)



class_weights = list(class_weight.compute_class_weight(class_weight='balanced',
                                                 classes = np.unique(train_masks_reshaped_encoded),
                                                 y = train_masks_reshaped_encoded))

print("Class weights are...:", class_weights)

model = get_model()

model.compile(loss=[categorical_focal_loss(alpha=[class_weights], gamma=2)], metrics=["acc","mse"], optimizer='adam')

# es = EarlyStopping(monitor='val_acc', mode='auto',patience=10)
rl = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=10, verbose=1, mode='auto', min_lr=1e-05)



history = model.fit(X_train, y_train_cat, validation_data=(X_test, y_test_cat), epochs=100, batch_size=4, callbacks=[rl])


model.save('/root/project/code/kits/MUL_FINAL/full_multimodel.h5')
hist_df = pd.DataFrame(history.history) 
hist_csv_file = '/root/project/code/kits/MUL_FINAL/full_history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
