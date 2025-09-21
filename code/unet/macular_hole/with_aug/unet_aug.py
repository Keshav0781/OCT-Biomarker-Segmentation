import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import gaussian_filter, map_coordinates

# Set paths
base_path = "/home/ig53agos/hole"
train_data_path = os.path.join(base_path, "Train_original_images")
train_mask_path = os.path.join(base_path, "Train_masks_images")
val_data_path = os.path.join(base_path, "Val_original_images")
val_mask_path = os.path.join(base_path, "Val_masks_images") 
test_data_path = os.path.join(base_path, "Testing_original_images")
test_mask_path = os.path.join(base_path, "Testing_masks_images")

results_folder = os.path.join(base_path, "results", "unet_aug")
os.makedirs(results_folder, exist_ok=True)

img_width, img_height = 512, 512

def preprocess_mask(mask_path):
   processed_masks = []
   for filename in os.listdir(mask_path):
       if filename.endswith(".png"):
           mask_img = Image.open(os.path.join(mask_path, filename))
           mask_arr = np.array(mask_img)
           
           mask_red = (mask_arr[:, :, 0] == 255) & (mask_arr[:, :, 1] == 0) & (mask_arr[:, :, 2] == 0)
           mask_red = mask_red.astype(np.float32)
           
           mask_red = ndimage.binary_closing(mask_red, structure=np.ones((3,3)))
           mask_red = ndimage.binary_opening(mask_red, structure=np.ones((3,3)))
           mask_red = ndimage.gaussian_filter(mask_red.astype(float), sigma=0.5)
           mask_red = (mask_red > 0.5).astype(np.float32)
           
           mask_red = np.expand_dims(mask_red, axis=-1)
           processed_masks.append(mask_red)
   
   return np.array(processed_masks)

def load_data(data_path, mask_path):
   images = []
   for filename in os.listdir(data_path):
       if filename.endswith(".png"):
           img = load_img(os.path.join(data_path, filename), target_size=(img_width, img_height))
           img_arr = img_to_array(img) / 255.0
           images.append(img_arr)
   
   images = np.array(images)
   masks = preprocess_mask(mask_path)
   
   return images, masks

def elastic_transform(image, mask, alpha=500, sigma=20, random_state=None):
   if random_state is None:
       random_state = np.random.RandomState(None)

   shape = image.shape[:2]
   dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
   dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

   x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
   indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

   distorted_image = np.zeros_like(image)
   for c in range(image.shape[2]):
       distorted_image[:,:,c] = map_coordinates(image[:,:,c], indices, order=1, mode='reflect').reshape(shape)
   
   distorted_mask = map_coordinates(mask.squeeze(), indices, order=1, mode='reflect').reshape(shape)
   distorted_mask = np.expand_dims(distorted_mask, axis=-1)

   return distorted_image, distorted_mask

def apply_medical_augmentation(image, mask):
   image, mask = elastic_transform(image, mask)
   gamma = np.random.uniform(0.8, 1.2)
   image = np.power(image, gamma)
   noise = np.random.normal(0, 0.05, image.shape)
   image = np.clip(image + image * noise, 0, 1)
   return image, mask

def load_and_augment_data(data_path, mask_path, augmentation_factor=2):
   original_images, original_masks = load_data(data_path, mask_path)
   
   augmented_images = [original_images]
   augmented_masks = [original_masks]
   
   for _ in range(augmentation_factor - 1):
       aug_images = []
       aug_masks = []
       for img, mask in zip(original_images, original_masks):
           aug_img, aug_mask = apply_medical_augmentation(img, mask)
           aug_images.append(aug_img)
           aug_masks.append(aug_mask)
       augmented_images.append(np.array(aug_images))
       augmented_masks.append(np.array(aug_masks))
   
   augmented_images = np.concatenate(augmented_images, axis=0)
   augmented_masks = np.concatenate(augmented_masks, axis=0)
   
   return augmented_images, augmented_masks

# Load and augment data
train_images, train_masks = load_and_augment_data(train_data_path, train_mask_path)
val_images, val_masks = load_data(val_data_path, val_mask_path)
test_images, test_masks = load_data(test_data_path, test_mask_path)

def build_unet(input_shape):
   inputs = Input(input_shape)

   # Encoder
   conv1 = Conv2D(32, 3, padding='same')(inputs)
   conv1 = BatchNormalization()(conv1)
   conv1 = Activation('relu')(conv1)
   conv1 = Conv2D(32, 3, padding='same')(conv1)
   conv1 = BatchNormalization()(conv1) 
   conv1 = Activation('relu')(conv1)
   pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

   conv2 = Conv2D(48, 3, padding='same')(pool1)
   conv2 = BatchNormalization()(conv2)
   conv2 = Activation('relu')(conv2)
   conv2 = Conv2D(48, 3, padding='same')(conv2)
   conv2 = BatchNormalization()(conv2)
   conv2 = Activation('relu')(conv2)
   pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

   conv3 = Conv2D(96, 3, padding='same')(pool2)
   conv3 = BatchNormalization()(conv3)
   conv3 = Activation('relu')(conv3)
   conv3 = Conv2D(96, 3, padding='same')(conv3)
   conv3 = BatchNormalization()(conv3)
   conv3 = Activation('relu')(conv3)
   pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

   conv4 = Conv2D(192, 3, padding='same')(pool3)
   conv4 = BatchNormalization()(conv4)
   conv4 = Activation('relu')(conv4)
   conv4 = Conv2D(192, 3, padding='same')(conv4)
   conv4 = BatchNormalization()(conv4)
   conv4 = Activation('relu')(conv4)
   drop4 = Dropout(0.5)(conv4)
   pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

   # Bridge
   conv5 = Conv2D(384, 3, padding='same')(pool4)
   conv5 = BatchNormalization()(conv5)
   conv5 = Activation('relu')(conv5)
   conv5 = Conv2D(384, 3, padding='same')(conv5)
   conv5 = BatchNormalization()(conv5)
   conv5 = Activation('relu')(conv5)
   drop5 = Dropout(0.5)(conv5)

   # Decoder
   up6 = Conv2D(192, 2, padding='same')(UpSampling2D(size=(2, 2))(drop5))
   up6 = BatchNormalization()(up6)
   up6 = Activation('relu')(up6)
   merge6 = concatenate([drop4, up6], axis=3)
   conv6 = Conv2D(192, 3, padding='same')(merge6)
   conv6 = BatchNormalization()(conv6)
   conv6 = Activation('relu')(conv6)
   conv6 = Conv2D(192, 3, padding='same')(conv6)
   conv6 = BatchNormalization()(conv6)
   conv6 = Activation('relu')(conv6)

   up7 = Conv2D(96, 2, padding='same')(UpSampling2D(size=(2, 2))(conv6))
   up7 = BatchNormalization()(up7)
   up7 = Activation('relu')(up7)
   merge7 = concatenate([conv3, up7], axis=3)
   conv7 = Conv2D(96, 3, padding='same')(merge7)
   conv7 = BatchNormalization()(conv7)
   conv7 = Activation('relu')(conv7)
   conv7 = Conv2D(96, 3, padding='same')(conv7)
   conv7 = BatchNormalization()(conv7)
   conv7 = Activation('relu')(conv7)

   up8 = Conv2D(48, 2, padding='same')(UpSampling2D(size=(2, 2))(conv7))
   up8 = BatchNormalization()(up8)
   up8 = Activation('relu')(up8)
   merge8 = concatenate([conv2, up8], axis=3)
   conv8 = Conv2D(48, 3, padding='same')(merge8)
   conv8 = BatchNormalization()(conv8)
   conv8 = Activation('relu')(conv8)
   conv8 = Conv2D(48, 3, padding='same')(conv8)
   conv8 = BatchNormalization()(conv8)
   conv8 = Activation('relu')(conv8)

   up9 = Conv2D(32, 2, padding='same')(UpSampling2D(size=(2, 2))(conv8))
   up9 = BatchNormalization()(up9)
   up9 = Activation('relu')(up9)
   merge9 = concatenate([conv1, up9], axis=3)
   conv9 = Conv2D(32, 3, padding='same')(merge9)
   conv9 = BatchNormalization()(conv9)
   conv9 = Activation('relu')(conv9)
   conv9 = Conv2D(32, 3, padding='same')(conv9)
   conv9 = BatchNormalization()(conv9)
   conv9 = Activation('relu')(conv9)

   outputs = Conv2D(1, 1, activation='sigmoid')(conv9)

   model = Model(inputs=inputs, outputs=outputs)
   return model

def iou(y_true, y_pred):
   y_true = tf.cast(tf.convert_to_tensor(y_true), tf.float32)
   y_pred = tf.cast(tf.convert_to_tensor(y_pred), tf.float32)
   y_pred = tf.cast(tf.greater(y_pred, 0.5), tf.float32)
   intersection = tf.reduce_sum(y_true * y_pred)
   union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
   return intersection / (union + tf.keras.backend.epsilon())

def dice_coef(y_true, y_pred):
   y_true = tf.cast(tf.convert_to_tensor(y_true), tf.float32)
   y_pred = tf.cast(tf.convert_to_tensor(y_pred), tf.float32)
   y_pred = tf.cast(tf.greater(y_pred, 0.5), tf.float32)
   intersection = tf.reduce_sum(y_true * y_pred)
   return (2. * intersection + 1.) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1.)

model = build_unet((img_width, img_height, 3))
model.compile(optimizer=Adam(learning_rate=1e-4), 
            loss='binary_crossentropy',
            metrics=['accuracy', iou, dice_coef])

epochs = 50
history = model.fit(
   train_images,
   train_masks,
   batch_size=8,
   epochs=epochs,
   validation_data=(val_images, val_masks)
)

model.save(os.path.join(results_folder, 'macular_hole_segmentation_model.h5'))

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history.history['iou'], label='Training')
plt.plot(history.history['val_iou'], label='Validation')
plt.title('Model IoU')
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(history.history['dice_coef'], label='Training')
plt.plot(history.history['val_dice_coef'], label='Validation')
plt.title('Model Dice Coefficient')
plt.xlabel('Epoch')
plt.ylabel('Dice')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(results_folder, 'training_history.png'))
plt.close()

def post_process_prediction(pred_mask, threshold=0.4):
   pred_mask = np.squeeze(pred_mask)
   binary_mask = (pred_mask > threshold).astype(np.float32)
   kernel = np.ones((3, 3), np.uint8)
   binary_mask = ndimage.binary_opening(binary_mask, structure=kernel)
   binary_mask = ndimage.binary_closing(binary_mask, structure=kernel)
   if binary_mask.ndim == 2:
       binary_mask = np.expand_dims(binary_mask, axis=-1)
   return binary_mask

def visualize_results(images, true_masks, pred_masks, filenames, num_samples=71):
   sample_indices = np.random.choice(len(images), size=num_samples, replace=False)
   
   for i, idx in enumerate(sample_indices):
       plt.figure(figsize=(15, 5))
       
       plt.subplot(1, 3, 1)
       plt.imshow(images[idx])
       plt.title(f"Original Image: {filenames[idx]}")
       plt.axis('off')
       
       plt.subplot(1, 3, 2)
       gt_mask_display = np.zeros((*true_masks[idx].shape[:2], 3))
       gt_mask_display[:,:,0] = true_masks[idx].squeeze()
       plt.imshow(gt_mask_display)
       plt.title("Ground Truth Mask")
       plt.axis('off')
       
       plt.subplot(1, 3, 3)
       pred_mask_display = np.zeros((*pred_masks[idx].shape[:2], 3))
       pred_mask_display[:,:,0] = pred_masks[idx].squeeze()
       plt.imshow(pred_mask_display)
       plt.title("Predicted Mask")
       plt.axis('off')
       
       plt.tight_layout()
       plt.savefig(os.path.join(results_folder, f'result_{i+1}.png'))
       plt.close()

print("Starting prediction on test data...")
test_predictions = model.predict(test_images)
print("Prediction complete. Processing predictions...")

test_predictions_processed = np.array([post_process_prediction(pred) for pred in test_predictions])
print("Predictions processed successfully.")

test_iou = np.mean([iou(true, pred) for true, pred in zip(test_masks, test_predictions_processed)])
test_dice = np.mean([dice_coef(true, pred) for true, pred in zip(test_masks, test_predictions_processed)])

print(f"Mean IoU on test set: {test_iou:.4f}")
print(f"Mean Dice coefficient on test set: {test_dice:.4f}")

with open(os.path.join(results_folder, 'test_metrics.txt'), 'w') as f:
   f.write(f"Mean IoU on test set: {test_iou:.4f}\n")
   f.write(f"Mean Dice coefficient on test set: {test_dice:.4f}\n")

print("Generating visualizations...")
test_filenames = [f for f in os.listdir(test_data_path) if f.endswith('.png')]
visualize_results(test_images, test_masks, test_predictions_processed, test_filenames)

print("Segmentation complete. Results saved in:", results_folder)

