import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, Add
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import albumentations as A
from scipy.ndimage import gaussian_filter, map_coordinates

def create_model_directory(model_name):
   base_dir = os.path.join("results", model_name)
   os.makedirs(base_dir, exist_ok=True)
   os.makedirs(os.path.join(base_dir, "visualizations"), exist_ok=True)
   return base_dir

global IMG_H, IMG_W, NUM_CLASSES, CLASSES, COLORMAP

def load_dataset(image_path, masked_path, split=0.2):
   image_path = Path(image_path)
   masked_path = Path(masked_path)
   if not image_path.is_dir() or not masked_path.is_dir():
       print("Error: One or both provided paths are not directories.")
       return None, None, None
   image_files = sorted([f.name for f in image_path.iterdir() if f.is_file() and f.suffix.lower() in ('.jpg', '.png')])
   mask_files = sorted([f.name for f in masked_path.iterdir() if f.is_file() and f.suffix.lower() in ('.jpg', '.png')])
   assert len(image_files) == len(mask_files), "Number of images and masks do not match."
   image_paths = [str(image_path / f) for f in image_files]
   mask_paths = [str(masked_path / f) for f in mask_files]
   split_size = int(split * len(image_paths))
   train_x, valid_x = train_test_split(image_paths, test_size=split_size, random_state=42)
   train_y, valid_y = train_test_split(mask_paths, test_size=split_size, random_state=42)
   train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=42)
   train_y, test_y = train_test_split(train_y, test_size=split_size, random_state=42)
   return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def get_colormap():
   classes = ["Background", "Drusen", "Scar", "Liquid"]
   colormap = np.array([
       [0, 0, 0],        # Black (Background)
       [255, 0, 0],      # Red (Drusen)
       [0, 255, 0],      # Green (Scar)
       [0, 0, 255]       # Blue (Liquid)
   ], dtype=np.uint8)
   return classes, colormap

def enhance_green_mask_boundary(image, mask, edge_strength=0.12, contrast_strength=0.05):
   green_mask = mask[:,:,2].astype(np.float32)
   edges = cv2.Laplacian(green_mask, cv2.CV_32F)
   edges = (edges - edges.min()) / (edges.max() - edges.min())
   enhancement = 1 + (edges * edge_strength)
   enhanced_image = image.copy()
   for i in range(3):
       channel = enhanced_image[:,:,i]
       channel = np.where(green_mask > 0, channel * enhancement, channel)
       mean = np.mean(channel[green_mask > 0])
       channel[green_mask > 0] = (channel[green_mask > 0] - mean) * (1 + contrast_strength) + mean
       enhanced_image[:,:,i] = channel
   return np.clip(enhanced_image, 0, 1)

def custom_augment(image, mask):
   return enhance_green_mask_boundary(image, mask, edge_strength=0.12, contrast_strength=0.05), mask

def read_image_mask(x, y):
   x = cv2.imread(x, cv2.IMREAD_COLOR)
   y = cv2.imread(y, cv2.IMREAD_COLOR)
   assert x.shape == y.shape
   x = x / 255.0
   x = x.astype(np.float32)
   output = []
   for color in COLORMAP:
       cmap = np.all(np.equal(y, color), axis=-1)
       output.append(cmap)
   output = np.stack(output, axis=-1)
   output = output.astype(np.uint8)
   x, output = custom_augment(x, output)
   return x, output
   
def preprocess(x, y):
   def f(x, y):
       x = x.decode()
       y = y.decode()
       image, mask = read_image_mask(x, y)
       return image, mask
   image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.uint8])
   image.set_shape([None, None, 3])
   mask.set_shape([None, None, NUM_CLASSES])
   return image, mask

def conv_block(input, num_filters):
   x = Conv2D(num_filters, 3, padding="same")(input)
   x = BatchNormalization()(x)
   x = Activation("relu")(x)
   x = Conv2D(num_filters, 3, padding="same")(x)
   x = BatchNormalization()(x)
   x = Activation("relu")(x)
   return x

def residual_block(input, num_filters):
   x = conv_block(input, num_filters)
   r = Conv2D(num_filters, 1, padding="same")(input)
   r = BatchNormalization()(r)
   x = Add()([x, r])
   return x

def encoder_block(input, num_filters):
   x = residual_block(input, num_filters)
   p = MaxPool2D((2, 2))(x)
   return x, p

def decoder_block(input, skip_features, num_filters):
   x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
   x = Concatenate()([x, skip_features])
   x = residual_block(x, num_filters)
   return x

def build_deepres_unet(input_shape, num_classes):
   inputs = Input(input_shape)

   s1, p1 = encoder_block(inputs, 32)
   s2, p2 = encoder_block(p1, 48)
   s3, p3 = encoder_block(p2, 96)
   s4, p4 = encoder_block(p3, 192)

   b1 = residual_block(p4, 384)

   d1 = decoder_block(b1, s4, 192)
   d2 = decoder_block(d1, s3, 96)
   d3 = decoder_block(d2, s2, 48)
   d4 = decoder_block(d3, s1, 32)

   outputs = Conv2D(num_classes, 1, padding="same", activation="softmax")(d4)

   model = Model(inputs, outputs, name="Deep-ResU-Net")
   return model
def decode_segmentation_mask(mask, colormap):
   h, w, num_classes = mask.shape
   decoded_img = np.zeros((h, w, 3), dtype=np.uint8)
   for i in range(num_classes):
       decoded_img[mask[:, :, i] == 1] = colormap[i]
   return decoded_img

def display_results(image, true_mask, pred_mask, filename, save_dir):
   plt.figure(figsize=(15, 5))

   plt.subplot(1, 3, 1)
   plt.title(f"Original Image\n{filename}")
   plt.imshow(image)
   plt.axis('off')

   plt.subplot(1, 3, 2)
   plt.title("True Mask")
   plt.imshow(true_mask)
   plt.axis('off')

   plt.subplot(1, 3, 3)
   plt.title("Predicted Mask")
   plt.imshow(pred_mask)
   plt.axis('off')

   save_path = os.path.join(save_dir, "visualizations", filename)
   plt.savefig(save_path)
   plt.close()

def process_true_mask(mask_path, colormap):
   y = cv2.imread(mask_path, cv2.IMREAD_COLOR)
   y = cv2.resize(y, (IMG_W, IMG_H))
   output = []
   for color in colormap:
       cmap = np.all(np.equal(y, color), axis=-1)
       output.append(cmap)
   output = np.stack(output, axis=-1)
   output = output.astype(np.uint8)
   decoded_mask = decode_segmentation_mask(output, colormap)
   return decoded_mask

def predict_and_display(test_x, test_y, model, colormap, save_dir):
   for i, (img_path, mask_path) in enumerate(zip(test_x, test_y)):
       image = cv2.imread(img_path, cv2.IMREAD_COLOR)
       image = cv2.resize(image, (IMG_W, IMG_H))
       image = image / 255.0
       image = image.astype(np.float32)
       
       true_mask = process_true_mask(mask_path, colormap)
       
       input_img = np.expand_dims(image, axis=0)
       
       pred_mask = model.predict(input_img)[0]
       pred_mask = (pred_mask > 0.5).astype(np.uint8)
       
       pred_mask_decoded = decode_segmentation_mask(pred_mask, colormap)
       
       filename = f"result_{i}.png"
       display_results(image, true_mask, pred_mask_decoded, filename, save_dir)
def calculate_iou(y_true, y_pred, num_classes):
   y_true = y_true.flatten()
   y_pred = y_pred.flatten()
   iou = []
   for cls in range(num_classes):
       true_class = y_true == cls
       pred_class = y_pred == cls
       intersection = np.logical_and(true_class, pred_class).sum()
       union = np.logical_or(true_class, pred_class).sum()
       if union == 0:
           iou.append(np.nan)
       else:
           iou.append(intersection / union)
   return np.nanmean(iou)

def calculate_accuracy(y_true, y_pred):
   correct = np.equal(y_true, y_pred)
   accuracy = np.sum(correct) / correct.size
   return accuracy

def calculate_dice(y_true, y_pred, num_classes):
   dice_scores = []
   for cls in range(num_classes):
       true_class = y_true == cls
       pred_class = y_pred == cls
       intersection = np.logical_and(true_class, pred_class).sum()
       union = true_class.sum() + pred_class.sum()
       if union == 0:
           dice_scores.append(np.nan)
       else:
           dice_scores.append(2 * intersection / union)
   return np.nanmean(dice_scores)

def process_mask(mask_path, colormap):
   mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
   mask = cv2.resize(mask, (IMG_W, IMG_H))
   processed_mask = []
   for color in colormap:
       cmap = np.all(np.equal(mask, color), axis=-1)
       processed_mask.append(cmap)
   processed_mask = np.stack(processed_mask, axis=-1)
   processed_mask = processed_mask.astype(np.uint8)
   return processed_mask

def evaluate_model(model, image_paths, mask_paths, colormap, num_classes):
   iou_scores = []
   accuracies = []
   dice_scores = []
   for img_path, mask_path in zip(image_paths, mask_paths):
       image = cv2.imread(img_path, cv2.IMREAD_COLOR)
       image = cv2.resize(image, (IMG_W, IMG_H))
       image = image / 255.0
       image = image.astype(np.float32)
       true_mask = process_mask(mask_path, colormap)
       input_img = np.expand_dims(image, axis=0)
       pred_mask = model.predict(input_img, verbose=0)[0]
       pred_mask = np.argmax(pred_mask, axis=-1)
       true_mask = np.argmax(true_mask, axis=-1)
       iou = calculate_iou(true_mask, pred_mask, num_classes)
       accuracy = calculate_accuracy(true_mask, pred_mask)
       dice = calculate_dice(true_mask, pred_mask, num_classes)
       iou_scores.append(iou)
       accuracies.append(accuracy)
       dice_scores.append(dice)
   mean_iou = np.nanmean(iou_scores)
   mean_accuracy = np.mean(accuracies)
   mean_dice = np.nanmean(dice_scores)
   return mean_iou, mean_accuracy, mean_dice

if __name__ == "__main__":
   np.random.seed(42)
   tf.random.set_seed(42)

   IMG_H = 512
   IMG_W = 512
   NUM_CLASSES = 4
   input_shape = (IMG_H, IMG_W, 3)

   batch_size = 8
   lr = 1e-4
   num_epochs = 50
   
   image_path = "/home/ig53agos/lme_cluster_work/884_org_rgb_512_multi"
   masked_path = "/home/ig53agos/lme_cluster_work/884_mask_rgb_512_multi"

   (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(image_path, masked_path)
   print(f"Train: {len(train_x)}/{len(train_y)} - Valid: {len(valid_x)}/{len(valid_y)} - Test: {len(test_x)}/{len(test_y)}")
   print("")

   CLASSES, COLORMAP = get_colormap()

   model = build_deepres_unet(input_shape, NUM_CLASSES)

   def weighted_loss(y_true, y_pred):
       class_weights = tf.constant([0.5, 0.9, 1.4, 1.9], dtype=tf.float32)
       y_true = tf.cast(y_true, tf.float32)
       weights = tf.reduce_sum(class_weights * y_true, axis=-1)
       loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
       weighted_loss = loss * weights
       return tf.reduce_mean(weighted_loss)

   model.compile(
       loss=weighted_loss,
       optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
       metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=NUM_CLASSES)]
   )

   model_name = "Deep-ResU-Net"
   save_dir = create_model_directory(model_name)

   train_images = [read_image_mask(x, y)[0] for x, y in zip(train_x, train_y)]
   train_masks = [read_image_mask(x, y)[1] for x, y in zip(train_x, train_y)]
   valid_images = [read_image_mask(x, y)[0] for x, y in zip(valid_x, valid_y)]
   valid_masks = [read_image_mask(x, y)[1] for x, y in zip(valid_x, valid_y)]

   train_images = np.array(train_images)
   train_masks = np.array(train_masks)
   valid_images = np.array(valid_images)
   valid_masks = np.array(valid_masks)

   print(f"Training on {len(train_images)} images")
   print(f"Validating on {len(valid_images)} images")

   history = model.fit(
       train_images, train_masks,
       validation_data=(valid_images, valid_masks),
       epochs=num_epochs,
       batch_size=batch_size,
       verbose=1
   )

   plt.figure(figsize=(12, 4))
   
   plt.subplot(1, 3, 1)
   plt.plot(history.history['loss'], label='Training')
   plt.plot(history.history['val_loss'], label='Validation')
   plt.title('Model Loss')
   plt.xlabel('Epoch')
   plt.ylabel('Loss')
   plt.legend()
   
   plt.subplot(1, 3, 2)
   plt.plot(history.history['accuracy'], label='Training')
   plt.plot(history.history['val_accuracy'], label='Validation')
   plt.title('Model Accuracy')
   plt.xlabel('Epoch')
   plt.ylabel('Accuracy')
   plt.legend()
   
   plt.subplot(1, 3, 3)
   plt.plot(history.history['mean_io_u'], label='Training')
   plt.plot(history.history['val_mean_io_u'], label='Validation')
   plt.title('Model IoU')
   plt.xlabel('Epoch')
   plt.ylabel('IoU')
   plt.legend()
   
   plt.tight_layout()
   plt.savefig(os.path.join(save_dir, 'training_history.png'))
   plt.close()

   train_iou, train_accuracy, train_dice = evaluate_model(model, train_x, train_y, COLORMAP, NUM_CLASSES)
   valid_iou, valid_accuracy, valid_dice = evaluate_model(model, valid_x, valid_y, COLORMAP, NUM_CLASSES)
   test_iou, test_accuracy, test_dice = evaluate_model(model, test_x, test_y, COLORMAP, NUM_CLASSES)

   with open(os.path.join(save_dir, 'evaluation_results.txt'), 'w') as f:
       f.write(f"Training IoU: {train_iou:.4f}, Training Accuracy: {train_accuracy:.4f}, Training Dice: {train_dice:.4f}\n")
       f.write(f"Validation IoU: {valid_iou:.4f}, Validation Accuracy: {valid_accuracy:.4f}, Validation Dice: {valid_dice:.4f}\n")
       f.write(f"Test IoU: {test_iou:.4f}, Test Accuracy: {test_accuracy:.4f}, Test Dice: {test_dice:.4f}\n")

   predict_and_display(test_x, test_y, model, COLORMAP, save_dir)

   print(f"Training and evaluation completed. Results saved in the '{save_dir}' directory.")
