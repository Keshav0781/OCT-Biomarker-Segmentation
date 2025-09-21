import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Add, Multiply, Activation


# Set paths for train, validation, and test data
base_path = "/home/ig53agos/hole"
train_data_path = os.path.join(base_path, "Train_original_images")
train_mask_path = os.path.join(base_path, "Train_masks_images")
val_data_path = os.path.join(base_path, "Val_original_images")
val_mask_path = os.path.join(base_path, "Val_masks_images")
test_data_path = os.path.join(base_path, "Testing_original_images")
test_mask_path = os.path.join(base_path, "Testing_masks_images")

# Results folder
results_folder = os.path.join(base_path, "results", "attention")
os.makedirs(results_folder, exist_ok=True)

# Image dimensions
img_width, img_height = 512, 512

def preprocess_mask(mask_path):
    processed_masks = []
    for filename in os.listdir(mask_path):
        if filename.endswith(".png"):
            mask_img = Image.open(os.path.join(mask_path, filename))
            mask_arr = np.array(mask_img)
            
            # Extract the red channel (macular hole regions)
            mask_red = (mask_arr[:, :, 0] == 255) & (mask_arr[:, :, 1] == 0) & (mask_arr[:, :, 2] == 0)
            mask_red = mask_red.astype(np.float32)
            
            # Improve mask quality
            mask_red = ndimage.binary_closing(mask_red, structure=np.ones((3,3)))
            mask_red = ndimage.binary_opening(mask_red, structure=np.ones((3,3)))
            mask_red = ndimage.gaussian_filter(mask_red.astype(float), sigma=0.5)
            mask_red = (mask_red > 0.5).astype(np.float32)
            
            mask_red = np.expand_dims(mask_red, axis=-1)  # Add a channel dimension
            processed_masks.append(mask_red)
    
    return np.array(processed_masks)

def load_data(data_path, mask_path):
    images = []
    masks = []
    for filename in os.listdir(data_path):
        if filename.endswith(".png"):
            img = load_img(os.path.join(data_path, filename), target_size=(img_width, img_height))
            img_arr = img_to_array(img) / 255.0
            images.append(img_arr)
    
    images = np.array(images)
    masks = preprocess_mask(mask_path)
    
    return images, masks

# Load data
train_images, train_masks = load_data(train_data_path, train_mask_path)
val_images, val_masks = load_data(val_data_path, val_mask_path)
test_images, test_masks = load_data(test_data_path, test_mask_path)

def attention_gate(x, g, inter_channel):
    theta_x = Conv2D(inter_channel, (1, 1), padding='same')(x)
    phi_g = Conv2D(inter_channel, (1, 1), padding='same')(g)
    
    f = Activation('relu')(Add()([theta_x, phi_g]))
    psi_f = Conv2D(1, (1, 1), padding='same')(f)
    
    rate = Activation('sigmoid')(psi_f)
    
    att_x = Multiply()([x, rate])
    
    return att_x

def build_attention_unet(input_shape):
    inputs = Input(input_shape)

    # Encoder (reduced number of filters)
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, 3, activation='relu', padding='same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    # Bridge
    conv5 = Conv2D(512, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, 3, activation='relu', padding='same')(conv5)
    drop5 = Dropout(0.5)(conv5)

    # Decoder with Attention Gates (reduced number of filters)
    up6 = Conv2D(256, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(drop5))
    att6 = attention_gate(drop4, up6, 128)
    merge6 = concatenate([att6, up6], axis=3)
    conv6 = Conv2D(256, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(256, 3, activation='relu', padding='same')(conv6)

    up7 = Conv2D(128, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
    att7 = attention_gate(conv3, up7, 64)
    merge7 = concatenate([att7, up7], axis=3)
    conv7 = Conv2D(128, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(128, 3, activation='relu', padding='same')(conv7)

    up8 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7))
    att8 = attention_gate(conv2, up8, 32)
    merge8 = concatenate([att8, up8], axis=3)
    conv8 = Conv2D(64, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv2D(64, 3, activation='relu', padding='same')(conv8)

    up9 = Conv2D(32, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8))
    att9 = attention_gate(conv1, up9, 16)
    merge9 = concatenate([att9, up9], axis=3)
    conv9 = Conv2D(32, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv2D(32, 3, activation='relu', padding='same')(conv9)

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

# Build and compile the U-Net model
model = build_attention_unet((img_width, img_height, 3))
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy', iou, dice_coef])

# Train the model
epochs = 50
history = model.fit(
    train_images,
    train_masks,
    batch_size=8,
    epochs=epochs,
    validation_data=(val_images, val_masks)
)

# Save the model
model.save(os.path.join(results_folder, 'macular_hole_segmentation_model.h5'))

# Plot and save training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['iou'], label='Training IoU')
plt.plot(history.history['val_iou'], label='Validation IoU')
plt.title('Model IoU')
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(results_folder, 'training_history.png'))
plt.close()

def post_process_prediction(pred_mask, threshold=0.4):
    # Ensure pred_mask is 2D
    pred_mask = np.squeeze(pred_mask)
    
    # Threshold the prediction
    binary_mask = (pred_mask > threshold).astype(np.float32)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    binary_mask = ndimage.binary_opening(binary_mask, structure=kernel)
    binary_mask = ndimage.binary_closing(binary_mask, structure=kernel)
    
    # Add channel dimension back if necessary
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
        # Display ground truth mask on black background
        gt_mask_display = np.zeros((*true_masks[idx].shape[:2], 3))
        gt_mask_display[:,:,0] = true_masks[idx].squeeze()  # Red channel
        plt.imshow(gt_mask_display)
        plt.title("Ground Truth Mask")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        # Display predicted mask on black background
        pred_mask_display = np.zeros((*pred_masks[idx].shape[:2], 3))
        pred_mask_display[:,:,0] = pred_masks[idx].squeeze()  # Red channel
        plt.imshow(pred_mask_display)
        plt.title("Predicted Mask")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_folder, f'result_{i+1}.png'))
        plt.close()

# Predict on test data
print("Starting prediction on test data...")
test_predictions = model.predict(test_images)
print("Prediction complete. Processing predictions...")

test_predictions_processed = np.array([post_process_prediction(pred) for pred in test_predictions])
print("Predictions processed successfully.")

# Calculate mean IoU and Dice coefficient for test set
print("Calculating IoU and Dice coefficient...")
test_iou = np.mean([iou(true, pred) for true, pred in zip(test_masks, test_predictions_processed)])
test_dice = np.mean([dice_coef(true, pred) for true, pred in zip(test_masks, test_predictions_processed)])

print(f"Mean IoU on test set: {test_iou:.4f}")
print(f"Mean Dice coefficient on test set: {test_dice:.4f}")

# Save metrics to a file
print(f"Saving test metrics to {os.path.join(results_folder, 'test_metrics.txt')}")
with open(os.path.join(results_folder, 'test_metrics.txt'), 'w') as f:
    f.write(f"Mean IoU on test set: {test_iou:.4f}\n")
    f.write(f"Mean Dice coefficient on test set: {test_dice:.4f}\n")
print("Test metrics saved successfully.")

# Visualize results
print(f"Saving visualization results to {results_folder}")
test_filenames = [f for f in os.listdir(test_data_path) if f.endswith('.png')]
visualize_results(test_images, test_masks, test_predictions_processed, test_filenames)
print("Visualization results saved successfully.")

print("Segmentation complete. Results saved in:", results_folder)