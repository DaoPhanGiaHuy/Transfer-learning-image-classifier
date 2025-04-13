# 📦 Transfer-learning-image-classifier

This project uses ResNet50 with transfer learning to classify CIFAR-10 images, achieving high accuracy through techniques like fine-tuning and data augmentation.

## 🧠 Introduction

This project presents two main approaches to classify images in the CIFAR-10 dataset:

- ✅ Traditional Convolutional Neural Network (CNN)
- ✅ Transfer Learning with ResNet50

## 🎯 Objectives

- Accurately classify images into 10 categories: airplane, car, bird, cat, deer, dog, frog, horse, ship, truck.
- Compare the performance between a custom-built CNN and a pretrained ResNet50 model.

## 🧰 Technologies Used

- TensorFlow & Keras  
- `tf.data.Dataset` for data processing and augmentation  
- Matplotlib for visualization  

---

## 📂 Code Structure

### 1. Load and Prepare Data

- Use CIFAR-10 from `keras.datasets`.
- Convert labels to one-hot encoding.
- Resize images from 32x32 to 224x224 to match ResNet50 input.

### 2. Preprocessing and Data Augmentation

- Normalize images (scale to [0, 1])
- Resize images to 224x224
- Apply Data Augmentation:
  - Random horizontal flip
  - Random brightness adjustment

### 3. Basic CNN Model

- 2 Conv2D + MaxPooling2D layers
- Flatten + Dense 128 + Dropout
- 10-class Softmax output

### 4. Train CNN

- Use `Adam` optimizer
- EarlyStopping with `patience=5`
- Epochs: 10
- Evaluate model on test set

### 5. ResNet50 Model (Transfer Learning)

- Load ResNet50 with `weights='imagenet'`
- Exclude top fully-connected layers (`include_top=False`)
- Freeze all ResNet50 layers
- Add:
  - GlobalAveragePooling2D
  - Dense 128 (ReLU)
  - Dense 10 (Softmax)

### 6. Train ResNet50

- Epochs: 25
- Callback: EarlyStopping
- Use augmented data
- Evaluate accuracy on test set

---

## 📈 Results

| Model              | Accuracy               |
|--------------------|------------------------|
| CNN                | ~56%                   |
| ResNet50 (Frozen)  | ~40%                   |

> 📝 Note: ResNet50 has not been fine-tuned yet, which limits its accuracy. Unfreezing some layers could improve performance.

---

## 📊 Training Charts

- Accuracy and Loss charts for both CNN and ResNet50 are visualized using Matplotlib.
- Helps track convergence and assess overfitting.

---

## ✅ Conclusion

- CNNs are suitable for small, easy-to-train models.
- Transfer Learning shines when using small datasets or leveraging powerful pretrained models.
- ResNet50 requires fine-tuning for better performance on CIFAR-10.

---

## 🚀 Future Development

- Fine-tune parts of the ResNet50 model (e.g., unfreeze final layers).
- Use Learning Rate Scheduling techniques.
- Apply advanced Regularization techniques like DropBlock, CutMix.
- Deploy on a web interface or Android app.

---

## 👨‍💻 Author

**Dao Phan Gia Huy**  
📧 Email: giahuy070903@gmail.com  
📁 Repo: [Transfer-learning-image-classifier](https://github.com/DaoPhanGiaHuy/Transfer-learning-image-classifier)

---

## 📌 Additional Info

- CIFAR-10 is a dataset of 60,000 color images (32x32) across 10 classes.
- Transfer Learning reduces training time and improves performance on small datasets.
- ResNet50 is a powerful architecture pretrained on over 1 million ImageNet images.

---
