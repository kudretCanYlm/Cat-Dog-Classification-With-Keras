import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, losses, metrics
from keras import layers, models
import os
import shutil

orginal_dataset_dir = "C:\\Users\\kudret\\Desktop\\manning-keras\\PetImages"
orginal_dataset_dir_cat = orginal_dataset_dir+"\\Cat"
orginal_dataset_dir_dog = orginal_dataset_dir+"\\Dog"
base_dir = "C:\\Users\\kudret\\Desktop\\manning-keras\\PetImagesSmall"

train_dir = os.path.join(base_dir, "train")
validation_dir = os.path.join(base_dir, "validation")
test_dir = os.path.join(base_dir, "test")


if not os.path.isdir(base_dir):
    os.mkdir(base_dir)

    
    os.mkdir(train_dir)

    os.mkdir(validation_dir)

    os.mkdir(test_dir)

    train_cats_dir = os.path.join(train_dir, "cats")
    os.mkdir(train_cats_dir)

    train_dogs_dir = os.path.join(train_dir, "dogs")
    os.mkdir(train_dogs_dir)

    validation_cats_dir = os.path.join(validation_dir, "cats")
    os.mkdir(validation_cats_dir)

    validation_dogs_dir = os.path.join(validation_dir, "dogs")
    os.mkdir(validation_dogs_dir)

    test_cats_dir = os.path.join(test_dir, "cats")
    os.mkdir(test_cats_dir)

    test_dogs_dir = os.path.join(test_dir, "dogs")
    os.mkdir(test_dogs_dir)

    fnames = ["{}.jpg".format(i) for i in range(1000)]

    for fname in fnames:
        src = os.path.join(orginal_dataset_dir_cat, fname)
        dst = os.path.join(train_cats_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ['{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(orginal_dataset_dir_cat, fname)
        dst = os.path.join(validation_cats_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ["{}.jpg".format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(orginal_dataset_dir_cat, fname)
        dst = os.path.join(test_cats_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ["{}.jpg".format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(orginal_dataset_dir_dog, fname)
        dst = os.path.join(train_dogs_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ["{}.jpg".format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(orginal_dataset_dir_dog, fname)
        dst = os.path.join(validation_dogs_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ["{}.jpg".format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(orginal_dataset_dir_dog, fname)
        dst = os.path.join(test_dogs_dir, fname)
        shutil.copyfile(src, dst)

# Building your network

model = models.Sequential()
# l1
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))

# l2
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))

# l3
model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))

# l4
model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))

# flatten
model.add(layers.Flatten())
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dropout(0.6))
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dropout(0.6))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dropout(0.6))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dropout(0.6))
model.add(layers.Dense(32, activation="relu"))
model.add(layers.Dropout(0.6))
model.add(layers.Dense(16, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

# Configuring the model for training
model.compile(loss="binary_crossentropy",
              optimizer=optimizers.rmsprop_v2.RMSprop(lr=0.001),
              metrics=["accuracy"])

# Data preprocessing
# 1 Read the picture files.
# 2 Decode the JPEG content to RGB grids of pixels.
# 3 Convert these into floating-point tensors.
# 4 Rescale the pixel values (between 0 and 255) to the [0, 1] interval (as you know, neural networks prefer to deal with small input values).


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
    )

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",
    save_format="jpg"
    
)
"""Because you use
binary_crossentropy
loss, you need binary
labels."""

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",
    save_format="jpg"
)

# Fitting the model using a batch generator
#hata
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=50
)


# Saving the model
model.save("cats_and_dogs_small_1.h5")

# Displaying curves of loss and accuracy during training
print(history.history)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#PILLOW version
#8.3.1