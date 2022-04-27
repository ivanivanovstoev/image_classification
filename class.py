import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import tensorflow_addons as tfa

# зареждаме целия набор от даннии.
all_ds   = tfds.load("eurosat", with_info=True)

# разделяме на тренировъчен, тестов и валидация 60%, 20% и 20% съответно.
train_ds = tfds.load("eurosat", split="train[:60%]")
test_ds  = tfds.load("eurosat", split="train[60%:80%]")
valid_ds = tfds.load("eurosat", split="train[80%:]")
# Имената на класовете.
class_names = all_ds[1].features["label"].names
# Брой класве(10).
num_classes = len(class_names)
num_examples = all_ds[1].splits["train"].num_examples

# Печатаме графика с етикетите и брой данни във всеки клас.
fig, ax = plt.subplots(1, 1, figsize=(14,10))
labels, counts = np.unique(np.fromiter(all_ds[0]["train"].map(lambda x: x["label"]), np.int32), 
                       return_counts=True)
#Български превод на етикетите.
class_names = ['Eдногод.','Гори','Трев.','Магист.','Промиш.','Пасища','Многог.','Жилища','Реки','Ез.Море']

plt.ylabel('Бройки')
plt.xlabel('Етикети')
sns.barplot(x = [class_names[l] for l in labels], y = counts, ax=ax) 
for i, x_ in enumerate(labels):
  ax.text(x_-0.2, counts[i]+5, counts[i])
# Поставяме заглавие.
ax.set_title("Графика проказваща бройките от изображения във всеки клас")
# Запазваме картинката.
plt.savefig("class_samples.png")

def prepare_for_training(ds, batch_size=64, shuffle_buffer_size=1000):
  if isinstance(True, str):
    ds = ds.cache(True)
  else:
    ds = ds.cache()
  ds = ds.map(lambda d: (d["image"], tf.one_hot(d["label"], num_classes)))
  # Разбъркваме.
  ds = ds.shuffle(buffer_size=shuffle_buffer_size)
  ds = ds.repeat()
  # Разделяме на бачове.
  ds = ds.batch(batch_size)
  # Бачовете се свалят на заденфон докато модела се обучава.
  ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  return ds

batch_size = 64

# Предварителна обработка.
train_ds = prepare_for_training(train_ds, batch_size=batch_size)
valid_ds = prepare_for_training(valid_ds, batch_size=batch_size)

# Размерности.
for el in valid_ds.take(1):
  print(el[0].shape, el[1].shape)
for el in train_ds.take(1):
  print(el[0].shape, el[1].shape)

#Първия бач.
batch = next(iter(train_ds))

def show_batch(batch):
  plt.figure(figsize=(16, 16))
  for n in range(min(32, batch_size)):
      ax = plt.subplot(batch_size//8, 8, n + 1)
      # Показваме.
      plt.imshow(batch[0][n])
      # Поаставяме заглавие и етикет.
      plt.title(class_names[tf.argmax(batch[1][n].numpy())])
      plt.axis('off')
      plt.savefig("sample-images.png")

# Показваме един бач и етикетите му.
show_batch(batch)

model_url = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_l/feature_vector/2"

# Сваляме и зареждаме слоя със обучените характеристики.
keras_layer = hub.KerasLayer(model_url, output_shape=[1280], trainable=True)

m = tf.keras.Sequential([
  keras_layer,
  tf.keras.layers.Dense(num_classes, activation="softmax")
])
# Изграждаме модела. Входа е катринка (64, 64, 3).
m.build([None, 64, 64, 3])
m.compile(
    loss="categorical_crossentropy", 
    optimizer="adam", 
    metrics=["accuracy", tfa.metrics.F1Score(num_classes)]
)

m.summary()

model_name = "satellite-classification"
model_path = os.path.join("results", model_name + ".h5")
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True, verbose=1)

n_training_steps   = int(num_examples * 0.6) // batch_size
n_validation_steps = int(num_examples * 0.2) // batch_size

''' history = m.fit(
    train_ds, validation_data=valid_ds,
    steps_per_epoch=n_training_steps,
    validation_steps=n_validation_steps,
    verbose=1, epochs=5, 
    callbacks=[model_checkpoint]
)
 '''
# Брой тестови стъпки. 
n_testing_steps = int(all_ds[1].splits["train"].num_examples * 0.2)

m.load_weights(model_path)

# Зареждаме тестовите картинки като NumPy array.
images = np.array([ d["image"] for d in test_ds.take(n_testing_steps) ])
print("Картинки :", images.shape)

# Зареждаме тестовите етикети като  NumPy array.
labels = np.array([ d["label"] for d in test_ds.take(n_testing_steps) ])
print("Етикети:", labels.shape)

# Извикваме модела върху тестовите картинки .
predictions = m.predict(images)
# Определяме всеки клас.
predictions = np.argmax(predictions, axis=1)
print("Предсказани:", predictions.shape)

from sklearn.metrics import f1_score

accuracy = tf.keras.metrics.Accuracy()
accuracy.update_state(labels, predictions)
print("Точност:", accuracy.result().numpy())
print("F1 Score:", f1_score(labels, predictions, average="macro"))

# Изчисляваме матрица на грешките.
cmn = tf.math.confusion_matrix(labels, predictions).numpy()
# Сменяме в %.
cmn = cmn.astype('float') / cmn.sum(axis=0)[:, np.newaxis]
# Отпечатваме.
fig, ax = plt.subplots(figsize=(8,4))
sns.heatmap(cmn, annot=True, fmt='.2f', 
            xticklabels=[f"пр_{c}" for c in class_names], 
            yticklabels=[f"ис_{c}" for c in class_names],
            cmap="Blues",
            cbar_kws={'label': 'Реална'}
            #cmap="rocket_r"
            )
plt.ylabel('Реална')
plt.xlabel('Предсказана')
# Записваме във файл .
plt.subplots_adjust(bottom=0.3)
plt.savefig("confusion-matrix.png")

plt.show()

def show_predicted_samples():
  plt.figure(figsize=(14, 14))
  for n in range(64):
      ax = plt.subplot(8, 8, n + 1)
      # Показваме карттинка.
      plt.imshow(images[n])
      # Отпечатваме етикет.
      if predictions[n] == labels[n]:
        # Вярно предсказания.
        ax.set_title(class_names[predictions[n]], color="green")
      else:
        # Грешно пресказание.
        ax.set_title(f"{class_names[predictions[n]]}/T:{class_names[labels[n]]}", color="red")
      plt.axis('off')
      plt.savefig("predicted-sample-images.png")

# Показваме батч от предсказанията на модела.
show_predicted_samples()