from utils.create_data_path import create_data_path
from models.cyclegan import get_cyclegan_model
from data.dataset import Dataset
import keras

dm_dir = r"./CDD-CESM/Low energy images of CDD-CESM" # Replace with path of digital mammo directory
cesm_dir = r"./CDD-CESM/Subtracted images of CDD-CESM" # Replace with path of contrast mammo directory

dm_paths, cesm_paths = create_data_path(dm_dir, cesm_dir, "jpg")

print(dm_paths[123])
print(cesm_paths[123])

DS = Dataset()
train_ds = DS.create_dataset(dm_paths, cesm_paths)

cyclegan_model = get_cyclegan_model()

cyclegan_model.built = True

cyclegan_model.load_weights("./weights/cyclegan_pretrained.weights.h5")

print("weights loaded")

checkpoint_filepath = "cyclegan_continued.weights.h5"
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath, save_weights_only=True
)

cyclegan_model.fit(
    train_ds,
    epochs=1,
    callbacks=[model_checkpoint_callback],
)