from utils.create_data_path import create_data_path , create_data_path_old 
from utils.generate_image import generate_images 
from utils.metrics import Calculate_metrics_sewar
from models.cyclegan import get_cyclegan_model
from data.dataset import Dataset
import keras

dm_dir = r"/kaggle/input/pix2pix-output/train/testA" # Replace with path of digital mammo directory
cesm_dir = r"/kaggle/input/pix2pix-output/train/testB" # Replace with path of contrast mammo directory

dm_paths, cesm_paths = create_data_path_old(dm_dir, cesm_dir)


DS = Dataset()
test_ds = DS.create_dataset(dm_paths, cesm_paths,train=False)

cyclegan_model = get_cyclegan_model()

cyclegan_model.built = True

cyclegan_model.load_weights("./cyclegan_tuned.weights.h5")


generate_images(cyclegan_model.gen_G,test_ds,100)
Calculate_metrics_sewar(cyclegan_model,test_ds)
