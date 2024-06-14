from utils.create_data_path import create_data_path
from models.cyclegan import get_cyclegan_model
from data.dataset import Dataset
import keras
from utils.create_data_path import create_data_path_old
from utils.generate_image import generate_images
import matplotlib.pyplot as plt


dm_dir = r"./CDD-CESM/Low energy images of CDD-CESM" # Replace with path of digital mammo directory
cesm_dir = r"./CDD-CESM/Subtracted images of CDD-CESM" # Replace with path of contrast mammo directory

dm_paths, cesm_paths = create_data_path_old(dm_dir, cesm_dir)



DS = Dataset()
train_ds = DS.create_dataset(dm_paths, cesm_paths)

cyclegan_model = get_cyclegan_model()

cyclegan_model.built = True

cyclegan_model.load_weights("./weights/cyclegan_pretrained.weights.h5") # path to new model weights 

# generate 20 sample images 
sample_num = 0
number_of_samples = 20
for test_input, target in train_ds:
        if sample_num >= number_of_samples:
            break

        prediction = cyclegan_model.gen_G(test_input, training=False)
        print(type(test_input))
        print(type(target))
        print(type(cyclegan_model.gen_G))
        print(type(prediction))

        plt.figure(figsize=(15, 15))
        display_list = [test_input[0], target[0], prediction[0]]
        title = ['Input Image', 'Ground Truth', 'Predicted Image']

        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.title(title[i])
            # Getting the pixel values in the [0, 1] range to plot.
            plt.imshow(display_list[i] * 0.5 + 0.5, cmap="gray")

        if not os.path.exists('/sampleimages'):
            os.makedirs('/sampleimages')
            print(f"Directory '{'/sampleimages'}' created.")
        else:
            print(f"Directory '{'/sampleimages'}' already exists.")

        # Save the figure with a filename based on the sample number
        plt.savefig(f'/sampleimages/sample_plot_{sample_num}.png')
        print(f"save sample {sample_num}")
        plt.close()  # Close the figure to avoid memory issues
        sample_num += 1
