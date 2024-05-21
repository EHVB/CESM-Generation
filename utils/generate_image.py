import matplotlib.pyplot as plt 

def generate_images(model, dataloader, number_of_samples=5):
  
  for sample_num, (test_input, target) in enumerate(dataloader.take(number_of_samples)):
    prediction = model(test_input, training=False)  
    plt.figure(figsize=(15, 15))
    display_list = [test_input[0], target[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow(display_list[i]*0.5 +0.5 , cmap="gray")

    # Save the figure with a filename based on the sample number
    plt.savefig(f'/sampleimages/sample_plot_{sample_num}.png')
    plt.close()  # Close the figure to avoid memory issues
  