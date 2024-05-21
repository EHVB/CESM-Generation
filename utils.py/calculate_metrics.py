from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error

def predict_calculate_metrics(model,dataloader):
     for test_input, target in dataloader.take()):
        prediction = model(test_img, training=False)

        data_range = target[0].numpy().max() - target[0].numpy().min() 
        ssim_score= ssim(target[0].numpy(), prediction[0].numpy(), channel_axis=-1,data_range=data_range)

        # Calculate PSNR
        psnr_score = psnr(target[0].numpy(), prediction[0].numpy())

        # Calculate MAE
        mse_score = mean_squared_error(target[0].numpy(), prediction[0].numpy())
        return ssim_score,psnr_score,mse_score