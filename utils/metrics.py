import sewar
import numpy as np

def Calculate_metrics_sewar(model,dataloader):
 
    ssim_CG_sewar=[]
    psnr_CG_sewar=[]
    vif_CG_sewar=[]
    for inp, tar in dataloader.take(-1):
        prediction = model.gen_G(inp, training=False)
        ssim_CG_sewar.append(sewar.full_ref.ssim(tar[0].numpy(), prediction[0].numpy(),MAX=2))
        psnr_CG_sewar.append(sewar.full_ref.psnr(tar[0].numpy(), prediction[0].numpy(),MAX=2))
        vif_CG_sewar.append(sewar.full_ref.vifp(tar[0].numpy(), prediction[0].numpy()))  

    print("ssim:",np.mean(ssim_CG_sewar),"psnr:",np.mean(psnr_CG_sewar),"vif:",np.mean(vif_CG_sewar))

    return ssim_CG_sewar,psnr_CG_sewar,vif_CG_sewar
