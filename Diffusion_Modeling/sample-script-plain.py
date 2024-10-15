import torch

import sys
sys.path.append('..') 


from video_diffusion_pytorch_project.video_diffusion_pytorch import Unet3D, GaussianDiffusion, Trainer

denoise_model = Unet3D(
    dim = 32,
    dim_mults = (1,2,4),
    channels = 1,
    attn_heads = 8,
    attn_dim_head = 16,
    resnet_groups = 8
)

# linearmap_model = LinearMappingModel()

diffusion = GaussianDiffusion(
    denoise_model,
    # linearmap_model,
    image_size = 4,
    num_frames = 189,
    channels= 1,
    timesteps = 200,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()


diffusion.denoise_fn.load_state_dict(torch.load('../ckpts/denoise_fn-trained-4-4-state-dict.pth'))


# print(model_state_dict.keys())
sampled_videos = diffusion.sample(batch_size = 8)

print("sampled_videos shape:", sampled_videos.shape)

torch.save(sampled_videos, '../samples/sampled_videos-4-4-uguided-p2.pth')
# diffusion.load_state_dict(model_state_dict)

print("Sampling Success!")