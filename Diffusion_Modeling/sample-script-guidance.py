import torch

import sys
sys.path.append('..')  


from video_diffusion_pytorch_guided.video_diffusion_pytorch import Unet3D, LinearMappingModel, GaussianDiffusion, Trainer

denoise_model = Unet3D(
    dim = 32,
    dim_mults = (1,2,4),
    channels = 1,
    attn_heads = 8,
    attn_dim_head = 16,
    resnet_groups = 8
)

linearmap_model = LinearMappingModel()

diffusion = GaussianDiffusion(
    denoise_model,
    linearmap_model,
    image_size = 4,
    num_frames = 189,
    channels= 1,
    timesteps = 200,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()


# videos = torch.randn(1, 3, 30, 32, 32) # video (batch, channels, frames, height, width) - normalized from -1 to +1
# loss = diffusion(videos)
# loss.backward()
# after a lot of training

# sampled_videos = diffusion.sample(batch_size = 4)
# sampled_videos.shape # (4, 3, 30, 32, 32)

# loaded_model = torch.load('diffusion-model-trained-4-4.pth')
# state_dict = loaded_model.state_dict()
# diffusion.denoise_fn.load_state_dict(torch.load('../ckpts/diffusion-model-trained-4-4-state-dict.pth'))


# Save the Denoising model
# loaded_model = torch.load('diffusion-model-pre-4-4.pth')
# state_dict = loaded_model.denoise_fn.state_dict()
# torch.save(state_dict, '../ckpts/denoise_fn-trained-4-4-state-dict.pth')

# Load the Denoising Model
# print("pre Loading Success!")
brain_region = 'VIS-R'

diffusion.denoise_fn.load_state_dict(torch.load('../ckpts/denoise_fn-trained-4-4-state-dict.pth'))
diffusion.linearmap_fn.load_state_dict(torch.load('../ckpts/linear_model_weights.pth'))
# print("after Loading Success!")


# torch.save(state_dict, '../ckpts/diffusion-model-trained-4-4-state-dict.pth')



perturbed_dim = 3

generate_batch = 7

# print(model_state_dict.keys())
cg_weight = 1.25
# 64 separate into 6
sampled_videos = diffusion.sample(batch_size = 16, perturbed_dim=perturbed_dim, cg_weight=cg_weight)

torch.save(sampled_videos, '../samples/sampled_videos-' \
            + '-guided-p' + str(generate_batch) +'-w' + str(cg_weight) + '.pth')
# diffusion.load_state_dict(model_state_dict)
