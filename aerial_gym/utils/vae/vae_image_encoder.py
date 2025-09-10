import torch
import os
from aerial_gym.utils.vae.VAE import VAE


def clean_state_dict(state_dict):
    clean_dict = {}
    for key, value in state_dict.items():
        if "module." in key:
            key = key.replace("module.", "")
        if "dronet." in key:
            key = key.replace("dronet.", "encoder.")
        clean_dict[key] = value
    return clean_dict


class VAEImageEncoder:
    """
    Class that wraps around the VAE class for efficient inference for the aerial_gym class
    """

    def __init__(self, config, device="cuda:0"):
        self.config = config
        self.vae_model = VAE(input_dim=1, latent_dim=self.config.latent_dims).to(device)
        # combine module path with model file name
        weight_file_path = os.path.join(self.config.model_folder, self.config.model_file)
        # load model weights
        print("Loading weights from file: ", weight_file_path)
        state_dict = clean_state_dict(torch.load(weight_file_path))
        self.vae_model.load_state_dict(state_dict)
        self.vae_model.eval()

    def encode(self, image_tensors):
        with torch.no_grad():
            # print(f"VAE input shape: {image_tensors.shape}")
            
            # 确保输入是 (batch, channels, height, width) 格式
            if len(image_tensors.shape) == 3:  # (batch, H, W)
                image_tensors = image_tensors.unsqueeze(1)  # 添加通道维 -> (batch, 1, H, W)
            elif len(image_tensors.shape) == 4:  # 已经是正确格式
                pass
            else:
                raise ValueError(f"Unexpected input shape: {image_tensors.shape}")
                
            x_res, y_res = image_tensors.shape[-2], image_tensors.shape[-1]
            # print(f"Image resolution: {x_res} x {y_res}")
            
            if self.config.image_res != (x_res, y_res):
                interpolated_image = torch.nn.functional.interpolate(
                    image_tensors,
                    self.config.image_res,
                    mode=self.config.interpolation_mode,
                )
            else:
                interpolated_image = image_tensors
                
            z_sampled, means, *_ = self.vae_model.encode(interpolated_image)
        
        if self.config.return_sampled_latent:
            returned_val = z_sampled
        else:
            returned_val = means
        return returned_val

    def decode(self, latent_spaces):
        """
        Decode a latent space to reconstruct full images
        """
        with torch.no_grad():
            if latent_spaces.shape[-1] != self.config.latent_dims:
                print(
                    f"ERROR: Latent space size of {latent_spaces.shape[-1]} does not match network size {self.config.latent_dims}"
                )
            decoded_image = self.vae_model.decode(latent_spaces)
        return decoded_image

    def get_latent_dims_size(self):
        """
        Function to get latent space dims
        """
        return self.config.latent_dims
