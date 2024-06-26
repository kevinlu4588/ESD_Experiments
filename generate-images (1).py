from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import LMSDiscreteScheduler
import torch
from PIL import Image
import pandas as pd
import argparse
import os
from esd_diffusers import FineTunedModel, StableDiffuser
import pandas as pd

def save_images(images, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, image in enumerate(images):
        img = Image.fromarray((image * 255).astype('uint8'))  # Convert tensor to PIL Image
        img.save(os.path.join(output_dir, f'image_{i}.png'))
        
def generate_images(model_path, num_img_steps, prompts_path, tune_method):
    generator = torch.manual_seed(4230)
    diffuser = StableDiffuser(scheduler='DDIM').to('cuda')
    finetuner = FineTunedModel(diffuser, train_method=tune_method)
    state_dict = torch.load(model_path)
    finetuner.load_state_dict(state_dict)

    prompts_df = pd.read_csv(prompts_path)
    prompts = prompts_df['Prompt'].tolist()

        # Function to process a batch of prompts
    def process_batch(batch_prompts):
        with finetuner:
            images = diffuser(
                batch_prompts,
                n_steps=num_img_steps,
                generator=generator
            )
        return images

    # Create a list to store the images
    all_images = []
    batch_size = 10
    batch_prompts = prompts[0: batch_size]
    batch_images = process_batch(batch_prompts)   
    save_images(batch_images, 'images/')
    # # Process prompts in batches
    # for i in range(0, len(prompts), batch_size):
    #     batch_prompts = prompts[i:i + batch_size]
    #     batch_images = process_batch(batch_prompts)
    #     all_images.extend(batch_images)

    return all_images
if __name__=='__main__':
    generate_images('/home/lu.kev/models/car_noxattn_200.pt', 10, '/home/lu.kev/Kevins Dataset.csv', 'noxattn')
    