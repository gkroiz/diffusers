import os
from diffusers import DiffusionPipeline
import torch_xla.core.xla_model as xm

model_path = "runwayml/stable-diffusion-v1-5"
lora_path = "sd-pokemon-model-lora"
images_path = lora_path + "/inference-images"

prompt = "cute dragon creature"
num_examples = 5

# create pipeline
pipeline = DiffusionPipeline.from_pretrained(model_path)
pipeline = pipeline.to(xm.xla_device())

# load attention processors
pipeline.unet.load_attn_procs(lora_path)

if not os.path.exists(images_path):
    os.mkdir(images_path)

# inference
for i in range(num_examples):
    image = (pipeline(prompt, callback=lambda *_: xm.mark_step()).images[0])
    image.save(f"{images_path}/image-{i}.png")

print('Done inference!')