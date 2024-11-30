import os
import re
import time
from io import BytesIO
import uuid
from dataclasses import dataclass
from glob import iglob
import argparse
from einops import rearrange
from fire import Fire
from PIL import ExifTags, Image
import spaces

import torch
import torch.nn.functional as F
import gradio as gr
import numpy as np
from transformers import pipeline

from flux.sampling import denoise, get_schedule, prepare, unpack
from flux.util import (configs, embed_watermark, load_ae, load_clip, load_flow_model, load_t5)
from huggingface_hub import login
login(token=os.getenv('Token'))

import torch


@dataclass
class SamplingOptions:
    source_prompt: str
    target_prompt: str
    # prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None

@torch.inference_mode()
def encode(init_image, torch_device):
    init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 127.5 - 1
    init_image = init_image.unsqueeze(0) 
    init_image = init_image.to(torch_device)
    with torch.no_grad():
        init_image = ae.encode(init_image.to()).to(torch.bfloat16)
    return init_image


device = "cuda" if torch.cuda.is_available() else "cpu"
name = 'flux-dev'
ae = load_ae(name, device)
t5 = load_t5(device, max_length=256 if name == "flux-schnell" else 512)
clip = load_clip(device)
model = load_flow_model(name, device=device)
offload = False
name = "flux-dev"
is_schnell = False
feature_path = 'feature'
output_dir = 'result'
add_sampling_metadata = True

@spaces.GPU(duration=120)
@torch.inference_mode()
def edit(init_image, source_prompt, target_prompt, num_steps, inject_step, guidance, seed):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    seed = None
        
    shape = init_image.shape

    new_h = shape[0] if shape[0] % 16 == 0 else shape[0] - shape[0] % 16
    new_w = shape[1] if shape[1] % 16 == 0 else shape[1] - shape[1] % 16

    init_image = init_image[:new_h, :new_w, :]

    width, height = init_image.shape[0], init_image.shape[1]


    init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 127.5 - 1
    init_image = init_image.unsqueeze(0) 
    init_image = init_image.to(device)
    with torch.no_grad():
        init_image = ae.encode(init_image.to()).to(torch.bfloat16)

    print(init_image.shape)

    rng = torch.Generator(device="cpu")
    opts = SamplingOptions(
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            width=width,
            height=height,
            num_steps=num_steps,
            guidance=guidance,
            seed=seed,
        )
    if opts.seed is None:
        opts.seed = torch.Generator(device="cpu").seed()
        
    print(f"Generating with seed {opts.seed}:\n{opts.source_prompt}")
    t0 = time.perf_counter()

    opts.seed = None

    #############inverse#######################
    info = {}
    info['feature'] = {}
    info['inject_step'] = inject_step

    with torch.no_grad():
        inp = prepare(t5, clip, init_image, prompt=opts.source_prompt)
        inp_target = prepare(t5, clip, init_image, prompt=opts.target_prompt)
    timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(name != "flux-schnell"))

    # inversion initial noise
    with torch.no_grad():
        z, info = denoise(model, **inp, timesteps=timesteps, guidance=1, inverse=True, info=info)
        
    inp_target["img"] = z

    timesteps = get_schedule(opts.num_steps, inp_target["img"].shape[1], shift=(name != "flux-schnell"))

    # denoise initial noise
    x, _ = denoise(model, **inp_target, timesteps=timesteps, guidance=guidance, inverse=False, info=info)

    # decode latents to pixel space
    x = unpack(x.float(), opts.width, opts.height)

    output_name = os.path.join(output_dir, "img_{idx}.jpg")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        idx = 0
    else:
        fns = [fn for fn in iglob(output_name.format(idx="*")) if re.search(r"img_[0-9]+\.jpg$", fn)]
        if len(fns) > 0:
            idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
        else:
            idx = 0
            
    device = torch.device("cuda")
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        x = ae.decode(x)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    fn = output_name.format(idx=idx)
    print(f"Done in {t1 - t0:.1f}s. Saving {fn}")
    # bring into PIL format and save
    x = x.clamp(-1, 1)
    x = embed_watermark(x.float())
    x = rearrange(x[0], "c h w -> h w c")

    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
    exif_data = Image.Exif()
    exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
    exif_data[ExifTags.Base.Make] = "Black Forest Labs"
    exif_data[ExifTags.Base.Model] = name
    if add_sampling_metadata:
        exif_data[ExifTags.Base.ImageDescription] = source_prompt
    img.save(fn, exif=exif_data, quality=95, subsampling=0)

    print("End Edit")
    return img



def create_demo(model_name: str, device: str = "cuda:0" if torch.cuda.is_available() else "cpu", offload: bool = False):
    is_schnell = model_name == "flux-schnell"

    with gr.Blocks() as demo:
        gr.Markdown(f"# RF-Edit Demo (FLUX for image editing)")
        
        with gr.Row():
            with gr.Column():
                # source_prompt = gr.Textbox(label="Source Prompt", value="")
                # target_prompt = gr.Textbox(label="Target Prompt", value="")
                source_prompt = gr.Text(
                    label="Source Prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter your source prompt",
                    container=False,
                    value="" 
                )
                target_prompt = gr.Text(
                    label="Target Prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter your target prompt",
                    container=False,
                    value="" 
                )
                init_image = gr.Image(label="Input Image", visible=True)
                
                
                generate_btn = gr.Button("Generate")
            
            with gr.Column():
                with gr.Accordion("Advanced Options", open=True):
                    num_steps = gr.Slider(1, 30, 25, step=1, label="Number of steps")
                    inject_step = gr.Slider(1, 15, 5, step=1, label="Number of inject steps")
                    guidance = gr.Slider(1.0, 10.0, 2, step=0.1, label="Guidance", interactive=not is_schnell)
                    # seed = gr.Textbox(0, label="Seed (-1 for random)", visible=False)
                    # add_sampling_metadata = gr.Checkbox(label="Add sampling parameters to metadata?", value=False)
                
                output_image = gr.Image(label="Generated Image")

        generate_btn.click(
            fn=edit,
            inputs=[init_image, source_prompt, target_prompt, num_steps, inject_step, guidance],
            outputs=[output_image]
        )


    return demo


# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="Flux")
#     parser.add_argument("--name", type=str, default="flux-dev", choices=list(configs.keys()), help="Model name")
#     parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to use")
#     parser.add_argument("--offload", action="store_true", help="Offload model to CPU when not in use")
#     parser.add_argument("--share", action="store_true", help="Create a public link to your demo")

#     parser.add_argument("--port", type=int, default=41035)
#     args = parser.parse_args()

demo = create_demo("flux-dev", "cuda")
demo.launch()