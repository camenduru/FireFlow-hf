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

torch_device = "cuda" if torch.cuda.is_available() else "cpu"
offload = False
device = "cuda" if torch.cuda.is_available() else "cpu"
name = 'flux-dev'
ae = load_ae(name, device="cpu" if offload else torch_device)
t5 = load_t5(device, max_length=256 if name == "flux-schnell" else 512)
clip = load_clip(device)
model = load_flow_model(name, device="cpu" if offload else torch_device)
is_schnell = False
output_dir = 'result'
add_sampling_metadata = True

@spaces.GPU(duration=120)
@torch.inference_mode()
def edit(init_image, source_prompt, target_prompt, editing_strategy, num_steps, inject_step, guidance, seed):
    global ae, t5, clip, model, name, is_schnell, output_dir, add_sampling_metadata
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
    if offload:
        model.cpu()
        torch.cuda.empty_cache()
        ae.encoder.to(device)
        
    with torch.no_grad():
        init_image = ae.encode(init_image.to()).to(torch.bfloat16)

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
    
    if offload:
        ae = ae.cpu()
        torch.cuda.empty_cache()
        t5, clip = t5.to(torch_device), clip.to(torch_device)
        
    print(f"Generating with seed {opts.seed}:\n{opts.source_prompt}")
    t0 = time.perf_counter()

    opts.seed = None

    #############inverse#######################
    info = {}
    info['feature'] = {}
    info['inject_step'] = min(inject_step, num_steps)
    info['reuse_v']= False
    info['editing_strategy']= " ".join(editing_strategy)
    info['start_layer_index'] = 0
    info['end_layer_index'] = 37
    qkv_ratio = '1.0,1.0,1.0'
    info['qkv_ratio'] = list(map(float, qkv_ratio.split(',')))

    with torch.no_grad():
        inp = prepare(t5, clip, init_image, prompt=opts.source_prompt)
        inp_target = prepare(t5, clip, init_image, prompt=opts.target_prompt)
    timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(name != "flux-schnell"))
    
    if offload:
        t5, clip = t5.cpu(), clip.cpu()
        torch.cuda.empty_cache()
        model = model.to(torch_device)

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
    
    if offload:
        model.cpu()
        torch.cuda.empty_cache()
        ae.decoder.to(x.device)
            
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
    # img.save(fn, exif=exif_data, quality=95, subsampling=0)

    print("End Edit")
    return img


def create_demo(model_name: str, device: str = "cuda:0" if torch.cuda.is_available() else "cpu"):
    is_schnell = model_name == "flux-schnell"
    title = r"""
        <h1 align="center">🔥FireFlow: Fast Inversion of Rectified Flow for Image Semantic Editing</h1>
        """
    description = r"""
        <b>Official 🤗 Gradio Demo</b> for <a href='https://github.com/HolmesShuan/FireFlow-Fast-Inversion-of-Rectified-Flow-for-Image-Semantic-Editing' target='_blank'><b>🔥FireFlow: Fast Inversion of Rectified Flow for Image Semantic Editing</b></a>.<br>
        <b>Tips</b> 🔔: If the results are not satisfactory, consider slightly increasing the total number of timesteps 📈. Each editing technique produces distinct effects, so feel free to experiment and explore their possibilities!
    """
    article = r"""
    If you find our work helpful, we would greatly appreciate it if you could ⭐ our <a href='https://github.com/HolmesShuan/FireFlow-Fast-Inversion-of-Rectified-Flow-for-Image-Semantic-Editing' target='_blank'>GitHub repository</a>. Thank you for your support!
    """
    css = '''
    .gradio-container {width: 85% !important}
    '''
    
    # Pre-defined examples
    examples = [
        ["example_images/dog.jpg", "Photograph of a dog on the grass", "Photograph of a cat on the grass", ['replace_v'], 8, 1, 2.0],
        ["example_images/gold.jpg", "3d melting gold render", "a cat in the style of 3d melting gold render", ['replace_v'], 8, 1, 2.0],
        ["example_images/gold.jpg", "3d melting gold render", "a cat in the style of 3d melting gold render", ['replace_v'], 10, 1, 2.0],
        ["example_images/art.jpg", "", "a vivid depiction of the Batman, featuring rich, dynamic colors,  and a blend of realistic and abstract elements with dynamic splatter art.", ['add_q'], 8, 1, 2.0],
    ]
    
    with gr.Blocks(css=css) as demo:
        # Add a title, description, and additional information
        gr.HTML(title)
        gr.Markdown(description)
        gr.Markdown(article)
        
        # Layout: Two columns
        with gr.Row():
            # Left Column: Inputs
            with gr.Column():
                init_image = gr.Image(label="Input Image", visible=True)
                source_prompt = gr.Textbox(label="Source Prompt", value="", placeholder="(Optional) Describe the content of the uploaded image.")
                target_prompt = gr.Textbox(label="Target Prompt", value="", placeholder="(Required) Describe the desired content of the edited image.")
                # CheckboxGroup for editing strategies
                editing_strategy = gr.CheckboxGroup(
                    label="Editing Technique",
                    choices=['replace_v', 'add_q', 'add_k'],
                    value=['replace_v'],  # Default: none selected
                    interactive=True
                )
                generate_btn = gr.Button("Generate")
            
            # Right Column: Advanced options and output
            with gr.Column():
                with gr.Accordion("Advanced Options", open=True):
                    num_steps = gr.Slider(
                        minimum=1, 
                        maximum=30, 
                        value=8, 
                        step=1, 
                        label="Total timesteps"
                    )
                    inject_step = gr.Slider(
                        minimum=1, 
                        maximum=15, 
                        value=1, 
                        step=1, 
                        label="Feature sharing steps"
                    )
                    guidance = gr.Slider(
                        minimum=1.0, 
                        maximum=8.0, 
                        value=2.0, 
                        step=0.1, 
                        label="Guidance", 
                        interactive=not is_schnell
                    )
                
                # Output display
                output_image = gr.Image(label="Generated Image")

        # Button click event to trigger the edit function
        generate_btn.click(
            fn=edit,
            inputs=[
                init_image, 
                source_prompt, 
                target_prompt, 
                editing_strategy,  # Include the selected editing strategies
                num_steps, 
                inject_step, 
                guidance
            ],
            outputs=[output_image]
        )
        
        # Add examples
        gr.Examples(
            examples=examples,
            inputs=[
                init_image, 
                source_prompt, 
                target_prompt, 
                editing_strategy, 
                num_steps, 
                inject_step, 
                guidance
            ]
        )
        
    return demo


demo = create_demo("flux-dev", "cuda")
demo.launch()