# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from pipeline_stable_diffusion_instruct_pix2pix import MyStableDiffusionInstructPix2PixPipeline as StableDiffusionInstructPix2PixPipeline
from diffusers import EulerAncestralDiscreteScheduler 
import json
from gpt_eval.eval_following import call_azure_gpt4v as eval_following
from gpt_eval.eval_keep_detail import call_azure_gpt4v as eval_keep_detail
from gpt_eval.eval_quality import call_azure_gpt4v as eval_quality
import time
import argparse
import os
import numpy as np
from PIL import Image, ImageOps
import math
import re
from Unet2DConditionModel import UNet2DConditionModel
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection
from tqdm import tqdm
from utils import load_json

edit_type = ["Local", "Remove", "Add", "Texture", "Background", "Global", "Style"]


def extract_score(string):
    match = re.search(r"'score': (\d+\.\d+)", string)
    return float(match.group(1))

def load_real_edit():
    res = []
    data = load_json("./gpt_eval/real_edit_instruct.json")
    for k,v in data.items():
        img_path = "./gpt_eval/Real-Edit/" + k
        for p in v:
            res.append((img_path, p.split(": ")[-1], p.split(".")[0]))
    return res


def get_argparse():
    parser = argparse.ArgumentParser(description="eval on the real-edit dataset")
    parser.add_argument("--model_path", type=str, default="stage2_reward_instruct_pix2pix")
    parser.add_argument("--reward_score", type=str, default="5,5,5")
    parser.add_argument("--scale", type=str, default="1.5,7.5")
    parser.add_argument("--resolution", type=int, default=512)
    args = parser.parse_args()
    return args

if __name__== "__main__":
    args = get_argparse()
    model_path = args.model_path
    resolution = args.resolution
    image_scale, text_scale = args.scale.split(",")
    image_scale, text_scale = float(image_scale), float(text_scale)
    save_dir = model_path
    reward_score = torch.Tensor([float(i) for i in args.reward_score.split(",")])
    reward_text = ["None; None; None"]
    print("reward score:", reward_score)
   
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                model_path,
                text_encoder=text_encoder,
                vae=vae,
                unet=unet,
                torch_dtype=torch.float16,
                safety_checker=None).to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    generator = torch.Generator(device="cuda").manual_seed(42)

    clip_image_processor = CLIPImageProcessor()
    image_encoder = CLIPVisionModelWithProjection.from_pretrained("./CLIP-ViT-H-14-laion2B-s32B-b79K").to("cuda")
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    
    neagtive_embeddings = text_encoder(tokenizer(
                            reward_text, 
                            max_length=tokenizer.model_max_length, 
                            padding="max_length", 
                            truncation=True, 
                            return_tensors="pt",
                        ).input_ids.to("cuda"))[1]
    
    res_score = []
    fyn_list, fscore_list = [], []   
    kyn_list, kscore_list = [], []
    qyn_list, qscore_list = [], []

    data = load_real_edit()

    for img_prompt_tuple in tqdm(data): 
        img_path, prompt, edited_type = img_prompt_tuple
        stime = time.time()

        input_image = Image.open(img_path).convert('RGB')

        width, height = input_image.size
        factor = resolution / max(width, height)
        factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
        width = int((width * factor) // 64) * 64
        height = int((height * factor) // 64) * 64
        img = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)

        clip_image = clip_image_processor(images=Image.open(img_path), return_tensors="pt").pixel_values.squeeze(0).to("cuda")
        with torch.no_grad():
            clip_image_embeds = image_encoder(clip_image.unsqueeze(0)).image_embeds  # [16, 1024]

        added_cond_kwargs = {"image_embeds":  clip_image_embeds.repeat(3,1).to("cuda"), "negative_embeds": neagtive_embeddings.repeat(3,1).to("cuda"), 'reward_scores': reward_score.unsqueeze(0).repeat(3,1).to("cuda")}

        with torch.no_grad():
            edited_img = pipe(prompt, image=img, num_inference_steps=30, added_cond_kwargs=added_cond_kwargs, image_guidance_scale=1.5, generator=generator).images[0]  # guidance_scale=text_scale,

        size = Image.open(img_path).convert('RGB').size
        img = img.resize((int(resolution * size[0] / size[1]), resolution))
        edited_img = edited_img.resize((int(resolution * size[0] / size[1]), resolution))
        print("Generating Time: {}".format(time.time() - stime))

        try:
            stime = time.time()
            follow_score_str = eval_following([img, edited_img], prompt)
            # time.sleep(2)
            print("Following Time: {}".format(time.time() - stime))

            stime = time.time()
            keep_score_str = eval_keep_detail([img, edited_img], prompt)
            # time.sleep(2)
            print("Keeping Time: {}".format(time.time() - stime))

            stime = time.time()
            quality_score_str = eval_quality(edited_img)
            # time.sleep(2)
            print("Quality Time: {}".format(time.time() - stime))
        except Exception as e:
            print(e)
            continue


        if isinstance(follow_score_str, str) and isinstance(keep_score_str, str) and isinstance(quality_score_str, str):
            try:
                fyn_list.append(1 if "yes" in follow_score_str else 0)
                fscore = extract_score(follow_score_str)
                fscore_list.append(fscore)

                kyn_list.append(1 if "yes" in keep_score_str else 0)
                kscore = extract_score(keep_score_str)
                kscore_list.append(kscore)
            
                qyn_list.append(1 if "yes" in quality_score_str else 0)
                qscore = extract_score(quality_score_str)
                qscore_list.append(qscore)
            except Exception as e:
                print(e)
                continue

    with open(save_dir + '.txt', 'a+') as file:
        file.write(args.reward_score)
        file.write('num: {}\t follow_yn: {:.2f}\t follow_score: {:.2f}\t keep_yn: {:.2f}\t keep_score: {:.2f}\t qua_yn: {:.2f}\t qua_score: {:.2f}\n'.format(len(fscore_list), sum(fyn_list)/len(fyn_list), sum(fscore_list)/len(fscore_list), sum(kyn_list)/len(kyn_list), sum(kscore_list)/len(kscore_list), sum(qyn_list)/len(qyn_list), sum(qscore_list)/len(qscore_list)))
        file.write("\n" + "-"* 20)
