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

from pathlib import Path
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import random
from utils import load_json
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from transformers import CLIPImageProcessor
from glob import glob


# Dataset
class Insp2pDataset(torch.utils.data.Dataset):

    def __init__(self, args):
        super().__init__()
        self.pix2pix_root_path = "./pix2pix-clip-filtered-hf/train/"
      
        pix2pix2 = load_json('./pix2pix-clip-filtered-hf/train/seeds.json') 
    
        self.seeds = [[0] + i for i in pix2pix2]

        self.resolution = args.resolution
        self.train_transforms = transforms.Compose(
            [
                transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
                transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            ]
        )
    
    def transform_img(self, img0, img1):
        try:
            img0 = img0.convert("RGB").resize((self.resolution, self.resolution))
            img1 = img1.convert("RGB").resize((self.resolution, self.resolution))
        except:
            img0 = Image.new('RGB', (self.resolution, self.resolution))
            img1 = Image.new('RGB', (self.resolution, self.resolution))

        img0 = np.array(img0).transpose(2, 0, 1)
        img1 = np.array(img1).transpose(2, 0, 1)
        images = np.concatenate([img0, img1])
        images = torch.tensor(images)
        images = 2 * (images / 255) - 1
        img0, img1 = self.train_transforms(images).chunk(2)
        return img0, img1

    def __getitem__(self, i):
        _, name, seeds = self.seeds[i]

        propt_dir = Path(self.pix2pix_root_path, name)
        seed = seeds[torch.randint(0, len(seeds), ()).item()]
        
        edit_str = load_json(propt_dir.joinpath("prompt.json"))["edit"]
      
        if isinstance(edit_str, str):
            text = edit_str
        else:
            text = random.choice(edit_str)

        raw_image_0 = Image.open(propt_dir.joinpath(f"{seed}_0.jpg"))
        raw_image_1 = Image.open(propt_dir.joinpath(f"{seed}_1.jpg"))

        original_images, edited_images = self.transform_img(raw_image_0, raw_image_1)

        return {
            'edit_prompt': text,
            "original_pixel_values": original_images,
            "edited_pixel_values": edited_images,
            "original_imge_mask": 1,
        }

    def __len__(self):
        return len(self.seeds)


class MRewardDataset(torch.utils.data.Dataset):

    def __init__(self, args):
        super().__init__()
        
        self.img_dir = "./pix2pix-clip-filtered-hf/train/"
        self.data_all = load_json("RewardEdit_20K.json")

        self.file_list = list(self.data_all.keys())
      
        self.resolution = args.resolution
        self.train_transforms = transforms.Compose(
            [
                transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
                transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            ]
        )
        self.clip_image_processor = CLIPImageProcessor()
    
    def transform_img(self, img0, img1):
        try:
            img0 = img0.convert("RGB").resize((self.resolution, self.resolution))
            img1 = img1.convert("RGB").resize((self.resolution, self.resolution))
        except:
            img0 = Image.new('RGB', (self.resolution, self.resolution))
            img1 = Image.new('RGB', (self.resolution, self.resolution))

        img0 = np.array(img0).transpose(2, 0, 1)
        img1 = np.array(img1).transpose(2, 0, 1)
        images = np.concatenate([img0, img1])
        images = torch.tensor(images)
        images = 2 * (images / 255) - 1
        img0, img1 = self.train_transforms(images).chunk(2)
        return img0, img1

    def __getitem__(self, i):

        data = self.data_all[self.file_list[i]]
        img_path_list = glob(self.img_dir + self.file_list[i] + "/*")
        original_img_path = img_path_list[0]
        edited_img_path = img_path_list[1]
        text = load_json(img_path_list[2])["edit"]
       
        raw_image_0 = Image.open(original_img_path)
        raw_image_1 = Image.open(edited_img_path)

        original_images, edited_images = self.transform_img(raw_image_0, raw_image_1)

        score_list = [d['Score'] for d in data]

        negative_prompt = "; ".join([d['Negative Prompt'] for d in data])

        clip_image = self.clip_image_processor(images=raw_image_0, return_tensors="pt").pixel_values.squeeze(0)
        return {
            'edit_prompt': text,
            "negative_prompt": negative_prompt,
            "original_pixel_values": original_images,
            "edited_pixel_values": edited_images,
            "original_imge_mask": 1,
            "score_list": score_list,
            "clip_image": clip_image,
        }

    def __len__(self):
        return len(self.file_list)