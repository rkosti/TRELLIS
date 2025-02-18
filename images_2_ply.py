import sys
import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

os.environ['TORCH_CUDA_ARCH_LIST']='8.6'

# Suppress FutureWarnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from pathlib import Path
import time 
from loguru import logger 
logger.remove()
logger.add(sys.stderr, format="{message}", level="INFO")
import numpy as np
import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
# from trellis.pipelines import TrellisImageToSparse3DPipeline
from trellis.utils import render_utils

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()
# config parameters 
sampling_steps = 12

# breakpoint()
# load images 
# data_path = Path("/home/atlas2/work/data/rendered_images/bad_miner_dataset/")
# images = list(Path("assets/example_multi_image").glob("*.png"))
# images.sort()
data_path = Path("/home/atlas2/work/data/rendered_images/bad_miner_dataset/")
save_folder = Path("reconstructed_examples")
for _dir in data_path.iterdir():
    dir_name = _dir.stem 
    img_list = list(_dir.glob("*.png"))
    img_list.sort()
    total_models = len(img_list) // 16
    for model_id in range(total_models):
        model_name = img_list[0 + model_id*16].stem.split(".")[0]
        sample_images = [Image.open(img_list[0 + model_id*16]), Image.open(img_list[6 + model_id*16]), Image.open(img_list[8 + model_id*16])]
        save_path = save_folder / dir_name
        if not save_path.exists():
            save_path.mkdir()
        # sample_images = [Image.open(img) for img in images[:3]]


        # Run the pipeline
        t1 = time.time()
        outputs = pipeline.run_multi_image(
            sample_images,
            seed=1,
            # Optional parameters
            sparse_structure_sampler_params={
                "steps": sampling_steps,
                "cfg_strength": 7.5,
            },
            slat_sampler_params={
                "steps": sampling_steps,
                "cfg_strength": 3,
            },
            formats='gaussian'
        )
        t2 = time.time()
        logger.info(f"Pipeline took: {t2 - t1} sec.")
        # logger.info(f" Coords: {outputs[0].shape} \n Occupancy list: {outputs[1].shape} ")
        # logger.info(f"conditioning features: {outputs[2].shape}")
        # breakpoint()

        save_filename = str(save_path / model_name)
        # saving the generated gaussian 3D model as ply file
        outputs['gaussian'][0].save_ply(save_filename + '.ply')


