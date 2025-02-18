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
# from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.pipelines import TrellisImageToSparse3DPipeline
from trellis.utils import render_utils

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageToSparse3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()

# breakpoint()
# load images 
images = list(Path("assets/example_multi_image").glob("*.png"))
images.sort()
sample_images = [Image.open(img) for img in images[:3]]

# config parameters 
sampling_steps = 12

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
logger.info(f" Coords: {outputs[0].shape} \n Occupancy list: {outputs[1].shape} ")
# logger.info(f"conditioning features: {outputs[2].shape}")
# breakpoint()

# # saving the generated gaussian 3D model as ply file
# outputs['gaussian'][0].save_ply(f'reconstructed_examples/{images[0].stem}_steps_{sampling_steps}.ply')


