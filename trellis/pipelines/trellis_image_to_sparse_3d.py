from typing import *
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from torchvision import transforms
from PIL import Image
import rembg
from .base import Pipeline
from . import samplers
from ..representations import Gaussian, Strivec, MeshExtractResult
from loguru import logger

class TrellisImageToSparse3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis image-to-3D models using only the sparse model.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        image_cond_model (str): The name of the image conditioning model.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        image_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.sparse_structure_sampler_params = {}
        self.rembg_session = None
        self._init_image_cond_model(image_cond_model)

    @staticmethod
    def from_pretrained(path: str) -> "TrellisImageToSparse3DPipeline":
        """
        Load a pretrained model using only the sparse model.

        Args:
            path (str): The path to the model.
        """
        pipeline = super(TrellisImageToSparse3DPipeline, TrellisImageToSparse3DPipeline).from_pretrained(path)
        new_pipeline = TrellisImageToSparse3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(
            samplers, args['sparse_structure_sampler']['name']
        )(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']
        new_pipeline._init_image_cond_model(args['image_cond_model'])
        return new_pipeline

    def _init_image_cond_model(self, name: str):
        """
        Initialize the image conditioning model.
        """
        dinov2_model = torch.hub.load('facebookresearch/dinov2', name, pretrained=True)
        dinov2_model.eval()
        self.models['image_cond_model'] = dinov2_model
        self.image_cond_model_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        """
        # Check if alpha channel is valid, otherwise remove background.
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            max_size = max(input.size)
            scale = min(1, 1024 / max_size)
            if scale < 1:
                input = input.resize(
                    (int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS
                )
            if getattr(self, 'rembg_session', None) is None:
                self.rembg_session = rembg.new_session('u2net')
            output = rembg.remove(input, session=self.rembg_session)
            output_np = np.array(output)
            alpha = output_np[:, :, 3]
            bbox = np.argwhere(alpha > 0.8 * 255)
            bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
            size = int(size * 1.2)
            bbox = (
                center[0] - size // 2,
                center[1] - size // 2,
                center[0] + size // 2,
                center[1] + size // 2,
            )
            output = output.crop(bbox)  # type: ignore
            output = output.resize((518, 518), Image.Resampling.LANCZOS)
            output = np.array(output).astype(np.float32) / 255
            output = output[:, :, :3] * output[:, :, 3:4]
            output = Image.fromarray((output * 255).astype(np.uint8))
        return output

    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, list[Image.Image]]) -> torch.Tensor:
        """
        Encode the image.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image to encode.

        Returns:
            torch.Tensor: The encoded features.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")

        image = self.image_cond_model_transform(image).to(self.device)
        features = self.models['image_cond_model'](image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens

    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information.
        """
        cond = self.encode_image(image)
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample sparse structures with the given conditioning.

        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The coordinates of the sparse structure and the decoded output.
        """
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples

        decoder = self.models['sparse_structure_decoder']
        decoded_op = decoder(z_s)
        coords = torch.argwhere(decoded_op > 0)[:, [0, 2, 3, 4]].int()

        return coords, decoded_op

    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        preprocess_image: bool = True,
    ) -> dict:
        """
        Run the pipeline using only the sparse model.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The seed for reproducibility.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            preprocess_image (bool): Whether to preprocess the image.

        Returns:
            dict: The output containing the sparse structure coordinates and the decoded output.
        """
        if preprocess_image:
            image = self.preprocess_image(image)
        cond = self.get_cond([image])
        torch.manual_seed(seed)
        coords, decoded_op = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        return {
            "sparse_structure_coords": coords,
            "decoded_op": decoded_op,
        }