import os

import groundingdino.datasets.transforms as T
import numpy as np
import torch
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.inference import predict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from huggingface_hub import hf_hub_download
from segment_anything import sam_model_registry
from segment_anything import SamPredictor

# Global constants for model URLs and filenames
_SAM_MODELS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
}

_GROUNDING_DINO_REPO_ID = "ShilongLiu/GroundingDINO"
_GROUNDING_DINO_CKPT_FILENAME = "groundingdino_swinb_cogcoor.pth"
_GROUNDING_DINO_CONFIG_FILENAME = "GroundingDINO_SwinB.cfg.py"

# Default cache path for downloading models
_DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/torch/hub/checkpoints")


def _load_model_hf(repo_id, filename, config_filename, cache_dir, device='cpu'):
    """Load a Hugging Face model with a specific checkpoint and configuration, using a custom cache directory."""
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=config_filename, cache_dir=cache_dir)

    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print(f"Model loaded from {cache_file} \n => {log}")
    model.eval()
    return model


def _transform_image(image) -> torch.Tensor:
    """Transform the input image for model inference."""
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image_transformed, _ = transform(image, None)
    return image_transformed


class LangSAM:

    def __init__(self, sam_type="vit_h", ckpt_path=None, return_prompts=False, cache_dir=_DEFAULT_CACHE_DIR):
        """Initialize the LangSAM object, setting up both SAM and GroundingDINO models.

        :param sam_type: Type of SAM model (e.g., "vit_h", "vit_l", "vit_b")
        :param ckpt_path: Path to the SAM model checkpoint. If None, it will download from the default URL.
        :param return_prompts: Boolean flag for whether to return prompts from GroundingDINO.
        :param cache_dir: Path to the cache directory where models will be stored.
        """
        self.sam_type = sam_type
        self.ckpt_path = ckpt_path
        self.return_prompts = return_prompts
        self.cache_dir = cache_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build models
        self.build_groundingdino()
        self.build_sam()

    def build_sam(self):
        """Build the SAM model, either loading from a custom checkpoint path or downloading from the default
        URL."""
        if self.ckpt_path is None:
            # No checkpoint path provided, use default URL
            checkpoint_url = _SAM_MODELS.get(self.sam_type, _SAM_MODELS["vit_h"])
            try:
                state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, model_dir=self.cache_dir)
                sam = sam_model_registry[self.sam_type]()
                sam.load_state_dict(state_dict, strict=True)
            except Exception as e:
                raise ValueError(f"Error loading SAM model: {self.sam_type}. "
                                 f"Check model type and checkpoint URL: {checkpoint_url}. Error: {str(e)}")
            sam.to(self.device)
            self.sam = SamPredictor(sam)
        else:
            # Load from the provided checkpoint path
            try:
                sam = sam_model_registry[self.sam_type](self.ckpt_path)
            except Exception as e:
                raise ValueError(f"Error loading SAM model: {self.sam_type}. Ensure the checkpoint path "
                                 f"matches the model type. Error: {str(e)}")
            sam.to(self.device)
            self.sam = SamPredictor(sam)

    def build_groundingdino(self):
        """Build the GroundingDINO model, loading the configuration and weights from Hugging Face."""
        self.groundingdino = _load_model_hf(repo_id=_GROUNDING_DINO_REPO_ID,
                                            filename=_GROUNDING_DINO_CKPT_FILENAME,
                                            config_filename=_GROUNDING_DINO_CONFIG_FILENAME,
                                            cache_dir=self.cache_dir)

    def predict_dino(self, image_pil, text_prompt, box_threshold, text_threshold):
        """Run the GroundingDINO model on an image with a text prompt to generate bounding boxes and logits.

        :param image_pil: Input image in PIL format.
        :param text_prompt: Text prompt for object detection.
        :param box_threshold: Threshold for bounding box confidence.
        :param text_threshold: Threshold for text prediction confidence.
        :return: Bounding boxes, logits, and phrases detected by GroundingDINO.
        """
        image_trans = _transform_image(image_pil)
        boxes, logits, phrases = predict(model=self.groundingdino,
                                         image=image_trans,
                                         caption=text_prompt,
                                         box_threshold=box_threshold,
                                         text_threshold=text_threshold,
                                         remove_combined=self.return_prompts,
                                         device=self.device)
        W, H = image_pil.size
        boxes = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

        return boxes, logits, phrases

    def predict_sam(self, image_pil, boxes):
        """Run the SAM model on an image with bounding boxes to generate masks.

        :param image_pil: Input image in PIL format.
        :param boxes: Bounding boxes generated by GroundingDINO.
        :return: Masks generated by SAM.
        """
        image_array = np.asarray(image_pil)
        self.sam.set_image(image_array)
        transformed_boxes = self.sam.transform.apply_boxes_torch(boxes, image_array.shape[:2])
        masks, _, _ = self.sam.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.sam.device),
            multimask_output=False,
        )
        return masks.cpu()

    def predict(self, image_pil, text_prompt, box_threshold=0.3, text_threshold=0.25):
        """Predict both bounding boxes and masks using GroundingDINO and SAM models.

        :param image_pil: Input image in PIL format.
        :param text_prompt: Text prompt for object detection.
        :param box_threshold: Threshold for bounding box confidence.
        :param text_threshold: Threshold for text prediction confidence.
        :return: Masks, bounding boxes, phrases, and logits.
        """
        boxes, logits, phrases = self.predict_dino(image_pil, text_prompt, box_threshold, text_threshold)
        masks = torch.tensor([])
        if len(boxes) > 0:
            masks = self.predict_sam(image_pil, boxes)
            masks = masks.squeeze(1)
        return masks, boxes, phrases, logits
