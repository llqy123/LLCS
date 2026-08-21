import numpy as np
from functools import partial
from six.moves import map, zip
import torch
from src.segment_anything import sam_model_registry

def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.
    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.
    Args:
        func (Function): A function that will be applied to a list of
            arguments
    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def mask2box(mask):
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    return x0, y0, x1, y1


def build_vfm(name):
    sam_ckpts = {
        "sam-B": "sam_vit_b_01ec64.pth",
        "sam-L": "sam_vit_l_0b3195.pth",
    }

    dinov2_ckpts = {
        "dinov2-L": "dinov2_vitl14_reg",
        "dinov2-B": "dinov2_vitb14_reg",
    }

    dino_ckpts = {
        "dino-B-8": "dino_vitb8",
        "dino-B-16": "dino_vitb16",
    }

    vfm = None
    # SAM
    if name.startswith("sam"):
        print(f"Using {name}......")
        if name in sam_ckpts:
            vit_type = "vit_b" if "B" in name else "vit_l"
            # checkpoint_name = sam_ckpts[name]
            try:
                if vit_type == "vit_b":
                    checkpoint_name = "checkpoints/sam_vit_b_01ec64.pth"
                else:
                    checkpoint_name = "checkpoints/sam_vit_l_0b3195.pth"
                vfm = sam_model_registry[vit_type](checkpoint=checkpoint_name).half()
            except Exception as e:
                raise RuntimeError(f"Failed to load SAM model '{name}' with checkpoint '{checkpoint_name}': {e}")
        else:
            raise NotImplementedError(f"VLM model '{name}' not supported under SAM category.")

    # DINOv2
    elif name.startswith("dinov2"):
        print(f"Using {name}......")
        if name in dinov2_ckpts:
            model_name = dinov2_ckpts[name]
            try:
                local_repo = "/data01/lqy/.cache/torch/hub/facebookresearch_dinov2_main"
                vfm = torch.hub.load(
                    local_repo,  # 本地路径
                    model_name,  # 比如 'dino_vitb16', 'dino_resnet50'
                    source='local'
                ).half()

                # vfm = torch.hub.load(
                #     'facebookresearch/dinov2',
                #     model_name,
                #     source='github'
                # ).half()
            except Exception as e:
                raise RuntimeError(f"Failed to load DINOv2 model '{name}': {e}")
        else:
            raise NotImplementedError(f"VLM model '{name}' not supported under DINOv2 category.")

    # DINO
    elif name.startswith("dino"):
        print(f"Using {name}......")
        if name in dino_ckpts:
            model_name = dino_ckpts[name]
            try:
                local_repo = "/data01/lqy/.cache/torch/hub/facebookresearch-dino-7c446df"
                vfm = torch.hub.load(
                    local_repo,
                    model_name,
                    source='local'
                ).half()
            except Exception as e:
                raise RuntimeError(f"Failed to load DINO model '{name}': {e}")
        else:
            raise NotImplementedError(f"VLM model '{name}' not supported under DINO category.")

    else:
        raise NotImplementedError(f"VLM model '{name}' not supported.")

    for p in vfm.parameters():
        p.requires_grad = False

    return vfm