"""JoyEcho Reference Batch - None-tolerant image batcher for reference images.

Combines up to 4 optional IMAGE inputs into one batch for
JoyEcho_Generate.reference_image. Unlike generic batch nodes (KJNodes
ImageBatchMulti crashes on None), inputs that are missing - e.g. a RefPicker
that found no character match and emitted its "no reference" None - are simply
skipped. If ALL inputs are missing, outputs None, which Generate treats as
"no reference wired": the item renders without identity seeding instead of
killing the queue.

Mixed sizes are resized (lanczos, center-crop semantics via common_upscale)
to the first present image's dimensions; Generate cover-fits every ref frame
to the video dimensions afterwards anyway.
"""

import torch

import comfy.utils


class JoyEcho_RefBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("reference_image",)
    FUNCTION = "batch"
    CATEGORY = "JoyAI-Echo"

    def batch(self, image_1=None, image_2=None, image_3=None, image_4=None):
        imgs = [i for i in (image_1, image_2, image_3, image_4)
                if i is not None and i.shape[0] > 0]
        if not imgs:
            print("[JoyEcho] RefBatch: no reference images present; "
                  "passing None (Generate skips identity seeding).", flush=True)
            return (None,)
        h, w = imgs[0].shape[1], imgs[0].shape[2]
        out = []
        for img in imgs:
            if img.shape[1] != h or img.shape[2] != w:
                img = comfy.utils.common_upscale(
                    img.movedim(-1, 1), w, h, "lanczos", "center").movedim(1, -1)
            out.append(img)
        return (torch.cat(out, dim=0),)


NODE_CLASS_MAPPINGS = {"JoyEcho_RefBatch": JoyEcho_RefBatch}
NODE_DISPLAY_NAME_MAPPINGS = {"JoyEcho_RefBatch": "JoyEcho Reference Batch (None-tolerant)"}
