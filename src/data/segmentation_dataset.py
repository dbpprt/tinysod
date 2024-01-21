import glob
import os
from typing import List, Tuple

from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(self, images_path: str, images_glob: str, masks_path: str, masks_glob: str):
        self.items: List[Tuple[str, str]] = []

        images = glob.glob(os.path.join(images_path, images_glob))
        masks = glob.glob(os.path.join(masks_path, masks_glob))

        # create a mapping dictionary base_name -> path, so we can easily lookup items
        base_name_path_mask_lookup_table = {os.path.basename(mask).rsplit(".", 1)[0]: mask for mask in masks}

        for image in images:
            base_name = os.path.basename(image)
            base_name = base_name.rsplit(".", 1)[0]
            mask = base_name_path_mask_lookup_table[base_name]

            # TODO: for now this is strict, but we could just skip missing masks.
            assert os.path.exists(mask)

            self.items.append((image, mask))

    def __getitem__(self, index):
        item = self.items[index]
        return item

        # image = Image.open(img_path).convert("RGB")

        # if metadata["rgba_masks"] is True:
        #     target = Image.open(target_path).convert("RGBA")
        #     target = np.asarray(target)
        #     target = target[:, :, 3]
        #     target = Image.fromarray(target).convert("L")
        # else:
        #     target = Image.open(target_path)

        # background, unknown, foreground = self.generate_trimap(np.asarray(target))
        # target = Image.merge("RGB", (target, Image.fromarray(unknown), Image.fromarray(foreground)))

        # return img, target, metadata
        pass

    def __len__(self):
        return len(self.items)
