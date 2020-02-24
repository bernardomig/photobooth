import os
import cv2

from torch.utils.data import Dataset


class DIV2K(Dataset):
    """The Diverse2K dataset contains high quality 2K images.
    It was used for the NTIRE (CVPR '17 and '18) and PIRM (ECCV '18) challenges.

    Args:
      root_dir: the directory containing the dataset.
      split: one of `train` or `valid`.
      config: contains the configuration for the degraded image,
              such as `bicubic/x2` or `bicubic/x4`.
      transforms: a function to transform the lowres and highres image.
              Should have as input the `lowres` and `highres` parameters.
    """

    def __init__(self, root_dir, split='train', config='bicubic/x2', transforms=None):
        if split not in {'train', 'valid'}:
            raise ValueError(
                "split must be `train` or `valid`, got {}"
                .format(split))

        algo, scale = config.split('/')

        hr_dir = os.path.join(
            root_dir,
            'DIV2K_{split}_HR'
            .format(split=split))
        lr_dir = os.path.join(
            root_dir,
            'DIV2K_{split}_LR_{algo}/{scale}'
            .format(split=split, algo=algo, scale=scale.upper()))

        rang = range(0, 800) if split == 'train' else range(800, 900)

        self.examples = [
            {
                'highres': os.path.join(hr_dir, '{:04d}.png'.format(idx+1)),
                'lowres': os.path.join(lr_dir, '{:04d}{}.png'.format(idx+1, scale)),
            }
            for idx in rang
        ]

        self.transforms = transforms

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        highres = example['highres']
        highres = cv2.imread(highres)
        highres = cv2.cvtColor(highres, cv2.COLOR_BGR2RGB)

        lowres = example['lowres']
        lowres = cv2.imread(lowres)
        lowres = cv2.cvtColor(lowres, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            tfmd = self.transforms(highres=highres, lowres=lowres)
            return tfmd['lowres'], tfmd['highres']
        else:
            return lowres, highres
