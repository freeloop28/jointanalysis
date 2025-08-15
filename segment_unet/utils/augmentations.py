import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(split='train', target_size=(120, 120)):
    if split == 'train':
        transform = A.Compose([
            A.Resize(*target_size),
            A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5)
            ], p=0.7),

            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=50, approximate=True, p=0.5),
                A.GridDistortion(p=0.3),
                A.OpticalDistortion(p=0.3),
            ], p=0.5),

            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=20, border_mode=0, p=0.5),

            A.OneOf([
                A.RandomBrightnessContrast(p=0.5),
                A.CLAHE(p=0.3),
                A.RandomGamma(p=0.3),
            ], p=0.5),

            A.OneOf([
                A.GaussianBlur(blur_limit=3, p=0.3),
                A.MotionBlur(blur_limit=3, p=0.3),
                A.ISONoise(p=0.3),
            ], p=0.4),

            A.CoarseDropout(max_holes=4, max_height=20, max_width=20, fill_value=0, p=0.3),

            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])
    else:
        transform = A.Compose([
            A.Resize(*target_size),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])
    return transform
