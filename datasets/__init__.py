from torchvision import transforms
from datasets import video_transforms
from .col_datasets import Colonoscopic
from .col_image_datasets import ColonoscopicImages
from .kva_datasets import Kvasir_Capsule
from .cho_image_datasets import CholecT45Images
from .kva_image_datasets import Kvasir_CapsuleImages

def get_dataset(args):
    temporal_sample = video_transforms.TemporalRandomCrop(args.num_frames * args.frame_interval)

    if args.dataset == "col":
        transform_col = transforms.Compose([
                    video_transforms.ToTensorVideo(),
                    video_transforms.CenterCropResizeVideo(args.image_size),
                    # video_transforms.RandomHorizontalFlipVideo(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
        return Colonoscopic(args, transform=transform_col, temporal_sample=temporal_sample)
    elif args.dataset == 'col_img':
        transform_ffs = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            video_transforms.UCFCenterCropVideo(args.image_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        return ColonoscopicImages(args, transform=transform_ffs, temporal_sample=temporal_sample)
    
    elif args.dataset == "kva":
        transform_col = transforms.Compose([
                    video_transforms.ToTensorVideo(),
                    video_transforms.CenterCropResizeVideo(args.image_size),
                    # video_transforms.RandomHorizontalFlipVideo(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
        return Kvasir_Capsule(args, transform=transform_col, temporal_sample=temporal_sample)
    elif args.dataset == 'kva_img':
        transform_ffs = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            video_transforms.UCFCenterCropVideo(args.image_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        return Kvasir_CapsuleImages(args, transform=transform_ffs, temporal_sample=temporal_sample)
    elif args.dataset == 'cho_img':
        transform_ffs = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            video_transforms.UCFCenterCropVideo(args.image_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        return CholecT45Images(args, transform=transform_ffs, temporal_sample=temporal_sample)
    
    else:
        raise NotImplementedError(args.dataset)