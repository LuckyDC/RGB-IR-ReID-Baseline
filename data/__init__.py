import torchvision.transforms as T
from torch.utils.data import DataLoader

from data.datasets import SYSUDataset
from data.samplers import CrossModalityIdentitySampler
from data.samplers import CrossModalityRandomSampler


def get_train_loader(root, sample_method, batch_size, p_size, k_size, image_size, random_flip=False, random_crop=False,
                     random_erase=False, color_jitter=False, padding=0, num_workers=4):
    # data pre-processing
    t = [T.Resize(image_size)]

    if random_flip:
        t.append(T.RandomHorizontalFlip())

    if color_jitter:
        t.append(T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0))

    if random_crop:
        t.extend([T.Pad(padding, fill=127), T.RandomCrop(image_size)])

    t.extend([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if random_erase:
        t.append(T.RandomErasing())

    transform = T.Compose(t)

    # dataset
    train_dataset = SYSUDataset(root, mode='train', transform=transform)

    # sampler
    assert sample_method in ['random', 'identity_uniform']
    if sample_method == 'identity_uniform':
        sampler = CrossModalityIdentitySampler(train_dataset, p_size, k_size)
    else:
        sampler = CrossModalityRandomSampler(train_dataset, batch_size)

    # loader
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              sampler=sampler,
                              drop_last=True,
                              pin_memory=True,
                              num_workers=num_workers)

    return train_loader


def get_test_loader(root, batch_size, image_size, num_workers=4):
    # transform
    transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # dataset
    gallery_dataset = SYSUDataset(root, mode='gallery', transform=transform)
    query_dataset = SYSUDataset(root, mode='query', transform=transform)

    # dataloader
    query_loader = DataLoader(dataset=query_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              pin_memory=False,
                              drop_last=False,
                              num_workers=num_workers)

    gallery_loader = DataLoader(dataset=gallery_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                pin_memory=False,
                                drop_last=False,
                                num_workers=num_workers)

    return gallery_loader, query_loader
