from torch.utils.data import Dataset

from PIL import Image
import json
import numpy as np
import os
import os.path
import random
from copy import deepcopy
from typing import Any, Callable, cast, Dict, List, Optional, Tuple


def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset structured as follows:
    .. code::
        directory/
        Â©ÃÂ©Â¤Â©Â¤ class_x
        Â©Â¦   Â©ÃÂ©Â¤Â©Â¤ xxx.ext
        Â©Â¦   Â©ÃÂ©Â¤Â©Â¤ xxy.ext
        Â©Â¦   Â©Â¸Â©Â¤Â©Â¤ ...
        Â©Â¦       Â©Â¸Â©Â¤Â©Â¤ xxz.ext
        Â©Â¸Â©Â¤Â©Â¤ class_y
            Â©ÃÂ©Â¤Â©Â¤ 123.ext
            Â©ÃÂ©Â¤Â©Â¤ nsdf3.ext
            Â©Â¸Â©Â¤Â©Â¤ ...
                Â©Â¸Â©Â¤Â©Â¤ asd932_.ext
    Args:
        directory (str): Root directory path.
    Raises:
        FileNotFoundError: If ``directory`` has no class folders.
    Returns:
        (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def make_dataset(
    directory: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).
    Args:
        directory (str): root dataset directory
        class_to_idx (Optional[Dict[str, int]]): Dictionary mapping class name to class index. If omitted, is generated
            by :func:`find_classes`.
        extensions (optional): A list of allowed extensions.
            Either extensions or is_valid_file should be passed. Defaults to None.
        is_valid_file (optional): A function that takes path of a file
            and checks if the file is a valid file
            (used to check of corrupt files) both extensions and
            is_valid_file should not be passed. Defaults to None.
    Raises:
        ValueError: In case ``class_to_idx`` is empty.
        ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
        FileNotFoundError: In case no valid file was found for any class.
    Returns:
        List[Tuple[str, int]]: samples of a form (path_to_sample, class)
    """
    directory = os.path.expanduser(directory)

    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances = []
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances

def convert_index_to_list(index):
    #print(index)
    index = index[0]
    return list(zip(index[:,0], index[:,1]))


def merge_crop_patches(img_A, img_B, FG_index_A, FG_index_B, feat_size, patch_size):
    new_img = np.zeros((feat_size * patch_size, feat_size * patch_size, 3), dtype=np.uint8)
    origin_shape = img_A.size

    input_size = 224

    #print(origin_shape)

    img_A = img_A.resize((input_size, input_size))
    img_B = img_B.resize((input_size, input_size))

    img_A = np.asarray(img_A)
    img_B = np.asarray(img_B)

    postive_patches_B = []
    for i in range(feat_size):
        for j in range(feat_size):
            temp = img_B[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            if (i, j) in FG_index_B:
                postive_patches_B.append(temp)

    m = np.random.permutation(len(postive_patches_B))
    postive_patches_B = np.array(postive_patches_B)[m]

    cnt = 0
    for i in range(feat_size):
        for j in range(feat_size):
            if (i, j) in FG_index_A:
                new_img[i * patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = img_A[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            else:
                new_img[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = postive_patches_B[cnt]
                cnt += 1

    #print(new_img.shape)


    #print(new_img.shape)

    #new_img = Image.new('RGB', (patch_size, patch_size*feat_size*feat_size), 255)

    #for i, patch in enumerate(selected_patches):
    #    new_img.paste(Image.fromarray(patch), (0, patch_size*i))

    #new_img = new_img.resize(feat_size*patch_size, feat_size*patch_size)
    new_img = Image.fromarray(new_img)

    new_img = new_img.resize(origin_shape)
    return new_img



class DatasetFolder(Dataset):
    """A generic data loader where the samples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/[...]/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/[...]/asd932_.ext
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.
     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
            self,
            root: str,
            mask_file: str,
            bg_prob: float,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super(DatasetFolder, self).__init__()

        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.bg_prob = bg_prob

        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)

        self.loader = loader
        self.extensions = extensions


        with open(mask_file, 'r') as f:
            t = json.load(f)
            self.mask = t
            #keys = list(t.keys())
            #mask = np.array(t[keys[0]])
            #print(mask.shape)


        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)

    def find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        """Same as :func:`find_classes`.
        This method can be overridden to only consider
        a subset of classes, or to adapt to a different dataset directory structure.
        """
        return find_classes(dir)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        input_size = 224

        path, target = self.samples[index]
        img_A = self.loader(path)
        key_A = '/'.join(path.split('/')[-2:])
        mask_A = np.array(self.mask[key_A])


        #print(len(FG_index_A), len(BG_index_A), len(FG_index_B), len(BG_index_B))
        feat_size = mask_A.shape[0]
        patch_size = input_size // mask_A.shape[0] #should be 32
        #print('feat size, patch size', feat_size, patch_size)

        index_A = np.dstack(np.unravel_index(np.argsort(-mask_A.ravel()), (feat_size, feat_size)))
        index_A = convert_index_to_list(index_A)
        FG_index_A = index_A[:8]
        BG_index_A = index_A[8:]

        temp1 = random.uniform(0,1)
        temp2 = random.uniform(0,1)

        #print(temp)
        if temp1<self.bg_prob:
            all_index = list(range(len(self.samples)))
            index_B = random.choice(all_index)

            path_B, _ = self.samples[index_B]
            img_B = self.loader(path_B)
            key_B = '/'.join(path_B.split('/')[-2:])
            mask_B = np.array(self.mask[key_B])
            index_B = np.dstack(np.unravel_index(np.argsort(-mask_B.ravel()), (feat_size, feat_size)))
            index_B = convert_index_to_list(index_B)
            FG_index_B = index_B[:8]
            BG_index_B = index_B[8:]

            temp_img_A = deepcopy(img_A)
            img1 = merge_crop_patches(temp_img_A, img_B, FG_index_A, BG_index_B, feat_size, patch_size)
        else:
            img1 = img_A
        '''
        if temp2<self.bg_prob:
            all_index = list(range(len(self.samples)))
            index_B = random.choice(all_index)

            path_B, _ = self.samples[index_B]
            img_B = self.loader(path_B)
            key_B = '/'.join(path_B.split('/')[-2:])
            mask_B = np.array(self.mask[key_B])
            index_B = np.dstack(np.unravel_index(np.argsort(-mask_B.ravel()), (feat_size, feat_size)))
            index_B = convert_index_to_list(index_B)
            FG_index_B = index_B[:8]
            BG_index_B = index_B[8:]

            temp_img_A = deepcopy(img_A)
            img2 = merge_crop_patches(temp_img_A, img_B, FG_index_A, BG_index_B, feat_size, patch_size)
        else:
            img2 = img_A
        '''

        img2 = img_A

        if self.transform is not None:
            #sample = self.transform(sample)
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            #neg_A_neg_B = self.transform(neg_A_neg_B)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return [img1, img2], target

    def __len__(self) -> int:
        return len(self.samples)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
            self,
            root: str,
            mask_file: str,
            bg_prob: float,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(ImageFolder, self).__init__(root, mask_file, bg_prob, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples