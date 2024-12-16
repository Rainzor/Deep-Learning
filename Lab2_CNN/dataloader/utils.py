import cv2
import torch

def load_image_multiprocess(args):
    """Single image loading function for use with multiprocessing."""
    image_path, label = args
    image = cv2.imread(image_path) # BGR
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    return image, label

def apply_transform_multiprocess(args):
    """
    Single image transformation function for use with multiprocessing.
    """
    img, transform = args
    img_pil = Image.fromarray(img)  # 转换为 PIL Image
    return transform(img_pil)
