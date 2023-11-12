import argparse
import os
import cv2
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt

from typing import Any, Union, Tuple
from copy import deepcopy


parser = argparse.ArgumentParser(
    description="Inference an image with onnxruntime backend."
)

parser.add_argument(
    "--encoder_model",
    type=str,
    required=True,
    help="Path to the SAM-Med2D onnx encoder model.",
)

parser.add_argument(
    "--decoder_model",
    type=str,
    required=True,
    help="Path to the SAM-Med2D onnx decoder model.",
)

parser.add_argument(
    "--img_path",
    type=str,
    default="../../data_demo/images/amos_0507_31.png",
    help="Path to the image",
)

parser.add_argument(
    "--input_size", 
    type=int, 
    default=1024, 
    help="input_size"
)

parser.add_argument(
    "--work_dir", 
    type=str, 
    default="workdir", 
    help="work dir"
)

args = parser.parse_args()

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))


def show_res(masks, scores, input_point, input_label, input_box, image):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        if input_box is not None:
            box = input_box[i]
            show_box(box, plt.gca())
        if (input_point is not None) and (input_label is not None):
            show_points(input_point, input_label, plt.gca())

        print(f"Score: {score:.3f}")
        plt.axis('off')
        plt.show()


class SamEncoder:
    """Sam encoder model.

    In this class, encoder model will encoder the input image.

    Args:
        model_path (str): sam encoder onnx model path.
        device (str): Inference device, user can choose 'cuda' or 'cpu'. default to 'cuda'.
        warmup_epoch (int): Warmup, if set 0,the model won`t use random inputs to warmup. default to 3.
    """

    def __init__(self,
                 model_path: str,
                 device: str = "cuda",
                 **kwargs):
        opt = ort.SessionOptions()

        if device == "cuda":
            provider = ['CUDAExecutionProvider']
        elif device == "cpu":
            provider = ['CPUExecutionProvider']
        else:
            raise ValueError("Invalid device, please use 'cuda' or 'cpu' device.")

        print("loading encoder model...")
        self.session = ort.InferenceSession(model_path,
                                            opt,
                                            providers=provider,
                                            **kwargs)

        self.target_length = 1024
        self.input_name = self.session.get_inputs()[0].name

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (neww, newh)

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(
            image.shape[0], image.shape[1], self.target_length
        )
        return cv2.resize(image, target_size)

    def transform(self, img: np.ndarray) -> np.ndarray:
        """image transform

        This function can convert the input image to \
            the required input format for onnx-encoder.

        Args:
            img (np.ndarray): input image, the image type should be BGR.

        Returns:
            np.ndarray: transformed image.
        """
        # Resize
        input_image = self.apply_image(img)
        return input_image.astype(np.float32)

    def _extract_feature(self, tensor: np.ndarray) -> np.ndarray:
        """extract image feature

        this function can use encoder to extract feature from transformed image.

        Args:
            tensor (np.ndarray): input image with BGR format.

        Returns:
            np.ndarray: image`s feature.
        """
        input_image = self.transform(tensor)
        features = self.session.run(None, {self.input_name: input_image})
        image_embeddings, interm_embeddings = features[0], np.stack(features[1:])
        return image_embeddings, interm_embeddings

    def __call__(self, img: np.array, *args: Any, **kwds: Any) -> Any:
        return self._extract_feature(img)

class SamDecoder:
    """Sam decoder model.

    This class is the sam prompt encoder and lightweight mask decoder.

    Args:
        model_path (str): decoder model path.
        device (str): Inference device, user can choose 'cuda' or 'cpu'. default to 'cuda'.
    """

    def __init__(self,
                 model_path: str,
                 device: str = "cuda",
                 img_size: int = 1024,
                 **kwargs):
        opt = ort.SessionOptions()

        if device == "cuda":
            provider = ['CUDAExecutionProvider']
        elif device == "cpu":
            provider = ['CPUExecutionProvider']
        else:
            raise ValueError("Invalid device, please use 'cuda' or 'cpu' device.")

        print("loading decoder model...")
        self.mask_threshold = 0.5
        self.target_length = img_size
        self.session = ort.InferenceSession(model_path,
                                            opt,
                                            providers=provider,
                                            **kwargs)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (neww, newh)

    def run(self,
            image_embeddings: np.ndarray,
            interm_embeddings: np.ndarray,
            origin_image_size: Union[list, tuple],
            point_coords: Union[list, np.ndarray] = None,
            point_labels: Union[list, np.ndarray] = None,
            boxes: Union[list, np.ndarray] = None,
            mask_input: np.ndarray = None):
        """decoder forward function

        This function can use image feature and prompt to generate mask. Must input
        at least one box or point.

        Args:
            image_embeddings (np.ndarray): the image feature from vit encoder.
            interm_embeddings (np.ndarray): the intermediate feature.
            origin_image_size (list or tuple): the input image size.
            point_coords (list or np.ndarray): the input points.
            point_labels (list or np.ndarray): the input points label, 1 indicates
                a foreground point and 0 indicates a background point.
            boxes (list or np.ndarray): A length 4 array given a box prompt to the
                model, in XYXY format.
            mask_input (np.ndarray): A low resolution mask input to the model,
                typically coming from a previous prediction iteration. Has form
                1xHxW, where for SAM, H=W=4 * embedding.size.

        Returns:
            the segment results.
        """
        if point_coords is None and point_labels is None and boxes is None:
            raise ValueError("Unable to segment, please input at least one box or point.")

        if image_embeddings.shape != (1, 256, 64, 64):
            raise ValueError("Got wrong image_embeddings shape!")

        if mask_input is None:
            mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
            has_mask_input = np.zeros(1, dtype=np.float32)
        else:
            has_mask_input = np.ones(1, dtype=np.float32)
            if mask_input.shape != (1, 1, 256, 256):
                raise ValueError("Got wrong mask!")

        if point_coords is not None:
            if isinstance(point_coords, list):
                point_coords = np.array(point_coords, dtype=np.float32)
            if isinstance(point_labels, list):
                point_labels = np.array(point_labels, dtype=np.float32)
            point_coords = np.concatenate([point_coords, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
            point_labels = np.concatenate([point_labels, np.array([-1])], axis=0)[None, :].astype(np.float32)
            
        if boxes is not None:
            if isinstance(boxes, list):
                boxes = np.array(boxes, dtype=np.float32)
            point_coords = boxes.reshape(2, 2)[None, :, :]
            point_labels = np.array([2, 3])[None, :].astype(np.float32)

        point_coords = self.apply_coords(point_coords, origin_image_size).astype(np.float32)
        assert point_coords.shape[0] == 1 and point_coords.shape[-1] == 2
        assert point_labels.shape[0] == 1

        input_dict = {"image_embeddings": image_embeddings,
                      "interm_embeddings": interm_embeddings,
                      "point_coords": point_coords,
                      "point_labels": point_labels,
                      "mask_input": mask_input,
                      "has_mask_input": has_mask_input,
                      "orig_im_size": np.array(origin_image_size, dtype=np.float32)}
        masks, iou_predictions, low_res_masks = self.session.run(None, input_dict)
        masks = masks > self.mask_threshold

        return masks, iou_predictions, low_res_masks

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes, original_size, new_size):
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size, new_size)
        return boxes.reshape(-1, 4)

def main():
    # Create save folder
    save_path = os.path.join(args.work_dir, 'ort_demo_results')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    base_name, file_extension = os.path.splitext(os.path.basename(args.img_path))

    # Initialize the SAM-Med2D onnx model
    encoder = SamEncoder(
        model_path=args.encoder_model,
        warmup_epoch=3
    )
    decoder = SamDecoder(
        model_path=args.decoder_model,
    )

    '''Specifying a specific object with a point'''
    img_file = cv2.imread(args.img_path)
    image = cv2.cvtColor(img_file, cv2.COLOR_BGR2RGB)
    image_embeddings, interm_embeddings = encoder(image)
    origin_image_size = image.shape[:2]
    point_coords = np.array([[500, 475]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.float32)
    masks, _, logits = decoder.run(
        image_embeddings=image_embeddings,
        interm_embeddings=interm_embeddings,
        origin_image_size=origin_image_size,
        point_coords=point_coords,
        point_labels=point_labels,
    )
    
    plt.figure(figsize=(10,10))
    plt.imshow(img_file)
    show_mask(masks, plt.gca())
    show_points(point_coords, point_labels, plt.gca())
    plt.axis('off')
    plt.savefig(os.path.join(save_path, base_name+'_point1'+file_extension))
    plt.show()  

    # '''Optimizing Segmentation Results by Point Interaction'''
    new_point_coords = np.array([[550, 500]], dtype=np.float32)
    new_point_labels = np.array([1], dtype=np.float32)
    point_coords = np.concatenate((point_coords, new_point_coords))
    point_labels = np.concatenate((point_labels, new_point_labels))
    mask_input = 1. / (1. + np.exp(-logits.astype(np.float32)))

    masks, _, logits = decoder.run(
        image_embeddings=image_embeddings,
        interm_embeddings=interm_embeddings,
        origin_image_size=origin_image_size,
        point_coords=point_coords,
        point_labels=point_labels,
        mask_input=mask_input,
    )

    plt.figure(figsize=(10,10))
    plt.imshow(img_file)
    show_mask(masks, plt.gca())
    show_points(point_coords, point_labels, plt.gca())
    plt.axis('off')
    plt.savefig(os.path.join(save_path, base_name+'_point2'+file_extension))
    plt.show()

    # '''Specifying a specific object with a bounding box'''
    boxes = np.array([64, 76, 940, 919])

    masks, _, _ = decoder.run(
        image_embeddings=image_embeddings,
        interm_embeddings=interm_embeddings,
        origin_image_size=origin_image_size,
        point_coords=None,
        point_labels=None,
        boxes=boxes
    )
    plt.figure(figsize=(10,10))
    plt.imshow(img_file)
    show_mask(masks, plt.gca())
    show_box(boxes, plt.gca())
    plt.axis('off')
    plt.savefig(os.path.join(save_path, base_name+'_box'+file_extension))
    plt.show()  
 

if __name__ == '__main__':
    main()