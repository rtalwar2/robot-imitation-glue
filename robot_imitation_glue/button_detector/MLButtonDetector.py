from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Tuple

import cv2
import torch
import requests
import numpy as np
from PIL import Image
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline


@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]


@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> "DetectionResult":
        return cls(
            score=detection_dict["score"],
            label=detection_dict["label"],
            box=BoundingBox(
                xmin=detection_dict["box"]["xmin"],
                ymin=detection_dict["box"]["ymin"],
                xmax=detection_dict["box"]["xmax"],
                ymax=detection_dict["box"]["ymax"],
            ),
        )


class MLButtonDetector:

    def __init__(self, detector_id, segmenter_id):
        self.labels = ["red round button."]
        self.threshold = 0.3
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.object_detector = pipeline(
            model=detector_id,
            task="zero-shot-object-detection",
            device=device,
            local_files_only=True,
        )
        self.segmentator = AutoModelForMaskGeneration.from_pretrained(
            segmenter_id, local_files_only=True
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(
            segmenter_id, local_files_only=True
        )

    def is_red_region(self, crop, red_ratio_threshold=0.4):
        """
        Returns True if a significant portion of the crop is 'red'.
        """
        R = crop[:, :, 0].astype(float)
        G = crop[:, :, 1].astype(float)
        B = crop[:, :, 2].astype(float)

        # Pixels that satisfy R > G and R > B by a margin
        red_pixels = (R > 120) & (R > G + 40) & (R > B + 40)

        ratio = red_pixels.mean()
        return ratio > red_ratio_threshold

    def is_round(self, box: BoundingBox, roundness_threshold=0.7):
        w = box.xmax - box.xmin
        h = box.ymax - box.ymin
        roundness = min(w, h) / max(w, h)
        return roundness > roundness_threshold

    def filter_detections_by_shape_and_color(self, img, detections):
        filtered = []

        for det in detections:
            box = det.box
            crop = img[box.ymin : box.ymax, box.xmin : box.xmax]

            # Safety check
            if crop.size == 0:
                continue

            if self.is_red_region(crop) and self.is_round(det.box):
                filtered.append(det)

        return filtered

    def mask_to_polygon(self, mask: np.ndarray) -> List[List[int]]:
        # Find contours in the binary mask
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Find the contour with the largest area
        largest_contour = max(contours, key=cv2.contourArea)

        # Extract the vertices of the contour
        polygon = largest_contour.reshape(-1, 2).tolist()

        return polygon

    def polygon_to_mask(
        self, polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Convert a polygon to a segmentation mask.

        Args:
        - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
        - image_shape (tuple): Shape of the image (height, width) for the mask.

        Returns:
        - np.ndarray: Segmentation mask with the polygon filled.
        """
        # Create an empty mask
        mask = np.zeros(image_shape, dtype=np.uint8)

        # Convert polygon to an array of points
        pts = np.array(polygon, dtype=np.int32)

        # Fill the polygon with white color (255)
        cv2.fillPoly(mask, [pts], color=(255,))

        return mask

    def load_image(self, image_str: str) -> Image.Image:
        if image_str.startswith("http"):
            image = Image.open(requests.get(image_str, stream=True).raw).convert("RGB")
        else:
            image = Image.open(image_str).convert("RGB")

        return image

    def get_boxes(self, results: DetectionResult) -> List[List[List[float]]]:
        boxes = []
        for result in results:
            xyxy = result.box.xyxy
            boxes.append(xyxy)

        return [boxes]

    def refine_masks(
        self, masks: torch.BoolTensor, polygon_refinement: bool = False
    ) -> List[np.ndarray]:
        masks = masks.cpu().float()
        masks = masks.permute(0, 2, 3, 1)
        masks = masks.mean(axis=-1)
        masks = (masks > 0).int()
        masks = masks.numpy().astype(np.uint8)
        masks = list(masks)

        if polygon_refinement:
            for idx, mask in enumerate(masks):
                shape = mask.shape
                polygon = self.mask_to_polygon(mask)
                mask = self.polygon_to_mask(polygon, shape)
                masks[idx] = mask

        return masks

    def detect(
        self,
        image: Image.Image,
        labels: List[str],
        threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
        """

        labels = [label if label.endswith(".") else label + "." for label in labels]

        results = self.object_detector(
            image, candidate_labels=labels, threshold=threshold
        )
        results = [DetectionResult.from_dict(result) for result in results]

        return results

    def segment(
        self,
        image: Image.Image,
        detection_results: List[Dict[str, Any]],
        polygon_refinement: bool = False,
    ) -> List[DetectionResult]:
        """
        Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if len(detection_results) > 0:
            boxes = self.get_boxes(detection_results)
            inputs = self.processor(
                images=image, input_boxes=boxes, return_tensors="pt"
            ).to(device)

            outputs = self.segmentator(**inputs)
            masks = self.processor.post_process_masks(
                masks=outputs.pred_masks,
                original_sizes=inputs.original_sizes,
                reshaped_input_sizes=inputs.reshaped_input_sizes,
            )[0]

            masks = self.refine_masks(masks, polygon_refinement)

            for detection_result, mask in zip(detection_results, masks):
                detection_result.mask = mask

        return detection_results

    def grounded_segmentation(
        self,
        image: Union[Image.Image, str],
        polygon_refinement: bool = False,
    ) -> Tuple[np.ndarray, List[DetectionResult]]:
        if isinstance(image, str):
            image = self.load_image(image)

        detections = self.detect(image, self.labels, self.threshold)
        detections = self.segment(image, detections, polygon_refinement)

        return np.array(image), detections
