


from typing import Any, Dict, List, Tuple
import cv2
from peekingduck.pipeline.nodes.abstract_node import AbstractNode

YELLOW = (0, 255, 255)        # in BGR format, per opencv's convention


def map_bbox_to_image_coords(
   bbox: List[float], image_size: Tuple[int, int]
) -> List[int]:

   width, height = image_size[0], image_size[1]
   x1, y1, x2, y2 = bbox
   x1 *= width
   x2 *= width
   y1 *= height
   y2 *= height
   return int(x1), int(y1), int(x2), int(y2)


class Node(AbstractNode):

   def init(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
      super().init(config, node_path=name, **kwargs)

   def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
      img = inputs["img"]
      bboxes = inputs["bboxes"]
      scores = inputs["bbox_scores"]
      img_size = (img.shape[1], img.shape[0])  # width, height

      for i, bbox in enumerate(bboxes):
         x1, y1, x2, y2 = map_bbox_to_image_coords(bbox, img_size)
         score = scores[i]
         score_str = f"{score:0.2f}"
         cv2.putText(
            img=img,
            text=score_str,
            org=(x1, y2),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,
            color=YELLOW,
            thickness=3,
         )

      return {}
