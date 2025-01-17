import os
import supervision as sv
import cv2 as cv
from groundingdino.util.inference import load_model, load_image, predict, annotate
import torch  # Ensure you have imported torch for tensor handling


CONFIG_PATH = "/home/admins/Downloads/semi_human_ws/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECK_POINT_PATH = "/home/admins/Downloads/semi_human_ws/GroundingDINO/weights/groundingdino_swint_ogc.pth"
my_model = load_model(CONFIG_PATH, CHECK_POINT_PATH)

# Fine-tuning parameters
TEXT_PROMPT = 'pallet under box'
BOX_THRESHOLD = 0.15  # Lowered for more object detection
TEXT_THRESHOLD = 0.25  # Lowered for more object detection

# Fine-tuning: Enable multiple prompts and ensemble predictions
ADDITIONAL_PROMPTS = ['box on pallet', 'stacked pallets', 'box near pallet']
ENSEMBLE_THRESHOLD = 0.5  # Confidence threshold for ensembling results

IMAGE_DIR_PATH = '/home/admins/Pallets'
ANNOTATED_SAVE_PATH = '/home/admins/Annotated_Pallets/'

os.makedirs(ANNOTATED_SAVE_PATH, exist_ok=True)

# Iterate through each image in the folder and annotate
for image_name in os.listdir(IMAGE_DIR_PATH):
    image_path = os.path.join(IMAGE_DIR_PATH, image_name)
    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    try:
        image_source1, myimage = load_image(image_path)
        all_boxes, all_scores, all_names = [], [], []

        # Predict using multiple prompts
        for prompt in [TEXT_PROMPT] + ADDITIONAL_PROMPTS:
            boxes, scores, names = predict(
                model=my_model,
                image=myimage,
                caption=prompt,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD
            )

            # Convert tensors to lists if needed
            if isinstance(boxes, torch.Tensor):  # Check if 'boxes' is a tensor
                boxes = boxes.cpu().numpy()
            if isinstance(scores, torch.Tensor):  # Check if 'scores' is a tensor
                scores = scores.cpu().numpy()
            if isinstance(names, torch.Tensor):  # Check if 'names' is a tensor
                names = names.cpu().numpy()

            all_boxes.extend(boxes)
            all_scores.extend(scores)
            all_names.extend(names)

        # Ensemble filtering: Keep only boxes with high confidence
        filtered_boxes, filtered_scores, filtered_names = [], [], []
        for i, score in enumerate(all_scores):
            if isinstance(score, torch.Tensor):  # If score is a tensor, convert to scalar
                score = score.item()
            if score >= ENSEMBLE_THRESHOLD:
                filtered_boxes.append(all_boxes[i])
                filtered_scores.append(score)
                filtered_names.append(all_names[i])

        # Annotate image with the filtered results
        annotate_image = annotate(
            image_source=image_source1,
            boxes=filtered_boxes,
            logits=filtered_scores,
            phrases=filtered_names
        )

        save_path = os.path.join(ANNOTATED_SAVE_PATH, f"annotated_{image_name}")
        cv.imwrite(save_path, annotate_image)
        print(f"Annotated image saved: {save_path}")

    except Exception as e:
        print(f"Error processing {image_name}: {e}")
