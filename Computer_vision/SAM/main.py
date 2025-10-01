import detector, sam
import torch
import cv2
import random


if __name__ == "__main__" : 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_path = "/workspace/CV_study/Computer_vision/SAM/COCO-128-2/train/000000000025_jpg.rf.782fe78a513b7eeded6172306f4f502c.jpg"
    output_path = "result.png"

    detect_cfg_path = "./configs/detr_r50_8xb2-150e_coco.py"
    detect_ckpt_path = "./weights/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth"
    sam_model_type = "vit_b"
    sam_ckpt_path = "./weights/sam_vit_b_01ec64.pth"


    detect_model = detector.initialize(detect_cfg_path, detect_ckpt_path, device)
    sam_model = sam.initialize(sam_model_type, sam_ckpt_path)

    bboxes = detector.inference(input_path, detect_model)

    input_img = cv.imread(input_path)

    for bbox in bboxes : 
        random_color = [random.randint(0,255) for _ in range(3)]
        sam_mask = sam.inference(input_path, bbox, sam_model)
        input_img = sam.visualize(input_img, sam_mask, random_color)
    
    cv2.imwrite(output_path, imput_img)



 