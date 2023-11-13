# export vit_b encoder
python3 scripts/export_onnx_encoder_model.py \
    --sam_checkpoint /home/cvhub/workspace/resources/weights/sam/sam_hq_vit_b.pth \
    --output /home/cvhub/workspace/resources/weights/sam/SAM-HQ/sam_hq_vit_b_encoder.onnx \
    --model-type vit_b \
    --quantize-out /home/cvhub/workspace/resources/weights/sam/SAM-HQ/sam_hq_vit_b_encoder_quant.onnx \
    --use-preprocess

# export vit_b decoder
python scripts/export_onnx_model.py \
    --checkpoint /home/cvhub/workspace/resources/weights/sam/SAM-HQ/sam_hq_vit_b.pth \
    --model-type vit_b \
    --output /home/cvhub/workspace/resources/weights/sam/SAM-HQ/sam_hq_vit_b_decoder.onnx

# export vit_l encoder
python3 scripts/export_onnx_encoder_model.py \
    --sam_checkpoint /home/cvhub/workspace/resources/weights/sam/sam_hq_vit_l.pth \
    --output /home/cvhub/workspace/resources/weights/sam/SAM-HQ/sam_hq_vit_l_encoder.onnx \
    --model-type vit_l \
    --quantize-out /home/cvhub/workspace/resources/weights/sam/SAM-HQ/sam_hq_vit_l_encoder_quant.onnx \
    --use-preprocess

# export vit_l decoder
python scripts/export_onnx_model.py \
    --checkpoint /home/cvhub/workspace/resources/weights/sam/SAM-HQ/sam_hq_vit_l.pth \
    --model-type vit_l \
    --output /home/cvhub/workspace/resources/weights/sam/SAM-HQ/sam_hq_vit_l_decoder.onnx

# export vit_h encoder
python3 scripts/export_onnx_encoder_model.py \
    --sam_checkpoint /home/cvhub/workspace/resources/weights/sam/SAM-HQ/sam_hq_vit_h.pth \
    --output /home/cvhub/workspace/resources/weights/sam/SAM-HQ/sam_hq_vit_h_encoder.onnx \
    --model-type vit_h \
    --quantize-out /home/cvhub/workspace/resources/weights/sam/SAM-HQ/sam_hq_vit_h_encoder_quant.onnx \
    --use-preprocess

# export vit_h decoder
python scripts/export_onnx_model.py \
    --checkpoint /home/cvhub/workspace/resources/weights/sam/SAM-HQ/sam_hq_vit_h.pth \
    --model-type vit_h \
    --output /home/cvhub/workspace/resources/weights/sam/SAM-HQ/sam_hq_vit_h_decoder.onnx

# inference vit_b
python scripts/main.py \
    --encoder_model /home/cvhub/workspace/resources/weights/sam/SAM-HQ/sam_hq_vit_b_encoder.onnx \
    --decoder_model /home/cvhub/workspace/resources/weights/sam/SAM-HQ/sam_hq_vit_b_decoder.onnx \
    --img_path /home/cvhub/workspace/projects/python/sam/sam-hq/demo/input_imgs/example4.png

# inference vit_b_quant
python scripts/main.py \
    --encoder_model /home/cvhub/workspace/resources/weights/sam/SAM-HQ/sam_hq_vit_b_encoder_quant.onnx \
    --decoder_model /home/cvhub/workspace/resources/weights/sam/SAM-HQ/sam_hq_vit_b_decoder.onnx \
    --img_path /home/cvhub/workspace/projects/python/sam/sam-hq/demo/input_imgs/example4.png

# inference vit_l
python scripts/main.py \
    --encoder_model /home/cvhub/workspace/resources/weights/sam/SAM-HQ/sam_hq_vit_l_encoder.onnx \
    --decoder_model /home/cvhub/workspace/resources/weights/sam/SAM-HQ/sam_hq_vit_l_decoder.onnx \
    --img_path /home/cvhub/workspace/projects/python/sam/sam-hq/demo/input_imgs/example4.png

# inference vit_l_quant
python scripts/main.py \
    --encoder_model /home/cvhub/workspace/resources/weights/sam/SAM-HQ/sam_hq_vit_l_encoder_quant.onnx \
    --decoder_model /home/cvhub/workspace/resources/weights/sam/SAM-HQ/sam_hq_vit_l_decoder.onnx \
    --img_path /home/cvhub/workspace/projects/python/sam/sam-hq/demo/input_imgs/example4.png

# inference vit_h
python scripts/main.py \
    --encoder_model /home/cvhub/workspace/resources/weights/sam/SAM-HQ/sam_hq_vit_h_encoder.onnx \
    --decoder_model /home/cvhub/workspace/resources/weights/sam/SAM-HQ/sam_hq_vit_h_decoder.onnx \
    --img_path /home/cvhub/workspace/projects/python/sam/sam-hq/demo/input_imgs/example4.png

# inference vit_h_quant
python scripts/main.py \
    --encoder_model /home/cvhub/workspace/resources/weights/sam/SAM-HQ/sam_hq_vit_h_encoder_quant.onnx \
    --decoder_model /home/cvhub/workspace/resources/weights/sam/SAM-HQ/sam_hq_vit_h_decoder.onnx \
    --img_path /home/cvhub/workspace/projects/python/sam/sam-hq/demo/input_imgs/example4.png