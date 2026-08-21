torchrun --nproc_per_node 2 --master_port 29800 -m training.main --batch-size=2 --lr=1e-5 --wd=0.1 --epochs=6 --workers=4 \
--model EVA02-CLIP-B-16 --pretrained eva --warmup 1000  --zeroshot-frequency 1 --dataset-type proposals_distill  \
--test-type coco_panoptic --train-data data/coco/coco_proposals.json \
--val-data data/coco/annotations/panoptic_val2017.json \
--embed-path metadata/coco_panoptic_clip_hand_craft_EVACLIP_ViTB16.npy --train-image-root data/coco/train2017 \
--val-image-root data/coco/val2017  --cache-dir checkpoints/EVA02_CLIP_B_psz16_s8B.pt --log-every-n-steps 50 \
--lock-image --save-frequency 1 --lock-image-unlocked-groups 12 --extract-type="v2" \
--name logs_dino_vit_base_16 --downsample-factor 16 --det-image-size 1024 \
--alpha 0.7 --train-embed-path metadata/coco_nouns_4764_clip_hand_craft_EVACLIP_ViTB16.npy --lsl-weight 0.1 --csl-weight 0.3 --vfm-type dino-B-16
