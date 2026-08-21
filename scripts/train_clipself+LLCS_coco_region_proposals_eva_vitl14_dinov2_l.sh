torchrun --nproc_per_node 2 --master_port 29700 -m training.main --batch-size=2 --lr=1e-5 --wd=0.1 --epochs=6 --workers=4 \
--model EVA02-CLIP-L-14-336 --pretrained eva --warmup 1000  --zeroshot-frequency 1 --dataset-type proposals_distill  \
--test-type coco_panoptic --train-data data/coco/coco_proposals.json \
--val-data data/coco/annotations/panoptic_val2017.json \
--embed-path metadata/coco_panoptic_clip_hand_craft_EVACLIP_ViTL14x336.npy --train-image-root data/coco/train2017 \
--val-image-root data/coco/val2017  --cache-dir checkpoints/EVA02_CLIP_L_336_psz14_s6B.pt --log-every-n-steps 50 \
--lock-image --save-frequency 6 --lock-image-unlocked-groups 24 --extract-type="v2" \
--name logs_dinov2_vit_large_14 --downsample-factor 14 --det-image-size 896 \
--alpha 0.95 --train-embed-path metadata/coco_nouns_4764_clip_hand_craft_EVACLIP_ViTL14x336.npy --lsl-weight 0.1 --csl-weight 0.3 --vfm-type dinov2-L
