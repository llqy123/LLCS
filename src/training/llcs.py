import random
import torch
import torch.nn.functional as F
from torch.nn.functional import normalize

class LLCS:
    def __call__(self, batch, model, vfm_model, noun_embeddings, dist_model, loss, device, cast_dtype, distributed, args):
        if distributed:
            model = model.module
            dist_model = dist_model.module
        images, normed_boxes, image_crops, proxy_image = batch       # note texts are not paired with images
        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        normed_boxes = normed_boxes.to(device=device, dtype=cast_dtype, non_blocking=True)
        image_crops = image_crops.to(device=device, dtype=cast_dtype, non_blocking=True)
        proxy_image = proxy_image.to(device=device, dtype=cast_dtype, non_blocking=True)
        if args.multiscale:
            cur_h, cur_w = images.shape[2:]
            assert cur_h == cur_w
            if cur_h == 1024:
                tar_sizes = [320, 640, 896, 1024]
            elif cur_h == 896:
                tar_sizes = [336, 448, 672, 896]
            else:
                raise NotImplementedError
            tar_size = random.choice(tar_sizes)
            images = F.interpolate(images, size=(tar_size, tar_size), mode='bilinear')

        rois_list = []
        crops_list = []
        for bboxes_per_image, crops_per_image in zip(normed_boxes, image_crops):
            valid = bboxes_per_image[:, -1] > 0.5
            rois_list.append(bboxes_per_image[valid, :4])
            crops_list.append(crops_per_image[valid])

        image_crops = torch.cat(crops_list)

        with torch.no_grad():
            teacher_crop_features = dist_model.encode_image(image_crops, normalize=False)
            if "dinov2" in args.vfm_type:
                vfm_features = vfm_model.get_intermediate_layers(proxy_image, reshape=True)[0]
                B, C, H, W = vfm_features.shape
                N = H * W
                vfm_maps = vfm_features.permute(0, 2, 3, 1).reshape(B, N, C)  # [B, N, C]
            elif "dino" in args.vfm_type:
                feat = vfm_model.get_intermediate_layers(proxy_image)[0]
                nb_im = feat.shape[0]
                patch_size = vfm_model.patch_embed.patch_size
                I, J = proxy_image[0].shape[-2] // patch_size, proxy_image[0].shape[-2] // patch_size
                vfm_maps = feat[:, 1:, :].reshape(nb_im, I, J, -1).permute(0, 3, 1, 2).flatten(2, 3).permute(0, 2, 1)
            elif "sam" in args.vfm_type:
                vfm_maps = vfm_model.image_encoder(proxy_image).flatten(2, 3).permute(0, 2, 1)

        student_roi_features, student_features = model.encode_pseudo_boxes(images, rois_list, normalize=False,
                                                         extract_type=args.extract_type)

        # Baseline
        normed_student_features = F.normalize(student_roi_features, dim=-1)
        normed_teacher_features = F.normalize(teacher_crop_features, dim=-1)
        loss_cosine = 1.0 - (normed_student_features *
                             normed_teacher_features).sum(-1).mean()

        # Local Semantic Learning
        noun_embeddings = noun_embeddings.to(student_roi_features.device)
        noun_embeddings = F.normalize(noun_embeddings, dim=-1)
        student_logits = 100 * torch.matmul(normed_student_features, noun_embeddings.T)
        teacher_logits = 100 * torch.matmul(normed_teacher_features, noun_embeddings.T)
        student_probs = F.log_softmax(student_logits, dim=-1)  # log P_s
        teacher_probs = F.softmax(teacher_logits, dim=-1)  # P_t
        kl_loss = F.kl_div(student_probs, teacher_probs, reduction="batchmean")

        # Consistent Semantic Learning
        student_maps = student_features  # [B, N, C]

        vfm_norm = 10 * F.normalize(vfm_maps, dim=-1)  # teacher patch tokens
        student_norm = 10 * F.normalize(student_maps, dim=-1)  # student patch tokens

        vfm_self_cor = torch.matmul(vfm_norm, vfm_norm.transpose(-1, -2))  # [B, N, N]
        student_self_cor = torch.matmul(student_norm, student_norm.transpose(-1, -2))  # [B, N, N]
        vfm_loss = F.mse_loss(vfm_self_cor, student_self_cor)

        losses = dict(loss_cosine=loss_cosine*args.cosine_weight,
                      loss_kl=kl_loss*args.lsl_weight,
                      loss_vfm=vfm_loss*args.csl_weight,
                      )

        return losses, len(images), model.logit_scale.exp()
