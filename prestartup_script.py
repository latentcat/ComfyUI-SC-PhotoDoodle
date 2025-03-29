# 导入原始模块
import comfy.ldm.flux.model as original_model
import torch
from torch import Tensor, nn
from einops import rearrange, repeat
import comfy.ldm.common_dit
import logging

# 保存原始类的引用
OriginalFlux = original_model.Flux

# 添加位置编码克隆函数
def prepare_latent_image_ids_2(height, width, device, dtype):
    """准备潜在图像ID，用于位置编码"""
    latent_image_ids = torch.zeros(height//2, width//2, 3, device=device, dtype=dtype)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height//2, device=device)[:, None]  # y坐标
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width//2, device=device)[None, :]   # x坐标
    return latent_image_ids

def position_encoding_clone(batch_size, original_height, original_width, device, dtype):
    """克隆版本的位置编码生成函数"""
    latent_image_ids = prepare_latent_image_ids_2(original_height, original_width, device, dtype)
    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape
    latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )
    cond_latent_image_ids = latent_image_ids
    latent_image_ids = torch.concat([latent_image_ids, cond_latent_image_ids], dim=-2)
    return latent_image_ids

# 定义您的替代类
class DoodleFlux(OriginalFlux):
    """
    增强版的 Flux 模型，添加了条件图像处理和自定义位置编码功能
    """
    def __init__(self, *args, **kwargs):
        # 调用原始初始化
        super().__init__(*args, **kwargs)
        logging.info("初始化增强版 Flux 模型，支持条件图像处理")
        
    def forward(self, x, timestep, context, y, guidance=None, control=None, transformer_options={}, **kwargs):
        bs, c, h, w = x.shape
        patch_size = self.patch_size
        x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch_size, patch_size))

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)

        h_len = ((h + (patch_size // 2)) // patch_size)
        w_len = ((w + (patch_size // 2)) // patch_size)
        
        # 支持条件图像处理
        condition_image = kwargs.get("condition_image", None)
        if condition_image is not None:
            # 处理条件图像
            cond_img = rearrange(condition_image, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)
            # 合并潜在表示
            img = torch.cat([img, cond_img], dim=1)
            
            # 创建掩码
            mask = torch.ones_like(img)
            mask[:, cond_img.shape[1]:, :] = 0
            kwargs["attention_mask"] = mask
        
        # 使用克隆版本的位置编码
        if kwargs.get("use_clone_pe", False):
            img_ids = position_encoding_clone(bs, h, w, x.device, x.dtype)
        else:
            # 原始位置编码方式
            img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
            img_ids[:, :, 1] = img_ids[:, :, 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype).unsqueeze(1)
            img_ids[:, :, 2] = img_ids[:, :, 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype).unsqueeze(0)
            img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

        txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)
        out = self.forward_orig(img, img_ids, context, txt_ids, timestep, y, guidance, control, transformer_options, attn_mask=kwargs.get("attention_mask", None))
        
        # 处理输出
        if condition_image is not None:
            # 只返回非条件部分
            out = out[:, :out.shape[1]-cond_img.shape[1], :]
            
        return rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)[:,:,:h,:w]

# 替换原始类
logging.info("正在用自定义 Flux 模型替换原始模型")
original_model.Flux = DoodleFlux

# 记录启动信息
logging.info("正在初始化 SC_PhotoDoodle 自定义采样器...")