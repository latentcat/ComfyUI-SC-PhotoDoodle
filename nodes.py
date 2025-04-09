import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
import comfy.model_management
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import logging


class PhotoDoodleSamplerAdvanced(SamplerCustomAdvanced):
    """
    扩展的自定义采样器，支持额外的条件图像和位置编码选项
    自动处理掩码和引导强度
    """

    @classmethod
    def INPUT_TYPES(s):
        # 继承原始输入类型并添加新的参数
        original_inputs = SamplerCustomAdvanced.INPUT_TYPES()

        # 添加新的必需参数，但移除不需要的参数
        original_inputs["required"].update(
            {
                "condition_image": ("LATENT", {"default": None}),
                "use_clone_pe": ("BOOLEAN", {"default": False}),
            }
        )

        # 不需要可选参数，使用内部默认值
        # original_inputs["optional"] = {}

        return original_inputs

    RETURN_TYPES = SamplerCustomAdvanced.RETURN_TYPES
    RETURN_NAMES = SamplerCustomAdvanced.RETURN_NAMES
    FUNCTION = "sample_with_condition"
    CATEGORY = "sampling/custom_sampling"

    def sample_with_condition(
        self,
        noise,
        guider,
        sampler,
        sigmas,
        latent_image,
        condition_image=None,
        use_clone_pe=False,
    ):
        """
        扩展的采样函数，支持条件图像和自定义位置编码
        自动处理掩码和引导强度
        """
        latent = latent_image
        latent_image = latent["samples"]
        latent = latent.copy()
        latent_image = comfy.sample.fix_empty_latent_channels(
            guider.model_patcher, latent_image
        )
        latent["samples"] = latent_image

        # 处理条件图像
        cond_latent = None
        if condition_image is not None:
            cond_latent = condition_image["samples"]

            # 确保条件图像在与潜在图像相同的设备上
            # 一次性移动到 GPU，避免每次推理都移动
            target_device = comfy.model_management.get_torch_device()
            if cond_latent.device != target_device:
                logging.info(
                    f"将条件图像从 {cond_latent.device} 移动到 {target_device}"
                )
                cond_latent = cond_latent.to(target_device)

            # 记录条件图像的形状和设备
            logging.debug(
                f"条件图像: shape={cond_latent.shape}, device={cond_latent.device}"
            )

        # 创建去噪掩码 - 只对非条件部分应用去噪
        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]
            logging.debug(f"使用噪声掩码: shape={noise_mask.shape}")

        # 准备回调
        x0_output = {}
        callback = latent_preview.prepare_callback(
            guider.model_patcher, sigmas.shape[-1] - 1, x0_output
        )

        # 确保 guider 有 model_options 属性
        if not hasattr(guider, "model_options"):
            guider.model_options = {}
            logging.info("为 guider 创建 model_options 属性")

        # 直接设置 model_options
        guider.model_options["condition_image"] = cond_latent
        guider.model_options["use_clone_pe"] = use_clone_pe
        guider.model_options["guider_id"] = id(guider)

        # 记录设置的选项
        logging.debug(
            f"设置 guider.model_options: condition_image={cond_latent is not None}, use_clone_pe={use_clone_pe}"
        )

        # 准备额外的关键字参数 - 同时通过 kwargs 和 model_options 传递
        extra_kwargs = {
            "condition_image": cond_latent,
            "use_clone_pe": use_clone_pe,
        }

        # 生成噪声张量
        noise_tensor = noise.generate_noise(latent)

        # 调用采样器
        samples = guider.sample(
            noise_tensor,
            latent_image,
            sampler,
            sigmas,
            denoise_mask=noise_mask,
            callback=callback,
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
            seed=noise.seed,
            **extra_kwargs,
        )

        latent["samples"] = samples
        return (latent, x0_output.get("x0", None))


class PhotoDoodleCrop:
    """
    图片裁切节点：在保持原图比例的情况下，尽可能最大化地裁切出目标宽高的区域
    如果原图尺寸不足，则放大至目标尺寸，保持比例且不留白
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop_image"
    CATEGORY = "image/processing"

    def crop_image(self, image, width, height):
        """
        裁切图片，保持比例，最大化填充目标区域，无留白

        参数:
            image: 输入图片张量 [B, H, W, C]
            width: 目标宽度
            height: 目标高度

        返回:
            裁切后的图片张量 [B, height, width, C]
        """
        # 转换为numpy处理
        result = []
        for img in image:
            img_np = img.cpu().numpy()

            # 获取原始尺寸
            orig_h, orig_w = img_np.shape[0], img_np.shape[1]

            # 计算目标比例和原始比例
            target_ratio = width / height
            orig_ratio = orig_w / orig_h

            # 策略：先调整比例（缩小或裁切），再缩放到目标尺寸
            if orig_ratio > target_ratio:
                # 原图更宽，需要调整宽度
                new_w = int(orig_h * target_ratio)
                offset_w = (orig_w - new_w) // 2
                adjusted = img_np[:, offset_w : offset_w + new_w, :]
            else:
                # 原图更高，需要调整高度
                new_h = int(orig_w / target_ratio)
                offset_h = (orig_h - new_h) // 2
                adjusted = img_np[offset_h : offset_h + new_h, :]

            # 调整后的图像缩放到目标尺寸
            pil_img = Image.fromarray((adjusted * 255).astype(np.uint8))
            resized_pil = pil_img.resize((width, height), Image.LANCZOS)
            resized_np = np.array(resized_pil).astype(np.float32) / 255.0

            result.append(torch.from_numpy(resized_np))

        # 堆叠所有处理后的图片
        return (torch.stack(result),)


class PhotoDoodleParams:
    """
    参数传递节点：将输入的参数原样传递到输出
    用于在ComfyUI工作流中连接和保存常用参数
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("seed", "width", "height", "cfg", "prompt")
    FUNCTION = "pass_params"
    CATEGORY = "utils/parameters"

    def pass_params(self, seed: int, width: int, height: int, cfg: float, prompt: str):
        """
        将输入的参数原样传递到输出
        
        参数:
            seed: 随机种子
            width: 图像宽度
            height: 图像高度
            cfg: CFG引导强度
            prompt: 提示词文本
            
        返回:
            原样输出的参数元组
        """
        return (seed, width, height, cfg, prompt)
