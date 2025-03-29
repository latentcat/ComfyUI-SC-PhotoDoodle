import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
import comfy.model_management
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced

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
        original_inputs["required"].update({
            "condition_image": ("LATENT", {"default": None}),
            "use_clone_pe": ("BOOLEAN", {"default": False}),
        })
        
        # 不需要可选参数，使用内部默认值
        # original_inputs["optional"] = {}
        
        return original_inputs

    RETURN_TYPES = SamplerCustomAdvanced.RETURN_TYPES
    RETURN_NAMES = SamplerCustomAdvanced.RETURN_NAMES
    FUNCTION = "sample_with_condition"
    CATEGORY = "sampling/custom_sampling"

    def sample_with_condition(self, noise, guider, sampler, sigmas, latent_image, 
                             condition_image=None, use_clone_pe=False):
        """
        扩展的采样函数，支持条件图像和自定义位置编码
        自动处理掩码和引导强度
        """
        latent = latent_image
        latent_image = latent["samples"]
        latent = latent.copy()
        latent_image = comfy.sample.fix_empty_latent_channels(guider.model_patcher, latent_image)
        latent["samples"] = latent_image

        # 处理条件图像
        cond_latent = None
        if condition_image is not None:
            cond_latent = condition_image["samples"]
        
        # 处理噪声掩码 - 使用默认值
        # 如果用户提供了掩码，我们仍然会使用它
        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]
        
        # 准备回调
        x0_output = {}
        callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)

        # 准备额外的关键字参数 - 使用默认值
        extra_kwargs = {
            "condition_image": cond_latent,
            "use_clone_pe": use_clone_pe,
        }
        
        # 禁用进度条（如果需要）
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        
        # 调用采样器，传递额外参数
        samples = guider.sample(
            noise.generate_noise(latent), 
            latent_image, 
            sampler, 
            sigmas, 
            denoise_mask=noise_mask, 
            callback=callback, 
            disable_pbar=disable_pbar, 
            seed=noise.seed,
            **extra_kwargs
        )
        samples = samples.to(comfy.model_management.intermediate_device())

        # 准备输出
        out = latent.copy()
        out["samples"] = samples
        
        # 处理去噪输出
        if "x0" in x0_output:
            out_denoised = latent.copy()
            out_denoised["samples"] = guider.model_patcher.model.process_latent_out(x0_output["x0"].cpu())
        else:
            out_denoised = out
            
        return (out, out_denoised) 