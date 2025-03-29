# 导入原始模块
import comfy.ldm.flux.model as original_model
import torch
from torch import Tensor, nn
from einops import rearrange, repeat
import comfy.ldm.common_dit
import logging
import comfy.samplers
import gc

# 保存原始类的引用
OriginalFlux = original_model.Flux

# 添加位置编码克隆函数
def prepare_latent_image_ids_2(height, width, device, dtype):
    """准备潜在图像ID，用于位置编码"""
    latent_image_ids = torch.zeros(height//2, width//2, 3, device=device, dtype=dtype)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height//2, device=device)[:, None]  # y坐标
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width//2, device=device)[None, :]   # x坐标
    logging.debug(f"生成位置编码: shape={latent_image_ids.shape}, device={device}, dtype={dtype}")
    return latent_image_ids

def position_encoding_clone(batch_size, original_height, original_width, device, dtype):
    """克隆版本的位置编码生成函数，支持条件图像"""
    # 只在调试模式下记录详细信息
    if logging.getLogger().level <= logging.DEBUG:
        logging.debug(f"使用克隆位置编码: batch_size={batch_size}, height={original_height}, width={original_width}")
    
    # 生成基础位置编码
    latent_image_ids = prepare_latent_image_ids_2(original_height, original_width, device, dtype)
    
    # 重塑为一维序列
    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape
    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )
    
    # 为条件图像创建相同的位置编码
    cond_latent_image_ids = latent_image_ids.clone()
    
    # 合并位置编码
    latent_image_ids = torch.concat([latent_image_ids, cond_latent_image_ids], dim=0)
    
    # 扩展到批次维度
    latent_image_ids = repeat(latent_image_ids, "hw c -> b hw c", b=batch_size)
    
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
        # 添加缓存字典
        self.position_encoding_cache = {}
        
    def forward(self, x, timestep, context, y, guidance=None, control=None, transformer_options={}, **kwargs):
        bs, c, h, w = x.shape
        patch_size = self.patch_size
        x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch_size, patch_size))

        # 只在调试模式下记录输入参数
        if logging.getLogger().level <= logging.DEBUG:
            logging.debug(f"DoodleFlux.forward 输入: shape={x.shape}, timestep={timestep.shape}, context={context.shape}")
        
        # 从多个可能的来源获取参数
        model_options = {}
        
        # 1. 从 transformer_options 中获取
        if "model_options" in transformer_options:
            model_options.update(transformer_options["model_options"])
        
        # 2. 从 kwargs 中获取
        for key in ["condition_image", "use_clone_pe", "denoise_mask"]:
            if key in kwargs:
                model_options[key] = kwargs[key]
        
        # 3. 从 guider 实例中获取 (如果在 transformer_options 中)
        if "guider" in transformer_options:
            guider = transformer_options["guider"]
            if hasattr(guider, "model_options"):
                model_options.update(guider.model_options)
        
        # 4. 从全局上下文中获取 - 使用更安全的方式避免警告
        try:
            for guider in [g for g in gc.get_objects() if isinstance(g, comfy.samplers.CFGGuider)]:
                if hasattr(guider, "model_options"):
                    model_options.update(guider.model_options)
        except Exception as e:
            logging.warning(f"获取全局 CFGGuider 时出错: {e}")
        
        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)

        h_len = ((h + (patch_size // 2)) // patch_size)
        w_len = ((w + (patch_size // 2)) // patch_size)
        
        # 支持条件图像处理
        condition_image = model_options.get("condition_image", None)
        if condition_image is None:
            # 尝试从 transformer_options 直接获取
            condition_image = transformer_options.get("condition_image", None)
        
        # 获取潜在掩码
        denoise_mask = model_options.get("denoise_mask", None)
        if denoise_mask is None:
            # 尝试从 transformer_options 直接获取
            denoise_mask = transformer_options.get("denoise_mask", None)
        
        # 如果有 denoise_mask，计算 latent_mask
        latent_mask = None
        if denoise_mask is not None:
            # 计算 latent_mask 为 denoise_mask 的补集
            latent_mask = 1. - denoise_mask
            
        # 使用克隆版本的位置编码
        use_clone_pe = model_options.get("use_clone_pe", False)
        
        # 使用缓存机制生成位置编码
        cache_key = (bs, h, w, use_clone_pe, x.device, x.dtype)
        if cache_key in self.position_encoding_cache:
            # 使用缓存的位置编码
            img_ids = self.position_encoding_cache[cache_key]
        else:
            # 生成新的位置编码并缓存
            if use_clone_pe:
                img_ids = position_encoding_clone(bs, h, w, x.device, x.dtype)
                logging.info(f"生成新的克隆位置编码: shape={img_ids.shape}")
            else:
                # 原始位置编码方式
                img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
                img_ids[:, :, 1] = img_ids[:, :, 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype).unsqueeze(1)
                img_ids[:, :, 2] = img_ids[:, :, 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype).unsqueeze(0)
                img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
                logging.info(f"生成新的原始位置编码: shape={img_ids.shape}")
            
            # 缓存位置编码
            self.position_encoding_cache[cache_key] = img_ids
        
        # 现在处理条件图像
        if condition_image is not None:
            # 处理条件图像
            cond_img = rearrange(condition_image, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)
            
            # 合并潜在表示
            img = torch.cat([img, cond_img], dim=1)
            
            # 如果使用克隆位置编码，我们需要确保位置编码与潜在表示匹配
            if not use_clone_pe:
                # 创建条件图像的位置编码
                cond_img_ids = img_ids.clone()
                # 合并位置编码
                img_ids = torch.cat([img_ids, cond_img_ids], dim=1)
        
        # 将 denoise_mask 和 latent_mask 都添加到 transformer_options
        if denoise_mask is not None:
            transformer_options["denoise_mask"] = denoise_mask
        if latent_mask is not None:
            transformer_options["latent_mask"] = latent_mask
        
        txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)
        
        # 确保 img_ids 和 txt_ids 具有相同的维度
        if img_ids.ndim != txt_ids.ndim:
            if img_ids.ndim == 2 and txt_ids.ndim == 3:
                # 如果 img_ids 是 2D，而 txt_ids 是 3D，添加一个维度
                img_ids = img_ids.unsqueeze(0).repeat(bs, 1, 1)
        
        out = self.forward_orig(img, img_ids, context, txt_ids, timestep, y, guidance, control, transformer_options, attn_mask=kwargs.get("attention_mask", None))
        
        # 处理输出
        if condition_image is not None:
            # 只返回非条件部分
            out = out[:, :out.shape[1]-cond_img.shape[1], :]
            
        result = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)[:,:,:h,:w]
        return result

# 替换原始类
logging.info("正在用自定义 Flux 模型替换原始模型")
original_model.Flux = DoodleFlux

# 记录启动信息
logging.info("正在初始化 SC_PhotoDoodle 自定义采样器...")

# 添加对 CFGGuider 的检查
def check_cfg_guider():
    """检查 CFGGuider 类的结构和方法"""
    try:
        # 检查 CFGGuider 类的结构
        import inspect
        
        # 获取 CFGGuider 类的所有方法和属性
        methods = [method for method in dir(comfy.samplers.CFGGuider) if not method.startswith('_')]
        logging.info(f"CFGGuider 类的方法: {methods}")
        
        # 检查 sample 方法
        if hasattr(comfy.samplers.CFGGuider, 'sample'):
            sample_sig = inspect.signature(comfy.samplers.CFGGuider.sample)
            sample_params = list(sample_sig.parameters.keys())
            logging.info(f"CFGGuider.sample 参数: {sample_params}")
            
            # 检查是否接受额外参数
            if '**kwargs' in str(sample_sig) or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sample_sig.parameters.values()):
                logging.info("CFGGuider.sample 支持可变关键字参数 (**kwargs)")
            else:
                logging.warning("CFGGuider.sample 不支持可变关键字参数，这可能导致额外参数被拒绝")
        
        # 检查继承关系
        mro = comfy.samplers.CFGGuider.__mro__
        logging.info(f"CFGGuider 的继承关系: {[cls.__name__ for cls in mro]}")
        
    except Exception as e:
        logging.error(f"检查 CFGGuider 时出错: {e}")

# 执行检查
check_cfg_guider()

# 直接修补 CFGGuider.sample 方法以支持额外参数
try:
    original_sample = comfy.samplers.CFGGuider.sample
    
    def patched_sample(self, noise, latent, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None, **extra_kwargs):
        """
        增强版的 CFGGuider.sample 方法，支持额外参数
        """
        # 只在首次调用时记录详细信息
        if not hasattr(self, '_sample_called'):
            logging.info(f"CFGGuider.sample 首次被调用，额外参数: {list(extra_kwargs.keys())}")
            self._sample_called = True
        
        # 保存额外参数到模型选项
        if not hasattr(self, 'model_options'):
            self.model_options = {}
            
        # 将额外参数添加到模型选项
        for key, value in extra_kwargs.items():
            self.model_options[key] = value
            
        # 特殊处理 denoise_mask 参数
        if "denoise_mask" in extra_kwargs and extra_kwargs["denoise_mask"] is not None:
            denoise_mask = extra_kwargs["denoise_mask"]
            # 如果需要，可以在这里添加掩码处理逻辑
            
        # 调用原始方法，但不传递额外参数
        args = [noise, latent, sampler, sigmas]
        kwargs = {'denoise_mask': denoise_mask, 'callback': callback, 'disable_pbar': disable_pbar, 'seed': seed}
        return original_sample(self, *args, **kwargs)
    
    # 应用补丁
    comfy.samplers.CFGGuider.sample = patched_sample
    logging.info("已成功修补 CFGGuider.sample 方法以支持额外参数")
    
except Exception as e:
    logging.error(f"修补 CFGGuider.sample 方法时出错: {e}")
