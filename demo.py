from __future__ import annotations

import base64
import io
from typing import Optional
from api_keys import DASHSCOPE
import argparse
import re
import os
import random
from pathlib import Path
from typing import Iterable, List, Tuple
from openai import OpenAI
import tempfile
import subprocess
import threading


from PIL import Image

def center_crop_resize(img: Image.Image, target_w: int = 832, target_h: int = 480) -> Image.Image:
    """
    居中裁剪为指定比例和分辨率，不拉伸。
    """
    src_w, src_h = img.size
    target_ratio = target_w / target_h
    src_ratio = src_w / src_h
    # 先裁剪，再resize
    if src_ratio > target_ratio:
        # 原图太宽，裁掉两侧
        new_w = int(src_h * target_ratio)
        left = (src_w - new_w) // 2
        box = (left, 0, left + new_w, src_h)
    else:
        # 原图太高，裁掉上下
        new_h = int(src_w / target_ratio)
        top = (src_h - new_h) // 2
        box = (0, top, src_w, top + new_h)
    img_cropped = img.crop(box)
    return img_cropped.resize((target_w, target_h), Image.LANCZOS)


def image_to_data_url(image: Image.Image, format: Optional[str] = None) -> str:
    """
    将 PIL Image 转换为 Data URL 字符串。

    输出格式：
        data:[MIME_type];base64,{base64_image}

    规则：
    - MIME_type 自动根据图像格式推导，确保为标准值（如 image/png、image/jpeg）。
    - 如果未显式提供 format，则优先使用 image.format；若仍不可得，默认使用 PNG。
    - 当保存为 JPEG 时，会自动转换到 RGB 以避免 Pillow 的模式不兼容错误。

    参数:
        image: PIL.Image.Image 实例。
        format: 目标保存格式（例如 'PNG'、'JPEG'、'JPG'）。不区分大小写。

    返回:
        符合 data URL 规范的字符串，如 'data:image/png;base64,.....'

    异常:
        TypeError: 当传入的 image 不是 PIL.Image.Image。
        ValueError: 当无法确定或不支持的图像格式。
    """

    if not isinstance(image, Image.Image):
        raise TypeError("image 必须是 PIL.Image.Image 实例")

    # 归一化与推断格式
    fmt = (format or image.format or "PNG").upper()
    if fmt == "JPG":
        fmt = "JPEG"

    # 更稳健的 MIME 推断：优先使用 Pillow 自带映射，其次用本地映射，最后保底为 image/{fmt.lower()}
    _MIME_FALLBACK = {
        "PNG": "image/png",
        "JPEG": "image/jpeg",
        "WEBP": "image/webp",
        "GIF": "image/gif",
        "BMP": "image/bmp",
        "TIFF": "image/tiff",
        "ICO": "image/x-icon",
        "PPM": "image/x-portable-pixmap",
        "PNM": "image/x-portable-anymap",
        "TGA": "image/x-targa",
        "AVIF": "image/avif",
        # 其他格式如 HEIC/HEIF 是否有效取决于 Pillow 的编译支持
        "HEIC": "image/heic",
        "HEIF": "image/heif",
    }

    mime = (getattr(Image, "MIME", {}) or {}).get(fmt) or _MIME_FALLBACK.get(fmt)
    if not mime:
        # 退而求其次，构造一个通用的 MIME；多数浏览器仍能识别常见格式
        mime = f"image/{fmt.lower()}"

    # JPEG 不能带 alpha／部分模式不兼容，必要时转为 RGB
    to_save = image
    if fmt == "JPEG" and image.mode not in ("L", "RGB"):
        to_save = image.convert("RGB")

    # 编码为 base64
    buffer = io.BytesIO()
    save_kwargs = {}
    if fmt == "JPEG":
        save_kwargs.update({"quality": 95, "optimize": True})
    to_save.save(buffer, format=fmt, **save_kwargs)
    base64_str = base64.b64encode(buffer.getvalue()).decode("ascii")

    return f"data:{mime};base64,{base64_str}"


def _iter_mp4_files(root: Path) -> Iterable[Path]:
    """递归遍历 root 下的 .mp4 文件（不区分大小写）。"""
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(".mp4"):
                yield Path(dirpath) / name


def _read_first_frame_pil(video_path: Path) -> Image.Image:
    """读取视频第一帧并返回为 PIL.Image.Image。

    优先使用 imageio（RGB），失败则回退到 OpenCV（BGR->RGB）。
    """
    # 尝试使用 imageio
    try:
        import imageio.v2 as iio  # type: ignore
        reader = iio.get_reader(str(video_path))
        try:
            frame = reader.get_data(0)  # RGB ndarray
        finally:
            reader.close()
        return center_crop_resize(Image.fromarray(frame))
    except Exception:
        pass

    # 回退到 OpenCV
    try:
        import cv2  # type: ignore
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError("无法打开视频")
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError("无法读取第一帧")
        # BGR -> RGB
        frame = frame[:, :, ::-1]
        return center_crop_resize(Image.fromarray(frame))
    except Exception as e:
        raise RuntimeError(f"读取第一帧失败: {video_path} | {e}")


def sample_mp4_first_frames(
    root_dir: os.PathLike | str,
    k: int,
    title_pattern: str,
    seed: Optional[int] = None,
) -> List[Tuple[str, str, Image.Image]]:
    """给定目录，随机采样 k 个（递归）mp4，并返回 (文件路径, 第一帧PIL图像)。

    - 若 mp4 数量少于 k，则采样所有可用文件。
    - 对损坏或无法读取第一帧的视频将自动跳过。
    - 返回顺序为采样后的顺序。
    """
    root = Path(root_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"目录不存在或不可用: {root}")

    all_mp4s = list(_iter_mp4_files(root))
    if not all_mp4s:
        return []

    rng = random.Random(seed)
    n = min(k, len(all_mp4s))
    sampled = rng.sample(all_mp4s, n)

    results: List[Tuple[str, Image.Image]] = []
    for p in sampled:
        try:
            img = _read_first_frame_pil(p)
            path = str(p)
            title = re.search(title_pattern, path).group(1) if re.search(title_pattern, path) else p.stem
            results.append((path, title, img))
        except Exception:
            # 跳过不可读的视频
            continue

    return results


class QwenCaptioner:
    def __init__(self):
        self.client = OpenAI(
            api_key=DASHSCOPE,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    
    def caption(self, img: Image.Image):
        url = image_to_data_url(img)
        completion = self.client.chat.completions.create(
            model="qwen3-vl-plus", # 此处以qwen3-vl-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/models
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            # 需要注意，传入Base64，图像格式（即image/{format}）需要与支持的图片列表中的Content Type保持一致。"f"是字符串格式化的方法。
                            # PNG图像：  f"data:image/png;base64,{base64_image}"
                            # JPEG图像： f"data:image/jpeg;base64,{base64_image}"
                            # WEBP图像： f"data:image/webp;base64,{base64_image}"
                            "image_url": {"url": f"{url}"}, 
                        },
                        {"type": "text", "text": "生成简短的描述性caption"},
                    ],
                }
            ],
        )
        return completion.choices[0].message.content
    

def main(args):
    camera_poses = [
        'Pan_Down',
        'Pan_Left',
        'Pan_Right',
        'Pan_Up',
        'Zoom_In',
        'Zoom_Out',
        'ACW',
        'CW',
        'Pan_Left_Back',
        'Pan_Left_Return',
        'Pan_Right_Left',
    ]
    src = sample_mp4_first_frames(args.dir, args.k, args.pattern)
    captioner = QwenCaptioner()
    suffix = "_5b" if "5b" in args.model else ""
    for sample in src:
        print(f"path: {sample[0]}, title: {sample[1]}")
        print(captioner.caption(sample[2]))
        with tempfile.NamedTemporaryFile(mode="w+b", suffix=".png", prefix=f"tmp_", delete=True) as f:
            sample[2].save(f, format="PNG")
            caption = captioner.caption(sample[2])
            caption.replace(" ", '')
            cams = []
            threads = []
            for thread_id in range(args.num_threads):
                cam = random.choice(camera_poses)
                while cam in cams:
                    cam = random.choice(camera_poses)
                cams.append(cam)
                threads.append(threading.Thread(target=lambda: subprocess.run(
                    [
                        "bash",
                        f"cam_ctrl{suffix}.sh", 
                        "--cam", cam, 
                        "--path_suffix", f"recam/{sample[1]}", 
                        "--image", f.name, 
                        "--text",
                        caption,
                    ],
                    check=True,
                    env={**os.environ, "visible": str(thread_id + args.offset)}
                )))
                threads[-1].start()

            for t in threads:
                t.join()
        dir_path = f"samples/wan-videos-fun-control{suffix}/recam/{sample[1]}"
        os.makedirs(f"{dir_path}", exist_ok=True)
        with open(f"{dir_path}/caption.txt", "w", encoding="utf-8") as f:
            f.write(caption)

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('-k', type=int, default=2, help='Number of mp4 files to sample')
        parser.add_argument('-d', '--dir', type=str, default='asset', help='Directory to search for mp4 files')
        parser.add_argument('-p', '--pattern', type=str, default=r'([^/\\]+)\.mp4$', help='Regex pattern to extract title from file path')
        parser.add_argument('-t', '--num_threads', type=int, default=3, help='Number of threads to run in parallel for each video')
        parser.add_argument('-m', '--model', type=str, default='14b', help='Model to use')
        parser.add_argument('--offset', type=int, default=0, help='Offset for visible env variable')
        args = parser.parse_args()
        main(args)
        # src = sample_mp4_first_frames(args.dir, args.k, args.pattern)
        # sample = src[0]
        # captioner = QwenCaptioner()
        # print(f"path: {sample[0]}, title: {sample[1]}")
        # print(captioner.caption(sample[2]))
        # with tempfile.NamedTemporaryFile(mode="w+b", suffix=".png", prefix="tmp_", delete=True) as f:
        #     sample[2].save(f, format="PNG")
        #     subprocess.run(
        #         ["python3", "examples/wan2.2_fun/predict_v2v_control_camera_5b.py", 
        #         "--cam", "Zoom_In", 
        #         "--path_suffix", f"demo/{sample[1]}", 
        #         "--image", f.name, 
        #         "--text", captioner.caption(sample[2])],
        #         check=True
        #     )
    except Exception as e:
        # 直接在异常发生处进入调试（Post-mortem debugging）
        import traceback, pdb
        traceback.print_exc()
        pdb.post_mortem(e.__traceback__)
