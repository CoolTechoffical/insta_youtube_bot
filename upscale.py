import os
import gc
import cv2

# ============================================================
# REAL-ESRGAN 4× SUPER-RESOLUTION
# ============================================================

MODEL_PATH = "models/RealESRGAN_x4plus.pth"

SCALE = 4

# Keep this True for maximum 3840×2160 output.
MAX_4K = True

_model = None


# ============================================================
# LOAD REAL-ESRGAN MODEL
# ============================================================

def load_model():

    global _model

    if _model is not None:
        return _model

    try:

        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

    except ImportError as e:

        raise RuntimeError(
            "Real-ESRGAN dependencies are missing.\n"
            "Install:\n"
            "basicsr\n"
            "realesrgan\n"
            "torch\n"
            "torchvision"
        ) from e

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Real-ESRGAN model not found:\n"
            f"{MODEL_PATH}\n\n"
            "Place RealESRGAN_x4plus.pth inside:\n"
            "models/"
        )

    # ========================================================
    # REAL-ESRGAN X4 ARCHITECTURE
    # ========================================================

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4
    )

    # ========================================================
    # UPSAMPLER
    # ========================================================

    _model = RealESRGANer(

        scale=4,

        model_path=MODEL_PATH,

        model=model,

        # Smaller tile = lower RAM usage
        tile=256,

        tile_pad=10,

        pre_pad=0,

        # CPU-safe
        half=False
    )

    return _model


# ============================================================
# OPTIONAL 4K LIMIT
# ============================================================

def limit_to_4k(image):

    if image is None:
        return None

    height, width = image.shape[:2]

    max_width = 3840
    max_height = 2160

    if (
        width <= max_width
        and height <= max_height
    ):
        return image

    ratio = min(
        max_width / width,
        max_height / height
    )

    new_width = max(
        1,
        int(width * ratio)
    )

    new_height = max(
        1,
        int(height * ratio)
    )

    return cv2.resize(
        image,
        (
            new_width,
            new_height
        ),
        interpolation=cv2.INTER_AREA
    )


# ============================================================
# UPSCALE ONE IMAGE
# ============================================================

def upscale_image(
    input_path,
    output_path,
    max_4k=True
):

    if not os.path.exists(input_path):

        raise FileNotFoundError(
            f"Input image not found:\n{input_path}"
        )

    # ========================================================
    # READ IMAGE
    # ========================================================

    image = cv2.imread(
        input_path,
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise ValueError(
            f"Unable to read image:\n{input_path}"
        )

    # ========================================================
    # LOAD AI MODEL
    # ========================================================

    upsampler = load_model()

    # ========================================================
    # REAL AI SUPER-RESOLUTION
    # ========================================================

    try:

        output, _ = upsampler.enhance(
            image,
            outscale=4
        )

    except Exception as e:

        del image
        gc.collect()

        raise RuntimeError(
            f"Real-ESRGAN enhancement failed:\n{e}"
        )

    # ========================================================
    # OPTIONAL 4K MAXIMUM
    # ========================================================

    if max_4k:

        output = limit_to_4k(
            output
        )

    # ========================================================
    # SAVE OUTPUT
    # ========================================================

    output_dir = os.path.dirname(
        output_path
    )

    if output_dir:

        os.makedirs(
            output_dir,
            exist_ok=True
        )

    extension = os.path.splitext(
        output_path
    )[1].lower()

    if extension in (
        ".jpg",
        ".jpeg"
    ):

        success = cv2.imwrite(
            output_path,
            output,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                98
            ]
        )

    elif extension == ".png":

        success = cv2.imwrite(
            output_path,
            output,
            [
                cv2.IMWRITE_PNG_COMPRESSION,
                2
            ]
        )

    elif extension == ".webp":

        success = cv2.imwrite(
            output_path,
            output,
            [
                cv2.IMWRITE_WEBP_QUALITY,
                98
            ]
        )

    else:

        success = cv2.imwrite(
            output_path,
            output
        )

    # ========================================================
    # MEMORY CLEANUP
    # ========================================================

    del image
    del output

    gc.collect()

    if not success:

        raise RuntimeError(
            f"Failed to save output image:\n"
            f"{output_path}"
        )

    return output_path


# ============================================================
# GET OUTPUT INFORMATION
# ============================================================

def get_image_dimensions(image_path):

    image = cv2.imread(
        image_path,
        cv2.IMREAD_IGNORE_ORIENTATION
    )

    if image is None:

        return None

    height, width = image.shape[:2]

    del image

    return {
        "width": width,
        "height": height
    }


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    input_image = "input.jpg"

    output_image = "output_4k.jpg"

    result = upscale_image(
        input_image,
        output_image,
        max_4k=True
    )

    print(
        "✅ Real-ESRGAN complete"
    )

    print(
        f"Output: {result}"
    )

    dimensions = get_image_dimensions(
        result
    )

    if dimensions:

        print(
            f"Resolution: "
            f"{dimensions['width']}×"
            f"{dimensions['height']}"
        )
