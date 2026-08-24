import os
import cv2
import zipfile
import shutil


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "RealESRGAN_x4plus"

MODEL_PATH = "models/RealESRGAN_x4plus.pth"

SCALE = 4

MAX_4K_WIDTH = 3840
MAX_4K_HEIGHT = 2160


# ============================================================
# MODEL CACHE
# ============================================================

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
            "torch\n"
            "torchvision\n"
            "basicsr\n"
            "realesrgan"
        ) from e

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Real-ESRGAN model not found:\n"
            f"{MODEL_PATH}"
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

    _model = RealESRGANer(
        scale=4,
        model_path=MODEL_PATH,
        model=model,
        tile=256,
        tile_pad=10,
        pre_pad=0,
        half=False
    )

    return _model


# ============================================================
# LIMIT IMAGE TO 4K
# ============================================================

def limit_to_4k(image):

    height, width = image.shape[:2]

    if (
        width <= MAX_4K_WIDTH
        and height <= MAX_4K_HEIGHT
    ):
        return image

    ratio = min(
        MAX_4K_WIDTH / width,
        MAX_4K_HEIGHT / height
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
# SAVE IMAGE
# ============================================================

def save_image(
    output_path,
    image
):

    extension = os.path.splitext(
        output_path
    )[1].lower()

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    if extension in (
        ".jpg",
        ".jpeg"
    ):

        success = cv2.imwrite(
            output_path,
            image,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                98
            ]
        )

    elif extension == ".png":

        success = cv2.imwrite(
            output_path,
            image,
            [
                cv2.IMWRITE_PNG_COMPRESSION,
                2
            ]
        )

    else:

        success = cv2.imwrite(
            output_path,
            image
        )

    if not success:

        raise RuntimeError(
            f"Failed to save image: {output_path}"
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
            input_path
        )

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    image = cv2.imread(
        input_path,
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise ValueError(
            f"Unable to read image:\n"
            f"{input_path}"
        )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    upsampler = load_model()

    # --------------------------------------------------------
    # REAL-ESRGAN
    # --------------------------------------------------------

    try:

        output, _ = upsampler.enhance(
            image,
            outscale=SCALE
        )

    except Exception as e:

        raise RuntimeError(
            f"Real-ESRGAN failed:\n{e}"
        )

    # --------------------------------------------------------
    # 4K LIMIT
    # --------------------------------------------------------

    if max_4k:

        output = limit_to_4k(
            output
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_image(
        output_path,
        output
    )

    del image
    del output

    return output_path


# ============================================================
# CLEANUP USER FILES
# ============================================================

def cleanup_user(user_id):

    user_dir = os.path.join(
        "upscale_work",
        str(user_id)
    )

    if os.path.exists(user_dir):

        shutil.rmtree(
            user_dir,
            ignore_errors=True
        )


# ============================================================
# FIND IMAGES IN DIRECTORY
# ============================================================

def find_images(directory):

    image_files = []

    for root, dirs, files in os.walk(
        directory
    ):

        for filename in files:

            if filename.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".webp",
                    ".bmp"
                )
            ):

                image_files.append(
                    os.path.join(
                        root,
                        filename
                    )
                )

    image_files.sort()

    return image_files


# ============================================================
# PROCESS ZIP
# ============================================================

def process_zip(
    zip_path,
    user_id,
    scale=4,
    progress_callback=None,
    cancel_callback=None
):

    if not os.path.exists(zip_path):

        raise FileNotFoundError(
            zip_path
        )

    # ========================================================
    # DIRECTORIES
    # ========================================================

    user_dir = os.path.join(
        "upscale_work",
        str(user_id)
    )

    extract_dir = os.path.join(
        user_dir,
        "input"
    )

    output_dir = os.path.join(
        user_dir,
        "output"
    )

    os.makedirs(
        extract_dir,
        exist_ok=True
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # ========================================================
    # EXTRACT ZIP
    # ========================================================

    try:

        with zipfile.ZipFile(
            zip_path,
            "r"
        ) as archive:

            archive.extractall(
                extract_dir
            )

    except zipfile.BadZipFile:

        raise ValueError(
            "Invalid or corrupted ZIP file."
        )

    # ========================================================
    # FIND IMAGES
    # ========================================================

    image_files = find_images(
        extract_dir
    )

    total = len(
        image_files
    )

    if total == 0:

        raise ValueError(
            "No supported images found in ZIP."
        )

    processed = 0
    failed = 0
    cancelled = False

    # ========================================================
    # PROCESS EACH IMAGE
    # ========================================================

    for index, input_path in enumerate(
        image_files,
        start=1
    ):

        # ----------------------------------------------------
        # CHECK CANCEL
        # ----------------------------------------------------

        if cancel_callback:

            try:

                if cancel_callback():

                    cancelled = True

                    break

            except Exception:

                pass

        # ----------------------------------------------------
        # OUTPUT NAME
        # ----------------------------------------------------

        filename = os.path.basename(
            input_path
        )

        name, extension = os.path.splitext(
            filename
        )

        output_path = os.path.join(
            output_dir,
            f"{name}_4K.jpg"
        )

        # ----------------------------------------------------
        # UPSCALE
        # ----------------------------------------------------

        try:

            upscale_image(
                input_path,
                output_path,
                max_4k=True
            )

            processed += 1

        except Exception as e:

            print(
                f"[UPSCALE ERROR] "
                f"{filename}: {e}"
            )

            failed += 1

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        percent = int(
            (index / total) * 100
        )

        if progress_callback:

            try:

                progress_callback(
                    percent,
                    index,
                    total
                )

            except Exception:

                pass

    # ========================================================
    # CANCELLED
    # ========================================================

    if cancelled:

        return (
            None,
            processed,
            failed,
            True
        )

    # ========================================================
    # CHECK OUTPUT
    # ========================================================

    output_images = find_images(
        output_dir
    )

    if not output_images:

        return (
            None,
            processed,
            failed,
            False
        )

    # ========================================================
    # CREATE OUTPUT ZIP
    # ========================================================

    output_zip = os.path.join(
        user_dir,
        f"{user_id}_4K_upscaled.zip"
    )

    try:

        with zipfile.ZipFile(
            output_zip,
            "w",
            compression=zipfile.ZIP_DEFLATED
        ) as archive:

            for file_path in output_images:

                archive.write(
                    file_path,
                    arcname=os.path.basename(
                        file_path
                    )
                )

    except Exception as e:

        raise RuntimeError(
            f"Failed to create output ZIP:\n{e}"
        )

    # ========================================================
    # RETURN
    # ========================================================

    return (
        output_zip,
        processed,
        failed,
        False
    )


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    input_image = "input.jpg"

    output_image = (
        "output_4K.jpg"
    )

    result = upscale_image(
        input_image,
        output_image,
        max_4k=True
    )

    print(
        f"Upscaled image saved:\n{result}"
    )
