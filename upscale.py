import os
import shutil
import zipfile
import cv2

BASE_DIR = "upscale_work"

SUPPORTED_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
)


def make_user_dirs(user_id):
    user_dir = os.path.join(BASE_DIR, str(user_id))
    input_dir = os.path.join(user_dir, "input")
    output_dir = os.path.join(user_dir, "output")

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    return user_dir, input_dir, output_dir


def cleanup_user(user_id):
    user_dir = os.path.join(BASE_DIR, str(user_id))

    if os.path.exists(user_dir):
        shutil.rmtree(user_dir, ignore_errors=True)


def upscale_image(input_path, output_path, scale=4):
    image = cv2.imread(input_path)

    if image is None:
        raise ValueError("Unable to read image")

    height, width = image.shape[:2]

    new_width = width * scale
    new_height = height * scale

    # Prevent accidental enormous memory allocation
    if new_width * new_height > 120_000_000:
        raise ValueError("Output image is too large")

    upscaled = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_LANCZOS4
    )

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    # Save as JPG
    success = cv2.imwrite(
        output_path,
        upscaled,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95
        ]
    )

    if not success:
        raise ValueError("Unable to save image")


def extract_zip(zip_path, input_dir):
    with zipfile.ZipFile(zip_path, "r") as z:

        # Basic ZIP path traversal protection
        for member in z.namelist():

            target = os.path.abspath(
                os.path.join(input_dir, member)
            )

            base = os.path.abspath(input_dir)

            if not target.startswith(base + os.sep):
                raise ValueError(
                    "Unsafe ZIP file detected"
                )

        z.extractall(input_dir)


def process_zip(
    zip_path,
    user_id,
    scale=4,
    progress_callback=None,
    cancel_callback=None
):
    """
    Returns:

        output_zip
        processed
        failed
        cancelled
    """

    user_dir, input_dir, output_dir = make_user_dirs(
        user_id
    )

    extract_zip(
        zip_path,
        input_dir
    )

    images = []

    # Find all images
    for root, dirs, files in os.walk(input_dir):

        for filename in files:

            if filename.lower().endswith(
                SUPPORTED_EXTENSIONS
            ):

                images.append(
                    os.path.join(
                        root,
                        filename
                    )
                )

    if not images:
        raise ValueError(
            "No supported images found in ZIP"
        )

    total = len(images)
    processed = 0
    failed = 0
    cancelled = False

    for index, input_path in enumerate(images):

        # Check cancellation
        if cancel_callback and cancel_callback():
            cancelled = True
            break

        relative = os.path.relpath(
            input_path,
            input_dir
        )

        relative_without_ext = os.path.splitext(
            relative
        )[0]

        output_path = os.path.join(
            output_dir,
            relative_without_ext + ".jpg"
        )

        try:

            upscale_image(
                input_path,
                output_path,
                scale
            )

            processed += 1

        except Exception as e:

            print(
                f"Upscale failed: "
                f"{input_path}: {e}"
            )

            failed += 1

        # Progress
        percent = int(
            ((index + 1) / total) * 100
        )

        if progress_callback:

            progress_callback(
                percent,
                index + 1,
                total
            )

    if cancelled:

        return (
            None,
            processed,
            failed,
            True
        )

    if processed == 0:

        raise ValueError(
            "No images could be processed"
        )

    # Create output ZIP
    output_zip = os.path.join(
        user_dir,
        "upscaled_4x.zip"
    )

    with zipfile.ZipFile(
        output_zip,
        "w",
        compression=zipfile.ZIP_DEFLATED
    ) as z:

        for root, dirs, files in os.walk(
            output_dir
        ):

            for filename in files:

                file_path = os.path.join(
                    root,
                    filename
                )

                arcname = os.path.relpath(
                    file_path,
                    output_dir
                )

                z.write(
                    file_path,
                    arcname
                )

    return (
        output_zip,
        processed,
        failed,
        False
    )
