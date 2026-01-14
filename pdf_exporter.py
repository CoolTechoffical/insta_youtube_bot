from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import inch

def images_to_pdf(image_caption_list, output_pdf):
    c = canvas.Canvas(output_pdf, pagesize=A4)
    page_w, page_h = A4

    for img_path, caption in image_caption_list:
        try:
            img = ImageReader(img_path)
        except:
            continue

        # Image size
        img_w = page_w - 1.2 * inch
        img_h = page_h - 2.5 * inch

        c.drawImage(
            img,
            0.6 * inch,
            1.5 * inch,
            img_w,
            img_h,
            preserveAspectRatio=True,
            anchor="c"
        )

        # Caption
        c.setFont("Helvetica-Bold", 12)
        c.drawCentredString(
            page_w / 2,
            1.0 * inch,
            caption
        )

        c.showPage()

    c.save()
