import base64

import pymupdf


def encode_image(pymu_page):
    doc_bytes = pymu_page.get_pixmap().tobytes()
    return base64.b64encode(doc_bytes).decode("utf-8")


def get_base64_image_from_pdf(file_path: str, page_number: list[int]):
    pdf_document = pymupdf.open(file_path)

    res = []
    for page in page_number:
        page = int(page)
        pymu_page = pdf_document[page - 1]
        base64_image = encode_image(pymu_page)
        res.append((page, base64_image))

    return res
