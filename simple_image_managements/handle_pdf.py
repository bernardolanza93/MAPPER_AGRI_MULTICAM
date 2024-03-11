

# Example usage:
input_folder = "/home/mmt-ben/MAPPER_AGRI_MULTICAM/pdf/inp"
output_folder = "/home/mmt-ben/MAPPER_AGRI_MULTICAM/pdf/out"
pdf_filename = "MMI1761.pdf"
import fitz  # PyMuPDF
import cv2
import numpy as np
import os
from PIL import Image
import io


def PDF_text_to_image_convertor(input_folder, output_folder, pdf_filename):
    # Path to input and output PDF
    input_pdf_path = os.path.join(input_folder, pdf_filename)
    output_pdf_path = os.path.join(output_folder, pdf_filename)

    # Open the PDF
    pdf_document = fitz.open(input_pdf_path)

    # Initialize text content and image list
    text_content = ""
    images = []

    # Extract text and images from each page
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)

        # Extract text
        text_content += page.get_text()

        # Extract images
        image_list = page.get_images(full=True)
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            base_image = pdf_document.extract_image(xref)
            image_data = base_image["image"]
            try:
                # Check if image data is valid
                Image.open(io.BytesIO(image_data)).verify()
                image = Image.open(io.BytesIO(image_data))
                images.append(image)
            except Exception as e:
                print(f"Error processing image {img_index + 1} on page {page_number + 1}: {e}")

    # Close the PDF document
    pdf_document.close()

    # Split text into lines
    text_lines = text_content.split('\n')

    # Calculate text size for formatting
    font_scale = 1
    font_thickness = 1
    font_face = cv2.FONT_HERSHEY_SIMPLEX

    max_text_width = 0
    text_height = 0

    for line in text_lines:
        text_size = cv2.getTextSize(line, font_face, font_scale, font_thickness)[0]
        max_text_width = max(max_text_width, text_size[0])
        text_height += text_size[1]

    # Add padding to text size
    padding_x = 20
    padding_y = 20
    text_width = max_text_width + padding_x
    text_height += padding_y

    # Create white image
    text_image = np.ones((text_height, text_width, 3), dtype=np.uint8) * 255

    # Put text on the image
    y = padding_y
    for line in text_lines:
        cv2.putText(text_image, line, (padding_x // 2, y), font_face, font_scale, (0, 0, 0), font_thickness,
                    cv2.LINE_AA)
        y += text_size[1]

    # Print the extracted text
    print("Extracted Text:\n", text_content)

    # Save the extracted text image
    text_image_path = os.path.join(output_folder, "extracted_text.png")
    cv2.imwrite(text_image_path, text_image)

    # Save the extracted text as an image in the output folder
    for img_index, img in enumerate(images):
        img.save(os.path.join(output_folder, f"image_{img_index}.png"))

    # Convert the text image to fitz.Pixmap format
    text_image_pil = Image.fromarray(text_image)
    text_image_bytes = io.BytesIO()
    text_image_pil.save(text_image_bytes, format='PNG')
    text_image_bytes.seek(0)
    text_image_fitz = fitz.Pixmap(text_image_bytes)

    # Create a new PDF containing the extracted text image and extracted images
    output_pdf = fitz.open()
    # Insert the extracted text image as the first page
    page = output_pdf.new_page(width=text_width, height=text_height)
    page.insert_image(page.rect, pixmap=text_image_fitz)
    # Insert the extracted images
    for img_index, img in enumerate(images):
        page = output_pdf.new_page(width=img.width, height=img.height)
        pix = fitz.Pixmap(img)
        page.insert_image(page.rect, pixmap=pix)

    # Save the new PDF
    output_pdf.save(output_pdf_path)
    output_pdf.close()

    # Return paths to the generated images
    return output_pdf_path


generated_pdf_path = PDF_text_to_image_convertor(input_folder, output_folder, pdf_filename)
print("Generated PDF Path:", generated_pdf_path)
