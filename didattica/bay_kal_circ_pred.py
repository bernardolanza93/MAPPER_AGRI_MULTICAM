import fitz
import numpy as np
import cv2

# file path you want to extract images from
file = "/home/mmt-ben/MAPPER_AGRI_MULTICAM/MMI_mix_v3.pdf"
#piu aumenta piu mantiene solo le cose chiare dell immagine
delighting_coeff= 0.4


def opencv_image_to_pdf(opencv_image, output_pdf):
    doc = fitz.open()
    img = fitz.open(opencv_image)  # open pic as document
    rect = img[0].rect  # pic dimension
    pdfbytes = img.convert_to_pdf()  # make a PDF stream
    img.close()  # no longer needed
    imgPDF = fitz.open("pdf", pdfbytes)  # open stream as PDF
    page = doc.new_page(width=rect.width,  # new page with ...
                        height=rect.height)  # pic dimension
    page.show_pdf_page(rect, imgPDF, 0)  # image fills the page

    doc.save(output_pdf)


with fitz.Document(file) as doc:
    for xref in {xref[0] for page in doc for xref in page.get_images(False) if xref[1] == 0}:
        # dictionary with image
        image_dict = doc.extract_image(xref)
        # image as OpenCV's Mat
        i = cv2.imdecode(np.frombuffer(image_dict["image"],
                                       np.dtype(f'u{image_dict["bpc"] // 8}')
                                       ),
                         cv2.IMREAD_GRAYSCALE)

        # Calculate the mean intensity of the grayscale image
        mean_intensity = np.mean(i)
        res = 255 - mean_intensity
        mean_intensity =delighting_coeff * res + mean_intensity
        print("mean:",mean_intensity)



        # Apply thresholding
        _, thresholded_image = cv2.threshold(i, mean_intensity, 255, cv2.THRESH_BINARY)

        cv2.imshow("OpenCV", thresholded_image)
        cv2.imshow("OpenChhV", i)
        cv2.waitKey(0)

        write_image = "thr.png"

        cv2.imwrite(write_image, thresholded_image)


output_pdf = "/home/mmt-ben/MAPPER_AGRI_MULTICAM/out.pdf"
opencv_image_to_pdf(write_image, output_pdf)