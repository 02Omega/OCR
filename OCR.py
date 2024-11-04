import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import PyPDF2
import io
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from optimized_loader import Essay, Dataset
from essay_judge import main as judge_essays


class HandwrittenOCR:
    def __init__(self):
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def preprocess_image(self, image):
        # Convert to RGB if it's not
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image

    def recognize_text(self, image):
        image = self.preprocess_image(image)
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)

        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text


class HandwrittenEssayProcessor:
    def __init__(self, output_dir="results/"):
        self.ocr = HandwrittenOCR()
        self.output_dir = output_dir

    def process_pdf(self, pdf_path, metadata=None):
        """
        Process a PDF containing handwritten essays and return the extracted text.

        Args:
            pdf_path (str): Path to the PDF file
            metadata (dict): Optional metadata about the essay (grade level, etc.)
        """
        pdf_file = open(pdf_path, 'rb')
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""

        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]

            if '/XObject' in page['/Resources']:
                xObject = page['/Resources']['/XObject'].get_object()
                for obj in xObject:
                    if xObject[obj]['/Subtype'] == '/Image':
                        size = (xObject[obj]['/Width'], xObject[obj]['/Height'])
                        data = xObject[obj].get_data()
                        img = Image.open(io.BytesIO(data))
                        text += self.ocr.recognize_text(img) + "\n"

        pdf_file.close()
        return text

    def create_essay_object(self, text, metadata=None):
        """
        Create an Essay object from the extracted text.

        Args:
            text (str): The extracted essay text
            metadata (dict): Optional metadata about the essay
        """
        if metadata is None:
            metadata = {}

        return Essay(
            essay_id=metadata.get('essay_id', 'OCR_ESSAY'),
            full_text=text,
            grade_level=metadata.get('grade_level', 'Unknown'),
            ell_status=metadata.get('ell_status', 'Unknown'),
            prompt_name=metadata.get('prompt_name', 'Unknown'),
            discourses={
                'Unannotated': [text],
                'Position': [],
                'Claim': [],
                'Counterclaim': [],
                'Rebuttal': [],
                'Evidence': [],
                'Concluding Statement': []
            }
        )

    def process_and_evaluate(self, pdf_paths, metadata_list=None):
        """
        Process multiple PDFs and evaluate them using the essay judge.

        Args:
            pdf_paths (list): List of paths to PDF files
            metadata_list (list): Optional list of metadata dictionaries
        """
        if metadata_list is None:
            metadata_list = [None] * len(pdf_paths)

        essays = []
        for pdf_path, metadata in zip(pdf_paths, metadata_list):
            # Extract text from PDF
            text = self.process_pdf(pdf_path, metadata)

            # Save the extracted text
            text_filename = f"{self.output_dir}extracted_{pdf_path.split('/')[-1]}.txt"
            with open(text_filename, 'w', encoding='utf-8') as f:
                f.write(text)

            # Create Essay object
            essay = self.create_essay_object(text, metadata)
            essays.append(essay)

        # Evaluate essays using the existing judge system
        judge_essays(
            num_essays=len(essays),
            set_type='custom',
            output_dir=self.output_dir
        )

        return essays


def main():
    # Example usage
    processor = HandwrittenEssayProcessor(output_dir="results/")

    # Example metadata
    metadata = {
        'essay_id': 'HW_001',
        'grade_level': '8',
        'ell_status': 'No',
        'prompt_name': 'Should students be required to wear school uniforms?'
    }

    # Process a single PDF
    pdf_paths = ['path/to/your/handwritten_document.pdf']
    metadata_list = [metadata]

    essays = processor.process_and_evaluate(pdf_paths, metadata_list)
    print(f"Processed {len(essays)} essays. Results saved in the results directory.")


if __name__ == "__main__":
    main()