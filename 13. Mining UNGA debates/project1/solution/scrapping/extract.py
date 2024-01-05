import os
import pdfplumber
import country_converter as coco


class PdfExtractor:
    def __init__(
        self,
        pdf_folder: str = "corpora/pdfs",
        txt_folder: str = "corpora/UN General Debate Corpus/TXT/Session 78 - 2023",
    ) -> None:
        self.pdf_folder = pdf_folder
        self.txt_folder = txt_folder
        os.makedirs(self.txt_folder, exist_ok=True)

    def _prepare_filename(self, pdf_file: str) -> str:
        pdf_file = pdf_file.split("/")[-1]
        iso_2 = pdf_file.split("_")[0]
        country_name = coco.convert(names=iso_2, to="ISO3", not_found=None)
        return country_name + "_78_2023.txt"

    def _process_text(self, text: str, pdf_file: str) -> str:
        if text is None or len(text) == 0:
            print(f"Could not find text in {pdf_file}")
            return text

        possible_starts = [
            text.lower().find("mr. president"),
            text.lower().find("mr president"),
            text.lower().find("ladies and gentlemen"),
            text.lower().find("your excellenc"),
            text.lower().find("excellencies"),
            text.lower().find("the world over"),
            text.lower().find("this year"),
            text.lower().find("esteemed president"),
            text.lower().find("president –"),
        ]
        starts = [start for start in possible_starts if start != -1]
        start = min(starts) if len(starts) > 0 else -1
        if start == -1:
            print(f"Could not find probable start in the text of {pdf_file}")
            start = 0
        return text[start:]

    def save_texts(self) -> None:
        files = os.listdir(self.pdf_folder)
        pdf_files = [
            os.path.join(self.pdf_folder, file)
            for file in files
            if file.endswith(".pdf")
        ]

        for pdf_file in pdf_files:
            with pdfplumber.open(pdf_file) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text()

                text = self._process_text(text, pdf_file)
                filename = self._prepare_filename(pdf_file)
                # prepare the txt file name
                txt_file = os.path.join(self.txt_folder, filename)
                with open(txt_file, "w") as f:
                    f.write(text)


def main():
    extractor = PdfExtractor()
    extractor.save_texts()


if __name__ == "__main__":
    main()
