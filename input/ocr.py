import os
import sys
import re
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import cv2
import pytesseract
from PIL import Image

import fitz
from pdf2image import convert_from_path
import pdfplumber

try:
    from img2table.document import Image as Img2TableImage
    from img2table.ocr import TesseractOCR as Img2TableTesseract
    TABLE_EXTRACTION_AVAILABLE = True
except ImportError:
    TABLE_EXTRACTION_AVAILABLE = False

import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger("ocr_processor")


@dataclass
class ExtractionResult:
    source_type: str
    file_path: str
    text: str = ""
    pages: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[pd.DataFrame] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_type": self.source_type,
            "file_path": self.file_path,
            "text": self.text,
            "pages": self.pages,
            "tables": [
                t.to_dict(orient="records")
                for t in self.tables
            ],
            "metadata": self.metadata,
        }

    def save_text(self, out_path: str):
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(self.text)

    def save_tables(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)

        for i, table in enumerate(self.tables):
            table.to_csv(
                os.path.join(out_dir, f"table_{i + 1}.csv"),
                index=False
            )


class UniversalExtractor:

    IMAGE_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tiff",
        ".tif",
        ".webp",
        ".gif"
    }

    TEXT_EXTENSIONS = {
        ".txt",
        ".md",
        ".csv",
        ".log"
    }

    PDF_EXTENSIONS = {
        ".pdf"
    }

    # PSM modes tried, in priority order, when hunting for the best OCR pass.
    # 6  = uniform block of text (good default for paragraphs / scanned pages)
    # 4  = single column of variable-sized text (good for forms / receipts)
    # 11 = sparse text, no particular order (good for scattered labels)
    # 3  = fully automatic page segmentation (fallback, no OSD)
    CANDIDATE_PSMS = [6, 4, 11, 3]

    def __init__(
        self,
        tesseract_lang: str = "eng",
        dpi: int = 300,
        min_chars_for_native_text: int = 20
    ):
        self.lang = tesseract_lang
        self.dpi = dpi
        self.min_chars_for_native_text = min_chars_for_native_text

        if TABLE_EXTRACTION_AVAILABLE:
            self.table_ocr_engine = Img2TableTesseract(
                lang=tesseract_lang
            )
        else:
            self.table_ocr_engine = None

            logger.warning(
                "img2table not installed. "
                "Structured table detection disabled."
            )

    def process(
        self,
        file_path: str,
        extract_tables: bool = True
    ) -> ExtractionResult:

        file_path = str(file_path)
        ext = Path(file_path).suffix.lower()

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"No such file: {file_path}"
            )

        if ext in self.TEXT_EXTENSIONS:
            return self._process_text_file(file_path)

        elif ext in self.IMAGE_EXTENSIONS:
            return self._process_image(
                file_path,
                extract_tables
            )

        elif ext in self.PDF_EXTENSIONS:
            return self._process_pdf(
                file_path,
                extract_tables
            )

        else:
            raise ValueError(
                f"Unsupported file extension '{ext}'. "
                f"Supported: "
                f"{self.TEXT_EXTENSIONS | self.IMAGE_EXTENSIONS | self.PDF_EXTENSIONS}"
            )

    def _process_text_file(
        self,
        file_path: str
    ) -> ExtractionResult:

        logger.info(
            f"Reading plain text file: {file_path}"
        )

        with open(
            file_path,
            "r",
            encoding="utf-8",
            errors="replace"
        ) as f:
            content = f.read()

        result = ExtractionResult(
            source_type="text",
            file_path=file_path,
            text=content
        )

        result.pages.append({
            "page": 1,
            "text": content,
            "method": "direct_read"
        })

        result.metadata["char_count"] = len(content)

        return result

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _blur_score(gray: np.ndarray) -> float:
        return cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()

    def _correct_orientation(
        self,
        gray: np.ndarray
    ) -> np.ndarray:
        """Detect and fix 90/180/270 degree rotation using Tesseract's OSD.

        This runs BEFORE the fine-grained deskew step, which only handles
        small (<15 degree) skew and cannot recover a sideways/upside-down
        page. Without this, a rotated page silently produces near-empty
        or garbled OCR output.
        """

        try:
            osd = pytesseract.image_to_osd(
                gray,
                output_type=pytesseract.Output.DICT
            )

            rotate_by = int(osd.get("rotate", 0)) % 360
            osd_conf = float(osd.get("orientation_conf", 0) or 0)

            # Only trust the OSD result when it's reasonably confident;
            # a low-confidence guess on a mostly-blank page can do more
            # harm than good.
            if rotate_by and osd_conf >= 1.0:

                if rotate_by == 90:
                    gray = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
                elif rotate_by == 180:
                    gray = cv2.rotate(gray, cv2.ROTATE_180)
                elif rotate_by == 270:
                    gray = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)

                logger.info(
                    f"Corrected page orientation by {rotate_by} degrees "
                    f"(OSD confidence {osd_conf:.1f})"
                )

        except pytesseract.TesseractError:
            # OSD fails on very sparse/blank/low-contrast images; that's
            # fine, we just skip orientation correction in that case.
            pass

        return gray

    def _preprocess_image(
        self,
        img: Image.Image
    ) -> Image.Image:

        rgb = np.array(
            img.convert("RGB")
        )

        gray = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2GRAY
        )

        # Detect and normalize inverted images (light text on dark
        # background). Tesseract expects dark text on a light background;
        # feeding it the inverse tanks accuracy.
        if np.mean(gray) < 100:
            gray = cv2.bitwise_not(gray)

        h, w = gray.shape

        target_long_edge = 2200

        long_edge = max(h, w)

        if long_edge < target_long_edge:

            scale = (
                target_long_edge /
                long_edge
            )

            gray = cv2.resize(
                gray,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC
            )

        # Contrast Limited Adaptive Histogram Equalization evens out
        # lighting (shadows, uneven scans, phone-camera glare) before
        # thresholding, which measurably helps character segmentation.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Edge-preserving denoise. Bilateral is kept (fast, preserves
        # stroke edges) but tuned slightly gentler so thin character
        # strokes survive.
        gray = cv2.bilateralFilter(
            gray,
            d=7,
            sigmaColor=50,
            sigmaSpace=50
        )

        score = self._blur_score(gray)

        if score < 150:

            gaussian = cv2.GaussianBlur(
                gray,
                (0, 0),
                sigmaX=3
            )

            amount = (
                2.0
                if score < 60
                else 1.3
            )

            gray = cv2.addWeighted(
                gray,
                1 + amount,
                gaussian,
                -amount,
                0
            )

        gray = self._correct_orientation(gray)
        gray = self._deskew(gray)

        # Try both a global (Otsu) and a local (adaptive) threshold and
        # keep whichever leaves a more plausible ratio of foreground
        # (text) pixels. Adaptive thresholding wins on unevenly lit scans
        # but can shred thin strokes on clean, uniform images where Otsu
        # is actually cleaner - so we no longer hardcode one choice.
        candidates = []

        _, otsu = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        candidates.append(otsu)

        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31,
            C=15
        )
        candidates.append(adaptive)

        best = None
        best_ratio_dist = 1.0

        for candidate in candidates:
            # A well-thresholded document page is mostly background:
            # foreground (black text, value 0) typically covers roughly
            # 2-25% of pixels. Pick whichever candidate lands closest to
            # that band as a cheap, fast proxy for "cleanest" binarization.
            foreground_ratio = 1.0 - (np.count_nonzero(candidate) / candidate.size)
            target = 0.10
            dist = abs(foreground_ratio - target)

            if foreground_ratio > 0.005 and dist < best_ratio_dist:
                best_ratio_dist = dist
                best = candidate

        gray = best if best is not None else adaptive

        # Kernel (1,1) is a no-op (fixed bug: it left morphology doing
        # nothing). A small (2,2) close fills tiny gaps in character
        # strokes without fusing adjacent characters together.
        kernel = np.ones((2, 2), np.uint8)

        gray = cv2.morphologyEx(
            gray,
            cv2.MORPH_CLOSE,
            kernel
        )

        return Image.fromarray(gray)

    @staticmethod
    def _deskew(
        gray: np.ndarray
    ) -> np.ndarray:

        try:

            inverted = cv2.bitwise_not(
                gray
            )

            thresh = cv2.threshold(
                inverted,
                0,
                255,
                cv2.THRESH_BINARY |
                cv2.THRESH_OTSU
            )[1]

            coords = np.column_stack(
                np.where(thresh > 0)
            )

            if coords.shape[0] < 20:
                return gray

            angle = cv2.minAreaRect(
                coords
            )[-1]

            angle = (
                -(90 + angle)
                if angle < -45
                else -angle
            )

            if (
                abs(angle) < 0.5
                or abs(angle) > 15
            ):
                return gray

            h, w = gray.shape

            center = (
                w // 2,
                h // 2
            )

            matrix = cv2.getRotationMatrix2D(
                center,
                angle,
                1.0
            )

            return cv2.warpAffine(
                gray,
                matrix,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )

        except Exception:
            return gray

    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------

    def _ocr_with_confidence(
        self,
        proc_img: Image.Image,
        psm: int
    ) -> Tuple[str, Optional[float]]:

        config = (
            f"--oem 3 --psm {psm} --dpi {self.dpi} "
            f"-c preserve_interword_spaces=1"
        )

        text = pytesseract.image_to_string(
            proc_img,
            lang=self.lang,
            config=config
        )

        tsv_data = pytesseract.image_to_data(
            proc_img,
            lang=self.lang,
            config=config,
            output_type=pytesseract.Output.DATAFRAME
        )

        tsv_data = tsv_data.dropna(
            subset=["text"]
        )

        # Ignore purely-whitespace "words" when scoring confidence; they
        # report conf=-1 or garbage values that skew the average down.
        tsv_data = tsv_data[
            tsv_data["text"].astype(str).str.strip() != ""
        ]

        valid_conf = tsv_data.loc[
            tsv_data["conf"] != -1,
            "conf"
        ]

        avg_conf = (
            valid_conf.mean()
            if not valid_conf.empty
            else None
        )

        return (
            text,
            float(avg_conf)
            if avg_conf is not None
            else None
        )

    def _best_ocr_pass(
        self,
        proc_img: Image.Image
    ) -> Tuple[str, Optional[float], str]:
        """Try several page-segmentation modes and keep the best result.

        The original code only ever compared PSM 6 against a single PSM 4
        retry, and only retried at all if confidence was already under 60.
        Different documents (paragraphs, forms, receipts, sparse labels)
        genuinely need different PSMs, so we score a broader candidate set
        and pick whichever pass has both reasonable confidence AND actually
        recovered a non-trivial amount of text.
        """

        best_text = ""
        best_conf: Optional[float] = None
        best_method = f"tesseract_ocr (psm {self.CANDIDATE_PSMS[0]})"
        best_score = -1.0

        for i, psm in enumerate(self.CANDIDATE_PSMS):

            text, conf = self._ocr_with_confidence(proc_img, psm=psm)

            stripped_len = len(text.strip())

            if stripped_len == 0:
                continue

            conf_component = (conf if conf is not None else 0.0)
            # Blend confidence with recovered text length (capped) so an
            # empty-but-"confident" pass never beats a pass that actually
            # extracted real content.
            score = conf_component + min(stripped_len, 500) / 50.0

            is_first_viable = best_score < 0

            if score > best_score:
                best_score = score
                best_text = text
                best_conf = conf
                best_method = f"tesseract_ocr (psm {psm})"

            # Early exit once we hit a high-confidence, substantial pass -
            # no need to keep burning time on every remaining PSM.
            if conf is not None and conf >= 85 and stripped_len >= 20:
                break

            del is_first_viable  # (kept for readability of intent above)

        return best_text, best_conf, best_method

    def _process_image(
        self,
        file_path: str,
        extract_tables: bool
    ) -> ExtractionResult:

        logger.info(
            f"Running OCR on image: {file_path}"
        )

        raw_img = Image.open(
            file_path
        )

        proc_img = self._preprocess_image(
            raw_img
        )

        text, avg_conf, method = self._best_ocr_pass(proc_img)
        text = self._clean_text(text)

        result = ExtractionResult(
            source_type="image",
            file_path=file_path,
            text=text
        )

        result.pages.append({
            "page": 1,
            "text": text,
            "method": method,
            "avg_confidence": (
                round(avg_conf, 2)
                if avg_conf is not None
                else None
            )
        })

        result.metadata["image_size"] = (
            raw_img.size
        )

        result.metadata["char_count"] = len(
            text
        )

        if extract_tables:
            result.tables.extend(
                self._extract_tables_from_image(
                    file_path
                )
            )

        return result

    @staticmethod
    def _clean_text(text: str) -> str:
        """Light, conservative post-processing of raw OCR output.

        Only strips artifacts that are unambiguously noise (trailing
        whitespace, runs of 3+ blank lines, stray control characters) -
        it never rewrites or "corrects" words, since silently changing
        recognized text is worse than leaving OCR noise in place.
        """

        if not text:
            return text

        # Drop non-printable control characters Tesseract occasionally
        # emits on noisy regions.
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

        # Trim trailing whitespace on each line.
        lines = [line.rstrip() for line in text.split("\n")]

        # Collapse 3+ consecutive blank lines down to 1.
        cleaned_lines: List[str] = []
        blank_run = 0

        for line in lines:
            if line.strip() == "":
                blank_run += 1
                if blank_run <= 1:
                    cleaned_lines.append(line)
            else:
                blank_run = 0
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()

    def _extract_tables_from_image(
        self,
        image_path: str
    ) -> List[pd.DataFrame]:

        if not TABLE_EXTRACTION_AVAILABLE:
            return []

        try:

            doc = Img2TableImage(
                image_path
            )

            extracted = doc.extract_tables(
                ocr=self.table_ocr_engine,
                implicit_rows=True,
                borderless_tables=True,
                min_confidence=50
            )

            tables = []

            for table in extracted:
                tables.append(
                    table.df
                )

            logger.info(
                f"Found {len(tables)} table(s) "
                f"in image {image_path}"
            )

            return tables

        except Exception as e:

            logger.warning(
                f"Table extraction failed "
                f"for image {image_path}: {e}"
            )

            return []

    def _process_pdf(
        self,
        file_path: str,
        extract_tables: bool
    ) -> ExtractionResult:

        logger.info(
            f"Processing PDF: {file_path}"
        )

        doc = fitz.open(
            file_path
        )

        result = ExtractionResult(
            source_type="pdf",
            file_path=file_path
        )

        all_text_chunks = []

        pages_needing_ocr = []

        native_text_per_page = {}

        for page_index in range(
            len(doc)
        ):

            page = doc[
                page_index
            ]

            native_text = page.get_text(
                "text"
            ).strip()

            native_text_per_page[
                page_index
            ] = native_text

            # A page can also need OCR if it's short on native text BUT
            # contains embedded raster images (e.g. a scanned figure
            # dropped into an otherwise-text page, or a page that's
            # entirely one big image with no text layer at all).
            has_images = bool(page.get_images(full=True))

            if (
                len(native_text) < self.min_chars_for_native_text
                and (has_images or len(native_text) == 0)
            ):

                pages_needing_ocr.append(
                    page_index
                )

        ocr_images = {}

        if pages_needing_ocr:

            logger.info(
                f"{len(pages_needing_ocr)} "
                f"of {len(doc)} page(s) "
                f"look scanned. "
                f"Running OCR."
            )

            rendered = convert_from_path(
                file_path,
                dpi=self.dpi,
                first_page=min(
                    pages_needing_ocr
                ) + 1,
                last_page=max(
                    pages_needing_ocr
                ) + 1
            )

            offset = min(
                pages_needing_ocr
            )

            for idx in pages_needing_ocr:

                ocr_images[idx] = (
                    rendered[
                        idx - offset
                    ]
                )

        for page_index in range(
            len(doc)
        ):

            if page_index in ocr_images:

                proc_img = (
                    self._preprocess_image(
                        ocr_images[
                            page_index
                        ]
                    )
                )

                page_text, page_conf, method = self._best_ocr_pass(proc_img)
                page_text = self._clean_text(page_text)

            else:

                page_text = (
                    native_text_per_page[
                        page_index
                    ]
                )

                method = (
                    "native_pdf_text"
                )

            all_text_chunks.append(
                page_text
            )

            result.pages.append({
                "page": page_index + 1,
                "text": page_text,
                "method": method
            })

        result.text = (
            "\n\n".join(
                all_text_chunks
            )
        )

        result.metadata["num_pages"] = (
            len(doc)
        )

        result.metadata["ocr_pages"] = [
            p + 1
            for p in pages_needing_ocr
        ]

        result.metadata["char_count"] = (
            len(result.text)
        )

        doc.close()

        if extract_tables:

            result.tables.extend(
                self._extract_tables_from_pdf(
                    file_path,
                    pages_needing_ocr
                )
            )

        return result

    def _extract_tables_from_pdf(
        self,
        file_path: str,
        ocr_page_indices: List[int]
    ) -> List[pd.DataFrame]:

        tables = []

        try:

            with pdfplumber.open(
                file_path
            ) as pdf:

                for i, page in enumerate(
                    pdf.pages
                ):

                    if i in ocr_page_indices:
                        continue

                    for raw_table in (
                        page.extract_tables()
                    ):

                        if (
                            raw_table
                            and len(raw_table) > 1
                        ):

                            df = pd.DataFrame(
                                raw_table[1:],
                                columns=raw_table[0]
                            )

                            tables.append(df)

        except Exception as e:

            logger.warning(
                f"pdfplumber table "
                f"extraction failed: {e}"
            )

        if (
            TABLE_EXTRACTION_AVAILABLE
            and ocr_page_indices
        ):

            try:

                from img2table.document import (
                    PDF as Img2TablePDF
                )

                doc = Img2TablePDF(
                    file_path,
                    pages=ocr_page_indices
                )

                extracted = doc.extract_tables(
                    ocr=self.table_ocr_engine,
                    implicit_rows=True,
                    borderless_tables=True,
                    min_confidence=50
                )

                for _, page_tables in (
                    extracted.items()
                ):

                    for table in page_tables:
                        tables.append(
                            table.df
                        )

            except Exception as e:

                logger.warning(
                    f"img2table PDF table "
                    f"extraction failed: {e}"
                )

        logger.info(
            f"Found {len(tables)} table(s) "
            f"in PDF {file_path}"
        )

        return tables


BASE_DIR = Path(
    __file__
).resolve().parent

IMAGE_FILE = (
    BASE_DIR / "imginput.png"
)

PDF_FILE = (
    BASE_DIR / "pdfinput.pdf"
)

TEXT_FILE = (
    BASE_DIR / "textinput.txt"
)

OUTPUT_FILE = (
    BASE_DIR / "input.txt"
)


def run_ocr(input_type: str) -> str:

    input_type = input_type.lower().strip()

    if input_type == "image":

        file_path = IMAGE_FILE

    elif input_type == "pdf":

        file_path = PDF_FILE

    elif input_type == "text":

        file_path = TEXT_FILE

    else:

        raise ValueError(
            "Invalid input type. "
            "Use 'image', 'pdf', or 'text'."
        )

    if not file_path.exists():

        raise FileNotFoundError(
            f"Input file not found: {file_path}"
        )

    logger.info(
        f"Input type: {input_type}"
    )

    logger.info(
        f"Input file: {file_path}"
    )

    extractor = UniversalExtractor()

    result = extractor.process(
        str(file_path),
        extract_tables=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            result.text
        )

    logger.info(
        f"Extracted text written to: "
        f"{OUTPUT_FILE}"
    )

    return result.text


def main():

    input_type = None

    if "--type" in sys.argv:

        index = sys.argv.index(
            "--type"
        )

        if index + 1 < len(
            sys.argv
        ):

            input_type = (
                sys.argv[
                    index + 1
                ].lower()
            )

    if input_type is None:

        if len(sys.argv) >= 2:

            input_type = (
                sys.argv[1].lower()
            )

        else:

            print(
                "Usage:"
            )

            print(
                "python ocr.py --type text"
            )

            print(
                "python ocr.py --type pdf"
            )

            print(
                "python ocr.py --type image"
            )

            sys.exit(1)

    try:

        run_ocr(
            input_type
        )

    except Exception as e:

        logger.error(
            f"OCR failed: {e}"
        )

        sys.exit(1)


if __name__ == "__main__":
    main()