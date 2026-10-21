# Data Folder

Place your college PDF documents here.

## Important
- Do NOT commit real confidential college documents to Git.
- The documents in this folder are **synthetic** and created only for development/testing.
- Replace them with real, authorized college documents before final submission.

## Expected files (for the evaluation questions)
- academic_regulations.pdf
- examination_guidelines.pdf
- student_handbook.pdf
- placement_guidelines.pdf
- fee_guidelines.pdf

## How to create the synthetic PDFs
Use any word processor (Word, Google Docs) to create a document with the sample
text shown in `docs/synthetic_document_content.md`, then export as PDF.

Alternatively, use the included `evaluation/make_synthetic_pdfs.py` script
(requires `reportlab`) to generate them automatically.