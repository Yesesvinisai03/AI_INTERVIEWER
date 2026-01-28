import pdfplumber
import docx
from io import BytesIO

def extract_resume_text(uploaded_file) -> str:
    if not uploaded_file:
        return ""

    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        text = []
        file_bytes = uploaded_file.getvalue()

        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t:
                    text.append(t)

        return "\n".join(text)


    if name.endswith(".docx"):
        doc = docx.Document(BytesIO(uploaded_file.read()))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    if name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")

    return ""
