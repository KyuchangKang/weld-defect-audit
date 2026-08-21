import sys
import pdfplumber

path = sys.argv[1]
start = int(sys.argv[2]) if len(sys.argv) > 2 else 1
end = int(sys.argv[3]) if len(sys.argv) > 3 else 4

with pdfplumber.open(path) as pdf:
    print('TOTAL PAGES:', len(pdf.pages))
    for i in range(start - 1, min(end, len(pdf.pages))):
        text = pdf.pages[i].extract_text() or ''
        print(f'===== PAGE {i + 1} =====')
        print(text)
