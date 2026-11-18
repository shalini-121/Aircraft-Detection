import os
import sys

def log(msg):
    print(msg)
    with open('report_debug.log', 'a') as f:
        f.write(msg + '\n')

log("Script started")

try:
    from docx import Document
    from docx.shared import Pt, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    log("docx imported successfully")
except Exception as e:
    log(f"Import failed: {e}")
    sys.exit(1)

def set_font_style(run, size=12, bold=False, italic=False, name='Times New Roman'):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    r = run._element
    r.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'), name)

def add_heading(doc, text, level=1, size=14, bold=True):
    p = doc.add_heading('', level=level)
    run = p.add_run(text)
    set_font_style(run, size=size, bold=bold)
    return p

def generate_report():
    log("Initializing Document object")
    doc = Document()
    
    log("Setting margins")
    for section in doc.sections:
        section.left_margin = Inches(1.5)
        section.right_margin = Inches(1.0)
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)

    log("Creating Cover Page")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for _ in range(3): doc.add_paragraph()
    
    title_run = p.add_run("AIRCRAFT DETECTION SYSTEM PROJECT REPORT")
    set_font_style(title_run, size=24, bold=True)
    
    doc.add_page_break()

    log("Adding Abstract")
    add_heading(doc, "ABSTRACT", level=1, size=16)
    doc.add_paragraph("This project uses Deep Learning to detect aircraft in satellite imagery.")
    
    log("Adding Appendices (Code)")
    add_heading(doc, "APPENDIX: CODE", level=1)
    
    # Just one file for now to test
    if os.path.exists('app.py'):
        log("Reading app.py")
        with open('app.py', 'r') as f:
            code = f.read()
        p = doc.add_paragraph()
        run = p.add_run(code)
        set_font_style(run, size=10, name='Courier New')

    log("Saving document")
    doc.save('AIRCRAFT_DETECTION_FINAL_REPORT.docx')
    log("Save successful")

if __name__ == "__main__":
    try:
        generate_report()
    except Exception as e:
        log(f"Execution failed: {e}")
        import traceback
        log(traceback.format_exc())
