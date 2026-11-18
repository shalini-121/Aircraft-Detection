import os
import sys
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn

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
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(6)
    return p

def add_paragraph(doc, text, size=12, alignment=WD_ALIGN_PARAGRAPH.JUSTIFY, bold=False, italic=False):
    p = doc.add_paragraph()
    p.alignment = alignment
    p.paragraph_format.line_spacing = 1.5
    run = p.add_run(text)
    set_font_style(run, size=size, bold=bold, italic=italic)
    return p

def generate_report():
    doc = Document()
    
    # Page Setup
    for section in doc.sections:
        section.left_margin = Inches(1.5)
        section.right_margin = Inches(1.0)
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)

    # 1. Cover Page
    for _ in range(5): doc.add_paragraph()
    title = doc.add_paragraph()
    details = "AIRCRAFT DETECTION SYSTEM\nIN SATELLITE IMAGERY\nUSING DEEP LEARNING ARCHITECTURES"
    run = title.add_run(details)
    set_font_style(run, size=24, bold=True)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_page_break()

    # 2. Abstract
    add_heading(doc, "ABSTRACT", level=1, size=16)
    add_paragraph(doc, "This project addresses the problem of automated aircraft detection in high-resolution satellite imagery...")
    doc.add_page_break()

    # 3. Comprehensive Content (README inspired)
    add_heading(doc, "PROJECT SPECIFICATIONS", level=1, size=16)
    
    sections = [
        ("Architecture", "The system uses a two-pronged approach with Faster R-CNN and RetinaNet..."),
        ("Backbone", "ResNet-50 with FPN is utilized for robust feature extraction..."),
        ("Dataset", "UCAS-AOD provides 1,000 images with high-quality annotations...")
    ]
    for title, text in sections:
        add_heading(doc, title, level=2)
        add_paragraph(doc, text)

    # 4. Appendix: Extended Source Code
    add_heading(doc, "APPENDIX: COMPLETE SOURCE CODE", level=1, size=16)
    
    code_files = ['app.py', 'dataset.py', 'train_faster_rcnn.py', 'train_retinanet.py', 'evaluate_faster_rcnn.py', 'evaluate_retinanet.py', 'convert_ucas_to_voc.py', 'README.md']
    
    for cf in code_files:
        if os.path.exists(cf):
            add_heading(doc, f"Module: {cf}", level=2)
            with open(cf, 'r', encoding='utf-8') as f:
                content = f.read()
            # Very verbose way to add code (line by line or chunked)
            p = doc.add_paragraph()
            run = p.add_run(content)
            set_font_style(run, size=9, name='Courier New')
            doc.add_page_break()

    output = 'AIRCRAFT_DETECTION_FULL_PROJECT_REPORT.docx'
    doc.save(output)
    print(f"✅ Full report generated: {output}")

if __name__ == "__main__":
    generate_report()
