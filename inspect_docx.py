from docx import Document

try:
    doc = Document('Aircraft_Detection_Research_Paper.docx')
    print("Document loaded successfully.")
    
    print("\n--- Paragraphs and Styles ---")
    for i, p in enumerate(doc.paragraphs):
        if p.text.strip():
            print(f"Index {i} [Style: {p.style.name}]: {p.text[:50]}...")
            
    print("\n--- Tables ---")
    for i, table in enumerate(doc.tables):
        print(f"Table {i} has {len(table.rows)} rows and {len(table.columns)} columns.")
        # Print first row content to identify
        if len(table.rows) > 0:
            row_text = [cell.text for cell in table.rows[0].cells]
            print(f"  Header: {row_text}")

except Exception as e:
    print(f"Error: {e}")
