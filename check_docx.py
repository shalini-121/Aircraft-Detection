try:
    import docx
    print(f"docx version: {docx.__version__}")
    print(f"docx location: {docx.__file__}")
except ImportError:
    print("docx NOT found")
