# Set directory for files

dir_path = "test_data/"

# Imports

import pymupdf
import pymupdf4llm

# Test reading a pdf

doc = pymupdf.open(f"{dir_path}1984_ashtiani.pdf")

# Test reading page count

page_count = doc.page_count

print(page_count)

# Test reading metadata

metadata = doc.metadata

print(metadata)

# Test reading first few pages
page = doc[7]
text = page.get_text("text")
print(text)

# Comparing output to pymupdf4llm

# Try converting doc to markdown of entire pdf

md_text = pymupdf4llm.to_markdown(f"{dir_path}1984_ashtiani.pdf")

# Read entire pdf

print(md_text)


