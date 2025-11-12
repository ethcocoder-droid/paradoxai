import os
import re

epub_dir = 'c:\\Users\\Student\\Downloads\\QuantumGodAI\\data\\extracted_epub\\OEBPS'
output_file = 'c:\\Users\\Student\\Downloads\\QuantumGodAI\\data\\gutenberg_text.txt'

def clean_html(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

with open(output_file, 'w', encoding='utf-8') as outfile:
    for filename in os.listdir(epub_dir):
        if filename.endswith('.xhtml'):
            filepath = os.path.join(epub_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as infile:
                content = infile.read()
                cleaned_content = clean_html(content)
                outfile.write(cleaned_content)
                outfile.write('\n\n')  # Add some separation between files

print(f'Extracted text to {output_file}')