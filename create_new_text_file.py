with open('downloaded/author-quote.txt', 'r') as f:
    text = f.read()

print(text[:101])

new_text = text.splitlines()
print(new_text[:1])
print(type(new_text))

new_text2 = new_text[1].split('\t')
print(new_text2)

with open('generated_text_file/quote.txt', 'w') as generated_text_file:
    generated_text_file.flush()
    for i in new_text:
        quote = i.split('\t')[1]
        quote.lower()
        generated_text_file.write(quote+'\n\n')