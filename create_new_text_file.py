with open('/media/mrugank/626CB0316CB00239/Drive E/tutorial/datascience/dataset/Quotables-master/author-quote.txt', 'r') as f:
    text = f.read()

print(text[:101])

new_text = text.splitlines()
print(new_text[:1])
print(type(new_text))

new_text2 = new_text[1].split('\t')
print(new_text2)

with open('/media/mrugank/626CB0316CB00239/for development purpose only/python/facebook pytorch/RNN/generated_text_file/quote.txt', 'w') as generated_text_file:
    generated_text_file.flush()
    for i in new_text:
        quote = i.split('\t')[1]
        quote.lower()
        generated_text_file.write(quote+'\n\n')