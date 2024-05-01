from minbpe.minbpe import BasicTokenizer

tokenizer = BasicTokenizer()
with open("shakespeare.txt","r") as f:
    text = f.read()
tokenizer.train(text, 500) 
text = "Hello world I am William Shakespeare."
print(tokenizer.encode(text))
print(tokenizer.decode(tokenizer.encode(text)))
tokenizer.save("tokenizer")