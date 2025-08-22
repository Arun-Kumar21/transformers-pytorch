from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")

enc1 = tokenizer.encode("Hello, how can i help you?")
enc2 = tokenizer.encode("The moon is beautiful, isn't it?")

print(enc1, enc2)
# Some encoding chars ==> (, -> 2) , (? -> 31), (<s> -> 0)