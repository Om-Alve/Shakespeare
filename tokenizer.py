import json
from utils import BytePairEncoder


# To load the model
loaded_encoder = BytePairEncoder.load_model('vocab.json', 'merges.txt', 500)
text = "hey this is om alve "
print(loaded_encoder.encode(text))
print(loaded_encoder.ids_to_token(loaded_encoder.encode(text)))
print(loaded_encoder.decode(loaded_encoder.encode(text)))