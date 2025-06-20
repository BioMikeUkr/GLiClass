from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer

model = GLiClassModel.from_pretrained("models/checkpoint-4")
tokenizer = AutoTokenizer.from_pretrained("models/checkpoint-2")

pipeline = ZeroShotClassificationPipeline(model, tokenizer, classification_type='multi-label', device='cuda:0')

text = "One day I will see the world!"
labels = ["travel", "dreams", "sport", "science", "politics"]
labels_2 = ["travel", "dreams", "sport", "science science"]
results = pipeline([text, text], [labels, labels_2], threshold=0.01)[0] #because we have one text

for result in results:
 print(result["label"], "=>", result["score"])

# head = model.model.cross_encoder_head.save_pretrained("models/head")
# print(head)
from gliclass.cross_encoder_heads.models.deberta_v2 import DebertaV2CrossEncoderHead

head = DebertaV2CrossEncoderHead.from_pretrained("models/head")
print(head)