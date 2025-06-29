from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer

model = GLiClassModel.from_pretrained("models/checkpoint-3000")
tokenizer = AutoTokenizer.from_pretrained("models/checkpoint-3000")

pipeline = ZeroShotClassificationPipeline(model, tokenizer, classification_type='multi-label', device='cuda:0')

text = "Beneath the ice, ancient microbes awaken, rewriting what we know about life."
text_2 = "A single photograph changed her destiny — and the course of an entire expedition."

labels = ["science", "A story uncovering hidden ecosystems and the boundaries of biology."]
labels_2 = ["adventure", "A tale of ambition, discovery, and the untamed beauty of the unknown."]

results = pipeline([text, text_2], [labels, labels_2], threshold=0.01)[0] #because we have one text

for result in results:
 print(result["label"], "=>", result["score"])

# # head = model.model.cross_encoder_head.save_pretrained("models/head")
# # print(head)
# from gliclass.cross_encoder_heads.models.deberta_v2 import DebertaV2CrossEncoderHead

# head = DebertaV2CrossEncoderHead.from_pretrained("models/head")
# print(head)