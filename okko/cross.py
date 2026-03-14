CROSS_ENCODER = "BAAI/bge-reranker-v2-m3"
CROSS_LIMIT = 16


class CrossEncoder:
    def __init__(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = "cuda:0"
        self.tokenizer = AutoTokenizer.from_pretrained(CROSS_ENCODER)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            CROSS_ENCODER, torch_dtype=torch.float16, use_safetensors=True,
        ).to(self.device).eval()

    def score(self, question, candidates):
        tokens = self.tokenizer(
            [question] * len(candidates),
            [candidate.scene.text for candidate in candidates],
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(self.device)
        with self.torch.inference_mode():
            return self.model(**tokens).logits.flatten().float().tolist()
