from transformers import AutoModel, AutoImageProcessor, AutoTokenizer
from PIL import Image
import torch.nn.functional as F
import torch


class EmbeddingService:
    """Embedding Service functionalities"""

    def __init__(self):
        # Detect device: use CUDA if available, else CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            print(f"✅ EmbeddingService using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ EmbeddingService using CPU (no CUDA). Install CUDA + faiss-gpu for faster ingestion.")

        self.embedding_model = AutoModel.from_pretrained("nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True).to(self.device)
        self.processor_embedding_model = AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5")
        self.text_tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
        self.text_model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True).to(self.device)

    def embed_image(self, image_data):
        """
        Embed an image using Nomic vision model.

        Args:
            image_data: Image path (str) or PIL Image to embed
        """
        if isinstance(image_data, str):
            image = Image.open(image_data).convert("RGB")
        else:
            image = image_data

        inputs = self.processor_embedding_model(image, return_tensors="pt")
        inputs = inputs.to(self.device)
        with torch.no_grad():
            model_output = self.embedding_model(**inputs)
            last_hidden_state = model_output.last_hidden_state
            cls_token = last_hidden_state[:, 0]
            features = F.normalize(cls_token, p=2, dim=1)
            return features.squeeze().cpu().numpy()

    def embed_text(self, text, task_type="search_document"):
        """
        Embed text using Nomic text model.

        Args:
            text: Text to embed
            task_type: Use "search_query" for queries or "search_document" for documents
        """
        prefixed_text = f"{task_type}: {text}"
        encoded = self.text_tokenizer(prefixed_text, padding=True, truncation=True, return_tensors="pt")
        encoded = encoded.to(self.device)
        try:
            with torch.no_grad():
                output = self.text_model(**encoded)
        except torch.cuda.OutOfMemoryError:
            print("⚠️ CUDA out of memory. Clearing cache and retrying on CPU for this batch...")
            torch.cuda.empty_cache()
            encoded = encoded.to("cpu")
            self.text_model = self.text_model.cpu()
            with torch.no_grad():
                output = self.text_model(**encoded)
            self.text_model = self.text_model.to(self.device)
        token_embeddings = output[0]
        attention_mask = encoded["attention_mask"]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled = F.layer_norm(pooled, normalized_shape=(pooled.shape[1],))
        features = F.normalize(pooled, p=2, dim=1)
        return features.squeeze().cpu().numpy()

# uv pip uninstall torch torchvision torchaudio
# uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# python -c "import torch; print(torch.__version__); print('CUDA Available:', torch.cuda.is_available())"
