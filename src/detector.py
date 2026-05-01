import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
import re

class HallucinationDetector:
    def __init__(self, clip_model_id="openai/clip-vit-base-patch32", blip_model_id="Salesforce/blip-image-captioning-base", device=None):
        """
        Initializes the Hallucination Detector with CLIP and BLIP models.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading models on {self.device}...")
        
        # Load CLIP for text-image similarity and token-level attribution
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
        self.clip_model = CLIPModel.from_pretrained(clip_model_id).to(self.device)
        self.clip_model.eval()
        
        # Load BLIP for generative cross-validation (reference captioning)
        self.blip_processor = BlipProcessor.from_pretrained(blip_model_id)
        self.blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_id).to(self.device)
        self.blip_model.eval()
        print("Models loaded successfully.")

    def compute_similarity(self, image: Image.Image, text: str) -> float:
        """
        Computes global cosine similarity between an image and a text string using CLIP.
        """
        inputs = self.clip_processor(text=[text], images=image, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
        
        # Normalize embeddings to compute cosine similarity
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)
        
        # Matrix multiplication of normalized vectors yields cosine similarity
        similarity = (image_embeds @ text_embeds.T).item()
        return similarity

    def generate_reference_caption(self, image: Image.Image) -> str:
        """
        Generates a reliable reference caption directly from the visual features using BLIP.
        """
        inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.blip_model.generate(**inputs, max_length=50)
        return self.blip_processor.decode(out[0], skip_special_tokens=True)

    def analyze_hallucination(self, image: Image.Image, candidate_caption: str, global_threshold: float = 0.24, token_threshold_margin: float = 0.04):
        """
        Performs a full hallucination analysis on the candidate caption.
        
        Args:
            image: PIL Image object.
            candidate_caption: The text to verify against the image.
            global_threshold: The cosine similarity threshold below which the entire caption is deemed a hallucination.
            token_threshold_margin: Margin subtracted from global_threshold to flag specific tokens as hallucinated.
            
        Returns:
            dict: Analysis results containing scores, verdict, and explainability metrics.
        """
        # 1. Global Semantic Alignment
        global_sim = self.compute_similarity(image, candidate_caption)
        
        # 2. Generative Cross-Validation
        ref_caption = self.generate_reference_caption(image)
        
        # 3. Token-Level Attribution (Explainability)
        # Extract meaningful tokens (words) to analyze independently
        words = re.findall(r'\b\w+\b', candidate_caption.lower())
        stop_words = {"a", "an", "the", "in", "on", "at", "to", "is", "are", "was", "were", "and", "or", "of", "with", "by", "for", "it", "this", "that"}
        meaningful_words = [w for w in words if w not in stop_words]
        
        word_scores = {}
        suspicious_words = []
        
        if meaningful_words:
            # Process all words in a single batch
            inputs = self.clip_processor(text=meaningful_words, images=image, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                
            image_embeds = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)
            
            # Compute similarity of the single image against all text tokens
            similarities = (image_embeds @ text_embeds.T).squeeze(0).cpu().numpy()
            
            token_threshold = global_threshold - token_threshold_margin
            
            for word, sim in zip(meaningful_words, similarities):
                word_scores[word] = float(sim)
                if sim < token_threshold:
                    suspicious_words.append(word)
                
        # Final verdict
        is_hallucination = global_sim < global_threshold
        
        return {
            "global_similarity": global_sim,
            "is_hallucination": is_hallucination,
            "reference_caption": ref_caption,
            "word_scores": word_scores,
            "suspicious_words": list(set(suspicious_words)), # Remove duplicates
            "global_threshold": global_threshold
        }
