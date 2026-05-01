from PIL import Image
import re
import random

class HallucinationDetector:
    def __init__(self, clip_model_id="mock", blip_model_id="mock", device=None):
        """
        Initializes the MOCK Hallucination Detector.
        NO MODELS ARE DOWNLOADED. THIS IS FOR UI DEMONSTRATION ONLY.
        """
        print("Initializing MOCK Models. Skipping all downloads for fast UI demo...")
        self.device = "mock"

    def compute_similarity(self, image: Image.Image, text: str) -> float:
        """
        Computes a fake global cosine similarity between 0.15 and 0.35.
        """
        return random.uniform(0.15, 0.35)

    def generate_reference_caption(self, image: Image.Image) -> str:
        """
        Generates a fake reference caption.
        """
        return "A dog playing with a ball in a grassy field."

    def analyze_hallucination(self, image: Image.Image, candidate_caption: str, global_threshold: float = 0.24, token_threshold_margin: float = 0.04):
        """
        Performs a fake hallucination analysis on the candidate caption.
        """
        # 1. Global Semantic Alignment (Fake)
        global_sim = self.compute_similarity(image, candidate_caption)
        
        # 2. Generative Cross-Validation (Fake)
        ref_caption = self.generate_reference_caption(image)
        
        # 3. Token-Level Attribution (Fake)
        words = re.findall(r'\b\w+\b', candidate_caption.lower())
        stop_words = {"a", "an", "the", "in", "on", "at", "to", "is", "are", "was", "were", "and", "or", "of", "with", "by", "for", "it", "this", "that"}
        meaningful_words = [w for w in words if w not in stop_words]
        
        word_scores = {}
        suspicious_words = []
        
        token_threshold = global_threshold - token_threshold_margin
        
        # Randomly assign a low score to one of the meaningful words to demonstrate the highlighting feature
        if meaningful_words:
            bad_word_index = random.randint(0, len(meaningful_words) - 1)
            for i, word in enumerate(meaningful_words):
                if i == bad_word_index and global_sim < global_threshold:
                    # Force a hallucination for the demo
                    sim = token_threshold - 0.05
                    suspicious_words.append(word)
                else:
                    sim = token_threshold + random.uniform(0.01, 0.1)
                word_scores[word] = float(sim)
                
        is_hallucination = global_sim < global_threshold
        
        # If no words were flagged but global sim is low, flag a random word
        if is_hallucination and not suspicious_words and meaningful_words:
             suspicious_words.append(random.choice(meaningful_words))
        
        return {
            "global_similarity": global_sim,
            "is_hallucination": is_hallucination,
            "reference_caption": ref_caption,
            "word_scores": word_scores,
            "suspicious_words": list(set(suspicious_words)),
            "global_threshold": global_threshold
        }
