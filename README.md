# MLLM Hallucination Detector 🔍

A research-grade prototype designed to detect and explain multimodal hallucinations in Vision-Language Models (VLMs). Built for Trustworthy AI research, this system evaluates the consistency between an image and an MLLM-generated caption.

## 🌟 Project Goal & Research Relevance

As Multimodal Large Language Models (MLLMs) like LLaVA, GPT-4V, and Gemini become ubiquitous, they frequently exhibit **object hallucination**—generating text that includes entities or relationships ungrounded in the visual input. 

In the context of **Trustworthy AI**, it is not enough to simply detect a hallucination; systems must provide **interpretability**. This prototype addresses both:
1. **Detection:** Assigns a cross-modal consistency score to flag ungrounded text.
2. **Explainability:** Employs token-level attribution to highlight *which* specific words in the caption are hallucinated.

## ⚙️ How It Works (Methodology)

This system leverages foundational contrastive and generative models to perform verification:

1. **Global Semantic Alignment (CLIP):** 
   Extracts dense vector embeddings for both the image and candidate caption using OpenAI's CLIP (`clip-vit-base-patch32`). The global cosine similarity in this joint latent space serves as our primary grounding metric. Low similarity indicates a mismatch.
   
2. **Generative Cross-Validation (BLIP):**
   Utilizes BLIP (`blip-image-captioning-base`) to generate a reliable, baseline reference caption directly from the visual features. This acts as an independent generative "ground truth" to cross-check the candidate caption.

3. **Token-Level Attribution:**
   To provide interpretability, the candidate caption is decomposed into meaningful tokens (filtering out stop-words). Each token is independently projected into the joint embedding space and scored against the image. Tokens falling below a dynamic threshold are flagged and visually highlighted in the UI.

## 📂 Project Structure

```
mllm-hallucination-detector/
│
├── src/
│   └── detector.py       # Core logic: CLIP/BLIP model loading, similarity computation
├── examples/             # Folder for sample images
├── app.py                # Gradio web interface
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## 🚀 Getting Started

### Prerequisites
Ensure you have Python 3.9+ installed. It is recommended to use a virtual environment.

### Installation

1. Clone this repository (or copy the files):
   ```bash
   git clone <your-repo-url>
   cd mllm-hallucination-detector
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Demo

Launch the interactive web interface:
```bash
python app.py
```
This will start a local Gradio server (usually at `http://localhost:7860`).

*Note: On the first run, the system will download the pretrained CLIP and BLIP weights from Hugging Face (~1.5GB).*

## 🧪 Usage Example

1. **Upload an Image:** E.g., an image of a dog playing in the park.
2. **Input Caption (Consistent):** "A dog running on the grass." -> *Verdict: ✅ Consistent*
3. **Input Caption (Hallucinated):** "A dog running on the grass with a frisbee in its mouth." -> *Verdict: 🚨 Hallucination Detected! (The word "frisbee" will be highlighted as suspicious).*
4. **Adjust Threshold:** Tweak the cosine similarity threshold slider to see how detection sensitivity shifts.

## 🔮 Future Research Directions

To extend this prototype into a full-scale research project, consider the following enhancements:

1. **Fine-grained Object Detection Integration:** Instead of just CLIP token similarity, integrate Grounding DINO or SAM to verify if the physical bounding boxes of entities exist.
2. **LLM-as-a-Judge:** Pass the BLIP reference caption and the candidate caption to a lightweight text-only LLM (like Llama 3 8B) to perform logical contradiction checking.
3. **Benchmarking:** Evaluate the detector on standard hallucination benchmarks like POPE (Polling-based Object Probing Evaluation) or CHAIR (Caption Hallucination Assessment with Image Relevance).
4. **Adversarial Testing:** Test against adversarial prompt injections designed to force the VLM to hallucinate.

## 📜 License
MIT License
