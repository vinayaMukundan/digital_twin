import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
from lime.lime_text import LimeTextExplainer
from sentence_transformers import SentenceTransformer, util

class VitaTwinAnalyzer:
    def __init__(self, model_path="./models/bert_mental_health"):
        print("Initializing VitaTwin Intelligence Engine...")
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        
        
        self.device = torch.device("cpu")
            
        self.model.to(self.device)
        
        self.model.eval() # Set to evaluation mode
        
        # LIME setup
        self.class_names = ['Normal', 'Risk']
        self.explainer = LimeTextExplainer(class_names=self.class_names)
        
        # New: Initialize Sentence Transformer for Insight Layer
        print("Loading Insight Layer (Sentence Transformers)...")
        self.insight_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Define clinical categories and their corresponding "Human-Readable" insights
        self.insight_library = {
            "Internalized Worthlessness": "The language patterns suggest deep feelings of being a burden or lack of self-value.",
            "Absolutist/Hopelessness": "The text shows 'all-or-nothing' thinking (e.g., 'always', 'never'), which is a common cognitive distortion in depression.",
            "Social Withdrawal": "Signals point toward a perceived lack of support or a sense of total isolation from others.",
            "High Emotional Volatility": "The vocabulary used indicates significant emotional turbulence or externalized frustration.",
            "Passive Suicidal Ideation": "The model detected dark imagery or themes related to an ending of struggle."
        }
        self.insight_keys = list(self.insight_library.keys())
        self.insight_embeddings = self.insight_model.encode(self.insight_keys, convert_to_tensor=True)
        
    def _get_clinical_insight(self, lime_words):
        """Part 4: The Insight Layer logic."""
        if not lime_words:
            return "No specific linguistic triggers identified for deeper insight."

        # Convert the top LIME words into a single string for comparison
        # e.g., "worthless heavy dark"
        query_text = " ".join([word for word, score in lime_words])
        query_embedding = self.insight_model.encode(query_text, convert_to_tensor=True)

        # Semantic similarity check
        cosine_scores = util.cos_sim(query_embedding, self.insight_embeddings)[0]
        best_match_idx = torch.argmax(cosine_scores).item()
        
        category = self.insight_keys[best_match_idx]
        return f"{category}: {self.insight_library[category]}"

    def _lime_predictor(self, texts):
        """Helper for LIME to get probabilities from raw text."""
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        return probs.cpu().numpy()

    def get_explanation(self, text):
        """Runs LIME and returns a formatted explanation sentence."""
        # num_samples=100 for speed; increase to 500 for better accuracy
        exp = self.explainer.explain_instance(
            text, 
            self._lime_predictor, 
            num_features=5, 
            num_samples=100
        )
        
        explanation_list = exp.as_list()
        # 1. Filter for positive weights and SORT by weight descending
        # LIME usually returns them sorted, but this ensures the 'top' words are truly the highest impact
        risk_terms = sorted(
            [(word, weight) for word, weight in explanation_list if weight > 0],
            key=lambda x: x[1],
            reverse=True
        )

        if not risk_terms:
            return "The model reached this conclusion based on overall sentence context rather than specific keywords."
        
        # 2. Format the words with their scores: e.g., 'word' (0.15)
        formatted_terms = [f"'{word}' ({weight:.2f})" for word, weight in risk_terms[:5]]

        # 3. Join into a natural sentence
        if len(formatted_terms) > 1:
            words_str = ", ".join(formatted_terms[:-1]) + f" and {formatted_terms[-1]}"
        else:
            words_str = formatted_terms[0]

        return f"The model identified this pattern primarily due to terms like {words_str}."
     
         
    def get_risk_factor(self, text):
        # 1. Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)

        # 2. Get Prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = F.softmax(outputs.logits, dim=1)
            
        # 3. Extract scores
        # Assuming Label 0 = Normal, Label 1 = Poisonous/Distressed
        conf_score = probabilities[0][1].item() 
        label_id = torch.argmax(probabilities).item()
        
        print("CONFIDENCE SCORE : ",conf_score)
        
        if label_id == 0:  # Predicted as NORMAL
            risk_level = "STABLE"
            insight = "No significant mental health risk detected."
        else:  # Predicted as POISONOUS/AT-RISK
            if conf_score > 0.90:
                risk_level = "CRITICAL/HIGH RISK"
                insight = "Urgent: Strong signals of severe distress. Immediate intervention advised."
            elif conf_score > 0.70:
                risk_level = "MODERATE RISK"
                insight = "Significant signs of instability. Suggest closer monitoring."
            else:
                # Even with low confidence, if BERT picks 'Poisonous', we shouldn't ignore it
                risk_level = "EVALUATION NEEDED"
                insight = "Potential distress detected, but model confidence is low. Human review required."

        # 5. Get LIME Explanation
        explanation_sentence = self.get_explanation(text)
        
        # 3. Get LIME data (Revised to get raw words for the Insight Layer)
        exp = self.explainer.explain_instance(text, self._lime_predictor, num_features=5, num_samples=100)
        explanation_list = exp.as_list()
        risk_terms = sorted([(w, s) for w, s in explanation_list if s > 0], key=lambda x: x[1], reverse=True)

        # 4. GET THE INSIGHT (Calling Part 4)
        if label_id == 1:
            clinical_insight = self._get_clinical_insight(risk_terms)
        else:
            clinical_insight = "Stable: No clinical archetypes triggered."

        # 5. Format existing LIME explanation sentence
        formatted_terms = [f"'{word}' ({weight:.2f})" for word, weight in risk_terms[:5]]
        words_str = ", ".join(formatted_terms)
        explanation_sentence = f"Pattern triggered by: {words_str}"
        
        return {
            "text": text,
            "label": "Poisonous/At-Risk" if label_id == 1 else "Normal",
            "confidence": f"{conf_score:.2%}",
            "risk_factor": risk_level,
            "insight": clinical_insight,
            "explanation": explanation_sentence
        }
 