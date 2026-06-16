import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KNeighborsClassifier

CLASSIFIER_VERSION = 4

SEED_DATA = {
    "feature_request": {
        "ai_personality": [
            "the AI should be more humanized",
            "AI feels too robotic, needs more natural conversation",
            "make the AI sound less mechanical",
            "the bot should feel more like a real interviewer",
            "AI responses feel scripted and unnatural",
            "would be better if AI had more human-like pauses",
            "interviewer should show more empathy",
            "the AI lacks warmth and personality",
            "make the tone more friendly and conversational",
            "AI should react more naturally to my answers",
        ],
        "interview_content": [
            "should have asked more questions",
            "Could have included direct questions on relevant topics",
            "can you finish interview after candidate finish answering",
            "questions were too easy for a senior role",
            "need harder technical questions",
            "should ask behavioral questions too",
            "the questions didn't match the job description",
            "would like more domain specific questions",
            "interview was too short, need more rounds",
            "add coding challenges along with verbal questions",
        ],
        "platform_features": [
            "add option to pause the interview",
            "should allow rescheduling if disconnected",
            "need a way to review my answers after",
            "please add text display of questions",
            "would be nice to choose interview difficulty",
            "let candidates pick topics they want to be asked about",
            "add a timer visible on screen",
            "allow candidates to skip a question",
            "provide a practice mode before the real interview",
            "add support for multiple languages",
        ],
    },
    "technical_issue": {
        "audio_issues": [
            "Double sounds",
            "voice was breaking up throughout the interview",
            "echo in the audio made it hard to understand",
            "AI voice was too fast to understand",
            "It was lagging so much and sometime i was not able to hear the voice",
            "microphone was not detected by the platform",
            "the audio kept cutting in and out",
            "I could hear myself echoing back",
            "AI voice volume was too low",
            "there was a buzzing noise throughout",
        ],
        "network_lag": [
            "too laggy and the ai assistant was getting stuck",
            "serin interview platform is showing network error",
            "because network issue I was left the interview in 2 minutes",
            "there was a delay between my answer and next question",
            "page took forever to load",
            "connection kept dropping during the interview",
            "the platform was extremely slow to respond",
            "kept getting timeout errors",
            "video call quality was poor due to network",
            "the interview buffered every few seconds",
        ],
        "platform_bugs": [
            "page kept refreshing during the interview",
            "camera was not working on the platform",
            "the timer was not showing correctly",
            "screen went black in the middle",
            "the submit button was not working",
            "my webcam feed was frozen",
            "the progress bar was stuck",
            "could not see the AI avatar on screen",
            "the text input box was not responding",
            "browser tab crashed during interview",
        ],
    },
    "ux_flow_issue": {
        "repetitive_questions": [
            "Some question were repetitive",
            "In the interview repeatedly same questions were asked",
            "AI asked the same thing in different words",
            "felt like the same question was rephrased three times",
            "too many overlapping questions on the same topic",
            "already answered that but it asked again",
            "questions were redundant",
            "kept circling back to the same point",
        ],
        "timing_interruption": [
            "AI interrupted me before I finished answering",
            "AI did not wait for my response",
            "7 minutes of interview was left and yet it still said it has concluded",
            "the interview ended abruptly without warning",
            "was cut off mid sentence",
            "AI moved to next question too quickly",
            "did not get enough time to think before answering",
            "the AI started talking while I was still speaking",
        ],
        "question_relevance": [
            "the AI changed question midway",
            "the follow up questions did not relate to my answer",
            "questions jumped between topics randomly",
            "the questions were not displayed on the interview",
            "follow up had nothing to do with what I said",
            "topics shifted without any logical connection",
            "question difficulty was all over the place",
            "asked about a technology I never mentioned",
        ],
        "no_acknowledgment": [
            "AI ignored what I said and moved to next question",
            "no acknowledgment of my answers",
            "felt like the AI was not listening",
            "no feedback on whether my answer was good or bad",
            "AI did not respond to what I said at all",
            "felt like talking to a wall",
            "no reaction to any of my responses",
            "the AI just moved on without any comment",
        ],
    },
    "positive_feedback": {
        "natural_experience": [
            "The interview felt very natural",
            "felt like talking to a real person",
            "really enjoyed the conversational style",
            "the AI felt surprisingly human",
            "conversation flowed very naturally",
            "did not feel like talking to a machine",
            "the AI had a very comfortable pace",
            "felt at ease throughout the interview",
            "natural back and forth like a real interview",
        ],
        "question_quality": [
            "questions were relevant to my resume",
            "great follow up questions based on my responses",
            "the AI understood my answers well",
            "questions were well structured and clear",
            "follow ups were intelligent and on point",
            "good mix of technical and behavioral questions",
            "questions were challenging but fair",
            "liked how questions built on my previous answers",
            "very relevant questions for my experience level",
        ],
        "overall_satisfaction": [
            "nice ai interview",
            "First time i gave my interview with AI and experience was mind blowing",
            "It's very genuine platform to give interview",
            "Easy to interact with and understand the questions",
            "smooth experience from start to finish",
            "very professional interview experience",
            "would recommend this to other candidates",
            "one of the best interview platforms I have used",
            "everything worked perfectly",
        ],
    },
    "interview_incomplete": {
        "disconnection": [
            "My interview got disconnected in between and around 6 minutes were still left",
            "interview was going well then a network issue arised and it did not resumed",
            "got kicked out and could not rejoin",
            "the session expired before I could finish",
            "lost connection and could not get back in",
            "disconnected twice during the interview",
            "Wi-Fi dropped and the interview did not resume",
            "call dropped and there was no reconnect option",
            "platform logged me out mid interview",
        ],
        "agent_failure": [
            "In between the interview the agent stop replying",
            "there is not agent response, Even it is not starting the conversation",
            "AI stopped responding after 5 minutes",
            "could not complete because platform crashed",
            "interview froze and I could not continue",
            "the AI just went silent and never came back",
            "agent was stuck loading and never responded",
            "waited 2 minutes but the AI did not say anything",
            "the AI froze after my third answer",
        ],
        "premature_end": [
            "interview auto submitted before time was up",
            "only answered 3 questions before it ended",
            "interview ended way too early",
            "was supposed to be 30 minutes but ended in 10",
            "the interview closed itself before I finished",
            "only got through half the questions",
            "it said time is up but I still had minutes left",
            "submitted automatically without letting me finish",
            "interview ended after just 2 questions",
        ],
    },
    "neutral_generic": {
        "positive_vague": [
            "good",
            "best",
            "nice",
            "Great Experience",
            "loved it",
            "really good",
            "excellent",
            "superb",
        ],
        "neutral_vague": [
            "it was okay",
            "decent",
            "fine",
            "not bad",
            "average experience",
            "nothing special",
            "it was alright",
            "ok",
            "not terrible",
            "it wasn't bad",
        ],
        "gratitude_only": [
            "thanks",
            "thank you",
            "thank you for the opportunity",
            "thanks for the interview",
            "appreciated",
            "thanks a lot",
        ],
    },
    "negative_generic": {
        "negative_vague": [
            "bad",
            "worst",
            "terrible",
            "awful",
            "horrible",
            "very bad",
            "pretty bad",
            "it was bad",
            "worst experience",
            "absolutely terrible",
            "pathetic",
            "rubbish",
        ],
        "mild_dissatisfaction": [
            "not good",
            "not great",
            "not impressed",
            "disappointing",
            "poor experience",
            "waste of time",
            "didn't like it",
            "okayish",
            "could be better",
            "below average",
        ],
    },
}

CATEGORY_LABELS = {
    "technical_issue": "Technical Issue",
    "ux_flow_issue": "UX / Flow Issue",
    "positive_feedback": "Positive Feedback",
    "interview_incomplete": "Interview Incomplete",
    "feature_request": "Feature Request",
    "neutral_generic": "Neutral / Generic",
    "negative_generic": "Negative / Generic",
}

CATEGORY_DESCRIPTIONS = {
    "technical_issue": "Audio, lag, network, or platform stability problems during the interview.",
    "ux_flow_issue": "Confusing flow, repetitive questions, timing issues, or display problems.",
    "positive_feedback": "Praise for the AI interview experience or platform quality.",
    "interview_incomplete": "Interview stopped, disconnected, or agent failed to respond.",
    "feature_request": (
        "Suggestions for new behavior, more interview content, or improving how "
        "human and natural the AI feels."
    ),
    "neutral_generic": "Short or vague feedback without a specific issue or praise.",
    "negative_generic": "Short or vague negative feedback expressing dissatisfaction without describing a specific problem.",
}

SUBCATEGORY_LABELS = {
    # Feature Request
    "ai_personality": "AI Personality & Humanization",
    "interview_content": "Interview Content & Questions",
    "platform_features": "Platform Features",
    # Technical Issue
    "audio_issues": "Audio & Voice Issues",
    "network_lag": "Network & Lag",
    "platform_bugs": "Platform & UI Bugs",
    # UX / Flow Issue
    "repetitive_questions": "Repetitive Questions",
    "timing_interruption": "Timing & Interruptions",
    "question_relevance": "Question Relevance & Flow",
    "no_acknowledgment": "Lack of Acknowledgment",
    # Positive Feedback
    "natural_experience": "Natural Experience",
    "question_quality": "Question Quality",
    "overall_satisfaction": "Overall Satisfaction",
    # Interview Incomplete
    "disconnection": "Disconnection",
    "agent_failure": "Agent Failure",
    "premature_end": "Premature End",
    # Neutral / Generic
    "positive_vague": "Positive but Vague",
    "neutral_vague": "Neutral / Ambivalent",
    "gratitude_only": "Gratitude Only",
    # Negative / Generic
    "negative_vague": "Negative but Vague",
    "mild_dissatisfaction": "Mild Dissatisfaction",
}

METHOD_LABELS = {
    "knn": "Distance-weighted KNN",
    "knn_low_confidence": "Distance-weighted KNN (low confidence)",
    "centroid_fallback": "Category centroid fallback",
}


def _flatten_seed_data(
    seed_data: dict[str, dict[str, list[str]]],
) -> tuple[list[str], list[str], list[str]]:
    """Flatten nested seed data into parallel lists of texts, categories,
    and subcategories."""
    texts, categories, subcategories = [], [], []
    for category, subcats in seed_data.items():
        for subcategory, reviews in subcats.items():
            for review in reviews:
                texts.append(review)
                categories.append(category)
                subcategories.append(subcategory)
    return texts, categories, subcategories


class ReviewClassifier:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", n_neighbors: int = 5):
        self.encoder = SentenceTransformer(model_name)
        self.knn = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            metric="cosine",
            weights="distance",
        )
        self.label_map: dict[str, int] = {}
        self.training_texts: list[str] = []
        self.training_labels: list[str] = []
        self.training_subcategories: list[str] = []
        self.centroids: dict[str, np.ndarray] = {}
        self.is_trained = False

    def build_from_labeled_seeds(
        self, seed_reviews: dict[str, dict[str, list[str]]]
    ) -> None:
        texts, labels, subcategories = _flatten_seed_data(seed_reviews)

        embeddings = self.encoder.encode(texts, normalize_embeddings=True)
        self.knn.fit(embeddings, labels)
        self.training_texts = texts
        self.training_labels = labels
        self.training_subcategories = subcategories
        self.label_map = {cat: i for i, cat in enumerate(seed_reviews.keys())}
        self.centroids = self._compute_centroids(embeddings, labels)
        self.is_trained = True

    def _compute_centroids(
        self, embeddings: np.ndarray, labels: list[str]
    ) -> dict[str, np.ndarray]:
        centroids: dict[str, np.ndarray] = {}
        for category in set(labels):
            category_embeddings = embeddings[[label == category for label in labels]]
            centroid = category_embeddings.mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            centroids[category] = centroid
        return centroids

    def _centroid_similarities(self, embedding: np.ndarray) -> dict[str, float]:
        return {
            category: float(np.dot(embedding[0], centroid))
            for category, centroid in self.centroids.items()
        }

    def classify(self, review: str) -> dict:
        if not self.is_trained:
            raise RuntimeError("Call build_from_labeled_seeds() first")

        embedding = self.encoder.encode([review], normalize_embeddings=True)

        knn_prediction = self.knn.predict(embedding)[0]
        probabilities = self.knn.predict_proba(embedding)[0]
        knn_confidence = float(max(probabilities))

        class_order = list(self.knn.classes_)
        prob_by_category = {
            category: float(prob)
            for category, prob in zip(class_order, probabilities)
        }

        centroid_sims = self._centroid_similarities(embedding)
        centroid_ranking = sorted(
            centroid_sims.items(), key=lambda item: item[1], reverse=True
        )
        best_centroid_category, best_centroid_score = centroid_ranking[0]

        method = "knn"
        prediction = knn_prediction
        confidence = knn_confidence

        if knn_confidence < 0.5 and best_centroid_score > knn_confidence:
            prediction = best_centroid_category
            confidence = best_centroid_score
            method = "centroid_fallback"
        elif knn_confidence < 0.5:
            method = "knn_low_confidence"

        distances, indices = self.knn.kneighbors(embedding)
        neighbors = []
        neighbor_labels = []
        for distance, index in zip(distances[0], indices[0]):
            label = self.training_labels[index]
            subcat = self.training_subcategories[index]
            neighbor_labels.append(label)
            neighbors.append(
                {
                    "text": self.training_texts[index],
                    "category": label,
                    "category_label": CATEGORY_LABELS.get(label, label),
                    "subcategory": subcat,
                    "subcategory_label": SUBCATEGORY_LABELS.get(subcat, subcat),
                    "similarity": round(float(1 - distance), 3),
                }
            )

        neighbors.sort(key=lambda item: item["similarity"], reverse=True)

        # Determine subcategory from the closest neighbor matching the
        # predicted category.
        predicted_subcategory = None
        for neighbor in neighbors:
            if neighbor["category"] == prediction:
                predicted_subcategory = neighbor["subcategory"]
                break

        reasoning = self._build_reasoning(
            prediction=prediction,
            confidence=confidence,
            method=method,
            knn_confidence=knn_confidence,
            neighbors=neighbors,
            neighbor_labels=neighbor_labels,
            prob_by_category=prob_by_category,
            centroid_sims=centroid_sims,
            best_centroid_category=best_centroid_category,
            best_centroid_score=best_centroid_score,
            subcategory=predicted_subcategory,
        )

        return {
            "category": prediction,
            "category_label": CATEGORY_LABELS.get(prediction, prediction),
            "category_description": CATEGORY_DESCRIPTIONS.get(prediction, ""),
            "subcategory": predicted_subcategory,
            "subcategory_label": SUBCATEGORY_LABELS.get(
                predicted_subcategory, predicted_subcategory
            ),
            "confidence": round(confidence, 3),
            "needs_human_review": confidence < 0.6,
            "method": method,
            "method_label": METHOD_LABELS.get(method, method),
            "probabilities": {
                CATEGORY_LABELS.get(cat, cat): round(prob, 3)
                for cat, prob in sorted(
                    prob_by_category.items(), key=lambda item: item[1], reverse=True
                )
            },
            "centroid_similarities": {
                CATEGORY_LABELS.get(cat, cat): round(score, 3)
                for cat, score in sorted(
                    centroid_sims.items(), key=lambda item: item[1], reverse=True
                )
            },
            "nearest_neighbors": neighbors,
            "reasoning": reasoning,
        }

    def _build_reasoning(
        self,
        prediction: str,
        confidence: float,
        method: str,
        knn_confidence: float,
        neighbors: list[dict],
        neighbor_labels: list[str],
        prob_by_category: dict[str, float],
        centroid_sims: dict[str, float],
        best_centroid_category: str,
        best_centroid_score: float,
        subcategory: str | None = None,
    ) -> str:
        top_neighbor = neighbors[0]
        agreement = neighbor_labels.count(prediction)
        total_neighbors = len(neighbor_labels)
        prediction_label = CATEGORY_LABELS.get(prediction, prediction)
        subcategory_label = SUBCATEGORY_LABELS.get(subcategory, subcategory)

        lines = [
            (
                f"Your review was compared against {len(self.training_texts)} labeled seed "
                f"examples using normalized sentence embeddings and distance-weighted "
                f"k-nearest neighbors (k={self.knn.n_neighbors})."
            ),
            (
                f'The closest match is a "{top_neighbor["category_label"]}" example '
                f'with {top_neighbor["similarity"] * 100:.1f}% semantic similarity: '
                f'"{top_neighbor["text"]}".'
            ),
        ]

        if subcategory_label:
            lines.append(
                f'Within "{prediction_label}", the review best fits the '
                f'"{subcategory_label}" subcategory.'
            )

        if agreement == total_neighbors:
            lines.append(
                f"All {total_neighbors} nearest neighbors belong to the same category, "
                f"which strongly supports this classification."
            )
        elif agreement > 1:
            lines.append(
                f"{agreement} of {total_neighbors} nearest neighbors are "
                f'"{prediction_label}", giving moderate agreement.'
            )
        else:
            runner_up = sorted(
                prob_by_category.items(), key=lambda item: item[1], reverse=True
            )[1]
            lines.append(
                f"Nearest neighbors are mixed across categories. "
                f'KNN assigned {knn_confidence * 100:.1f}% to its top pick, while '
                f'"{CATEGORY_LABELS.get(runner_up[0], runner_up[0])}" '
                f"remains a close alternative ({runner_up[1] * 100:.1f}%)."
            )

        if method == "centroid_fallback":
            lines.append(
                f"Because KNN confidence was only {knn_confidence * 100:.1f}%, the model "
                f'used category centroid similarity as a tie-breaker. The '
                f'"{CATEGORY_LABELS.get(best_centroid_category, best_centroid_category)}" '
                f"centroid scored {best_centroid_score * 100:.1f}%, which is more decisive."
            )
        elif method == "knn_low_confidence":
            lines.append(
                f"KNN confidence stayed at {knn_confidence * 100:.1f}%. Centroid similarity "
                f'also favored "{CATEGORY_LABELS.get(best_centroid_category, best_centroid_category)}" '
                f"at {best_centroid_score * 100:.1f}%, but not strongly enough to override KNN."
            )

        lines.append(CATEGORY_DESCRIPTIONS.get(prediction, ""))

        if confidence < 0.6:
            lines.append(
                "Confidence is below 60%, so this review should be flagged for human review."
            )

        return " ".join(lines)

    def save(self, path: str = "review_classifier.pkl") -> None:
        with open(path, "wb") as file:
            pickle.dump(
                {
                    "version": CLASSIFIER_VERSION,
                    "knn": self.knn,
                    "label_map": self.label_map,
                    "training_texts": self.training_texts,
                    "training_labels": self.training_labels,
                    "training_subcategories": self.training_subcategories,
                    "centroids": self.centroids,
                },
                file,
            )

    def load(self, path: str = "review_classifier.pkl") -> bool:
        with open(path, "rb") as file:
            data = pickle.load(file)

        if data.get("version") != CLASSIFIER_VERSION:
            return False

        self.knn = data["knn"]
        self.label_map = data["label_map"]
        self.training_texts = data["training_texts"]
        self.training_labels = data["training_labels"]
        self.training_subcategories = data.get("training_subcategories", [])
        self.centroids = data.get("centroids", {})
        self.is_trained = True
        return True


def load_classifier() -> ReviewClassifier:
    model_path = Path("review_classifier.pkl")
    classifier = ReviewClassifier()

    if model_path.exists() and classifier.load(model_path):
        return classifier

    classifier.build_from_labeled_seeds(SEED_DATA)
    classifier.save(model_path)
    return classifier
