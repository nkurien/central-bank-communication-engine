"""FinBERT baseline (ProsusAI/finbert, CC BY-NC-SA 4.0): inference only.

Run as published: never fine-tuned here and its weights are never redistributed, because
adaptations of it must carry the same ShareAlike licence. Its three sentiment labels are
read as stance the same way as the Loughran-McDonald baseline: positive is hawkish, negative
is dovish, neutral is neutral. That mapping is a hypothesis fixed in advance, not tuned; the
baseline tests whether general finance sentiment is enough. Confidence is the winning class
probability.
"""

from collections.abc import Callable, Sequence

from cbce.scoring.base import ScoreInput
from cbce.store.schema import Stance, StanceScore, Topic

FINBERT_MODEL = "ProsusAI/finbert"
# Pinned so the baseline cannot change under us if the repository is updated.
FINBERT_REVISION = "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031"

_LABEL_TO_STANCE = {
    "positive": Stance.HAWKISH,
    "negative": Stance.DOVISH,
    "neutral": Stance.NEUTRAL,
}

# Maps texts to one {label name: probability} dict each.
Predictor = Callable[[Sequence[str]], list[dict[str, float]]]


class FinBertScorer:
    name = "finbert"

    def __init__(self, predict: Predictor) -> None:
        self._predict = predict

    @classmethod
    def load(
        cls, *, local_files_only: bool = True, batch_size: int = 32, max_length: int = 128
    ) -> "FinBertScorer":
        """Load the pinned model. Weights must already be in the local Hugging Face cache
        unless `local_files_only` is turned off for a deliberate download.
        """
        import torch  # imported here: loading torch takes seconds and most code never needs it
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        options = {"revision": FINBERT_REVISION, "local_files_only": local_files_only}
        tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL, **options)
        model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL, **options)
        model.eval()
        names = {index: label.lower() for index, label in model.config.id2label.items()}

        def predict(texts: Sequence[str]) -> list[dict[str, float]]:
            results: list[dict[str, float]] = []
            for start in range(0, len(texts), batch_size):
                batch = list(texts[start : start + batch_size])
                encoded = tokenizer(
                    batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
                )
                with torch.no_grad():
                    probabilities = torch.softmax(model(**encoded).logits, dim=-1)
                for row in probabilities.tolist():
                    results.append({names[index]: p for index, p in enumerate(row)})
            return results

        return cls(predict)

    def score(self, items: Sequence[ScoreInput], run_id: str) -> list[StanceScore]:
        if not items:
            return []
        scores = []
        for item, probabilities in zip(
            items, self._predict([item.text for item in items]), strict=True
        ):
            # Ties break on the label name so the answer never depends on dict order.
            label, probability = max(probabilities.items(), key=lambda kv: (kv[1], kv[0]))
            if label not in _LABEL_TO_STANCE:
                raise ValueError(f"unexpected FinBERT label {label!r}")
            scores.append(
                StanceScore(
                    sentence_id=item.sentence_id,
                    scorer=self.name,
                    topic=Topic.OTHER,
                    stance=_LABEL_TO_STANCE[label],
                    confidence=probability,
                    run_id=run_id,
                )
            )
        return scores
