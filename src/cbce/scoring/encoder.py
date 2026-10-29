"""Supervised encoder baseline: a small pretrained transformer fine-tuned on labelled sentences.

The base model is DistilRoBERTa (Apache-2.0), pretrained in 2019 on OpenWebText, so it cannot
have seen the labelled data. It is fine-tuned here and never from FinBERT, whose licence
requires ShareAlike. Every training setting lives in `TrainConfig` and is fixed in advance, not
tuned on dev. Class index order is hawkish, neutral, dovish. Confidence is the winning class
probability.
"""

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from cbce.scoring.base import ScoreInput
from cbce.store.schema import Stance, StanceScore, Topic

BASE_MODEL = "distilbert/distilroberta-base"
BASE_REVISION = "fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b"

# Class index -> stance. The model's output layer follows this order.
CLASSES = (Stance.HAWKISH, Stance.NEUTRAL, Stance.DOVISH)

Predictor = Callable[[Sequence[str]], list[list[float]]]  # probabilities in CLASSES order


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float = 2e-5
    batch_size: int = 16
    epochs: int = 4
    max_length: int = 128
    weight_decay: float = 0.01
    warmup_fraction: float = 0.1
    max_grad_norm: float = 1.0
    seed: int = 0


class EncoderScorer:
    name = "encoder"

    def __init__(self, predict: Predictor) -> None:
        self._predict = predict

    def score(self, items: Sequence[ScoreInput], run_id: str) -> list[StanceScore]:
        if not items:
            return []
        scores = []
        for item, probabilities in zip(
            items, self._predict([item.text for item in items]), strict=True
        ):
            best = max(range(len(CLASSES)), key=lambda i: (probabilities[i], -i))
            scores.append(
                StanceScore(
                    sentence_id=item.sentence_id,
                    scorer=self.name,
                    topic=Topic.OTHER,
                    stance=CLASSES[best],
                    confidence=probabilities[best],
                    run_id=run_id,
                )
            )
        return scores


def fit(
    texts: Sequence[str],
    stances: Sequence[Stance],
    config: TrainConfig | None = None,
    *,
    base: str = BASE_MODEL,
    revision: str | None = BASE_REVISION,
    local_files_only: bool = True,
) -> EncoderScorer:
    """Fine-tune the base model on (text, stance) pairs and return a scorer.

    Deterministic for a fixed seed on one machine: the seed fixes initialisation of the new
    classification layer, dropout and the order of the batches.
    """
    import torch  # imported here: loading torch takes seconds and most code never needs it
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    config = config or TrainConfig()
    if len(texts) != len(stances) or not texts:
        raise ValueError("need the same, non-zero number of texts and stances")

    options = {"revision": revision, "local_files_only": local_files_only}
    torch.manual_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(base, **options)
    model = AutoModelForSequenceClassification.from_pretrained(
        base,
        num_labels=len(CLASSES),
        id2label={i: stance.value for i, stance in enumerate(CLASSES)},
        label2id={stance.value: i for i, stance in enumerate(CLASSES)},
        **options,
    )
    targets = torch.tensor([CLASSES.index(s) for s in stances])

    steps_per_epoch = math.ceil(len(texts) / config.batch_size)
    total_steps = steps_per_epoch * config.epochs
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    schedule = get_linear_schedule_with_warmup(
        optimizer, round(config.warmup_fraction * total_steps), total_steps
    )
    order = random.Random(config.seed)

    model.train()
    for _ in range(config.epochs):
        indices = list(range(len(texts)))
        order.shuffle(indices)
        for start in range(0, len(indices), config.batch_size):
            batch = indices[start : start + config.batch_size]
            encoded = tokenizer(
                [texts[i] for i in batch],
                padding=True,
                truncation=True,
                max_length=config.max_length,
                return_tensors="pt",
            )
            loss = model(**encoded, labels=targets[batch]).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
    model.eval()

    def predict(batch_texts: Sequence[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for start in range(0, len(batch_texts), 64):
            encoded = tokenizer(
                list(batch_texts[start : start + 64]),
                padding=True,
                truncation=True,
                max_length=config.max_length,
                return_tensors="pt",
            )
            with torch.no_grad():
                results.extend(torch.softmax(model(**encoded).logits, dim=-1).tolist())
        return results

    return EncoderScorer(predict)
