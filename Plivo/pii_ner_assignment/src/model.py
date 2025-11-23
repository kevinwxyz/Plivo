from transformers import AutoConfig, AutoModelForTokenClassification
from labels import LABEL2ID, ID2LABEL


def create_model(model_name: str, dropout: float = 0.1):
    """
    Create a token classification model with our label mapping and custom dropout.
    """

    # Load config and override dropout-related fields
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Different HF models use different names for dropout; set all that exist
    if hasattr(config, "hidden_dropout_prob"):
        config.hidden_dropout_prob = dropout
    if hasattr(config, "attention_probs_dropout_prob"):
        config.attention_probs_dropout_prob = dropout
    if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
        config.classifier_dropout = dropout

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        config=config,
    )
    return model
