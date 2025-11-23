import json
import random
import argparse
from pathlib import Path

# -----------------------
# Utility vocab and helpers
# -----------------------

DIGIT_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

FIRST_NAMES = [
    "ramesh", "suresh", "john", "mary", "anita", "rahul", "kevin", "riya", "michael", "sara"
]
LAST_NAMES = [
    "sharma", "kumar", "johnson", "patel", "smith", "fernandez", "lee", "singh"
]

CITIES = [
    "mumbai", "delhi", "bangalore", "chennai", "hyderabad", "new york", "san francisco",
    "london", "dubai", "singapore"
]

LOCATIONS = [
    "downtown area", "airport road", "central station", "business district",
    "industrial area", "near city mall"
]

EMAIL_DOMAINS = ["gmail dot com", "yahoo dot com", "hotmail dot com", "outlook dot com"]


def spoken_digits(n_digits=10, group=False):
    """
    Return a string like:
    - "nine eight seven six five four three two one zero"
    or grouped: "four two four two four two four two four two four two"
    """
    digits = [random.randint(0, 9) for _ in range(n_digits)]
    words = [DIGIT_WORDS[d] for d in digits]
    if group:
        # group in 4s like credit card
        groups = []
        for i in range(0, len(words), 4):
            groups.append(" ".join(words[i:i+4]))
        return " ".join(groups)
    return " ".join(words)


def random_phone_text():
    # phone numbers often 8–12 digits spoken
    n = random.randint(8, 12)
    return spoken_digits(n_digits=n, group=False)


def random_credit_card_text():
    # 16-digit credit card, grouped in 4s
    return spoken_digits(n_digits=16, group=True)


def random_person_name():
    return random.choice(FIRST_NAMES) + " " + random.choice(LAST_NAMES)


def random_email_text():
    name = random.choice(FIRST_NAMES) + " " + random.choice(LAST_NAMES)
    # STT style: "john dot doe at gmail dot com"
    parts = name.split()
    local = " dot ".join(parts)
    domain = random.choice(EMAIL_DOMAINS)
    return f"{local} at {domain}"


def random_date_text():
    # simple noisy dates like "twenty fourth march twenty twenty three"
    day = random.randint(1, 28)
    year = random.choice(["twenty twenty", "twenty twenty one", "twenty twenty two", "twenty twenty three"])
    months = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december"
    ]
    month = random.choice(months)

    # slightly noisy speech forms
    if day in [1, 21]:
        day_text = "first"
    elif day in [2, 22]:
        day_text = "second"
    elif day in [3, 23]:
        day_text = "third"
    else:
        day_text = f"{day}th"

    return f"{day_text} {month} {year}"


def random_city():
    return random.choice(CITIES)


def random_location():
    return random.choice(LOCATIONS)


# -----------------------
# Example templates
# -----------------------

def build_example(example_id: int):
    """
    Build one synthetic STT-style example with text + entity spans.
    """
    # choose which entity types to include in this utterance
    # sometimes multiple, sometimes few
    possible_entities = ["PHONE", "EMAIL", "PERSON_NAME", "CREDIT_CARD", "DATE", "CITY", "LOCATION"]
    n_entities = random.randint(1, 4)
    chosen = random.sample(possible_entities, n_entities)

    text = ""
    entities = []

    def add_chunk(chunk_text, label=None):
        """Append chunk_text to global text, optionally record entity span."""
        nonlocal text, entities
        if not chunk_text:
            return
        # add a space if text is not empty
        if text:
            text += " "
        start = len(text)
        text += chunk_text
        end = len(text)
        if label is not None:
            entities.append({
                "start": start,
                "end": end,
                "label": label
            })

    # some random non-entity preamble
    preambles = [
        "hi this is customer calling about my account",
        "please update my details",
        "i want to place an order",
        "i need help with my card and address",
        "can you check my profile information"
    ]
    add_chunk(random.choice(preambles), label=None)

    # add entities with natural-ish templates
    for ent in chosen:
        if ent == "PHONE":
            add_chunk("my phone number is", label=None)
            add_chunk(random_phone_text(), label="PHONE")

        elif ent == "CREDIT_CARD":
            add_chunk("my credit card number is", label=None)
            add_chunk(random_credit_card_text(), label="CREDIT_CARD")

        elif ent == "EMAIL":
            add_chunk("my email is", label=None)
            add_chunk(random_email_text(), label="EMAIL")

        elif ent == "PERSON_NAME":
            add_chunk("my name is", label=None)
            add_chunk(random_person_name(), label="PERSON_NAME")

        elif ent == "DATE":
            add_chunk("the booking date is", label=None)
            add_chunk(random_date_text(), label="DATE")

        elif ent == "CITY":
            add_chunk("i live in", label=None)
            add_chunk(random_city(), label="CITY")

        elif ent == "LOCATION":
            add_chunk("the location is", label=None)
            add_chunk(random_location(), label="LOCATION")

        # some small filler between entities
        fillers = [
            "and also",
            "and",
            "further",
            "also please note",
            "in case you need it"
        ]
        if random.random() < 0.6:
            add_chunk(random.choice(fillers), label=None)

    # some optional noisy tail
    tails = [
        "that is all",
        "thank you",
        "please confirm",
        "can you repeat that back",
        "hope that is clear"
    ]
    if random.random() < 0.8:
        add_chunk(random.choice(tails), label=None)

    # basic lowercase noisy style
    text = text.lower()

    return {
        "id": f"utt_{example_id:04d}",
        "text": text,
        "entities": entities
    }


def generate_dataset(n_examples: int, out_path: Path):
    with out_path.open("w", encoding="utf8") as f:
        for i in range(n_examples):
            ex = build_example(i + 1)
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train", type=int, default=800)
    parser.add_argument("--n_dev", type=int, default=150)
    parser.add_argument("--out_dir", type=str, default="data")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.jsonl"
    dev_path = out_dir / "dev.jsonl"

    print(f"Generating {args.n_train} train examples → {train_path}")
    generate_dataset(args.n_train, train_path)

    print(f"Generating {args.n_dev} dev examples → {dev_path}")
    generate_dataset(args.n_dev, dev_path)

    print("Done.")


if __name__ == "__main__":
    main()