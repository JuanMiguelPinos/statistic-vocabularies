import pandas as pd

from src.config import PROJECT_ROOT


DOMAINS = {
    "1": "Agriculture, forestry and fisheries",
    "2": "Business, industry and trade",
    "3": "Crime, justice and safety",
    "4": "Demography and population",
    "5": "Economy and national accounts",
    "6": "Education and training",
    "7": "Energy",
    "8": "Environment and climate",
    "9": "Government and public finance",
    "10": "Health",
    "11": "Housing and construction",
    "12": "Income, poverty and living conditions",
    "13": "Labour market",
    "14": "Other or multidisciplinary",
    "15": "Science, technology and digital society",
    "16": "Tourism",
    "17": "Transport and mobility",
}


def print_domain_menu() -> None:
    print("Available domains:")

    for number, domain in DOMAINS.items():
        print(f"{number:>2} = {domain}")

    print()
    print("Controls:")
    print("Enter = accept the predicted domain")
    print("? = display the domain list")
    print("p = return to the previous row")
    print("q = save and exit")


def save_progress(
    sample: pd.DataFrame,
    sample_path,
) -> None:
    sample.to_csv(
        sample_path,
        index=False,
        encoding="utf-8-sig",
    )


def update_correctness(
    sample: pd.DataFrame,
    row_index: int,
) -> None:
    predicted = str(
        sample.at[row_index, "predicted_domain"]
    ).strip()

    gold = str(
        sample.at[row_index, "gold_domain"]
    ).strip()

    sample.at[row_index, "is_correct"] = (
        "yes"
        if predicted == gold
        else "no"
    )


def main() -> None:
    sample_path = (
        PROJECT_ROOT
        / "evaluation"
        / "domain_gold_sample.csv"
    )

    if not sample_path.exists():
        raise FileNotFoundError(
            "evaluation/domain_gold_sample.csv was not found. "
            "Run python prepare_domain_evaluation.py first."
        )

    sample = pd.read_csv(
        sample_path,
        keep_default_na=False,
    )

    required_columns = {
        "sample_id",
        "term",
        "predicted_domain",
        "domain_score",
        "second_domain",
        "confidence",
        "reason",
        "gold_domain",
        "is_correct",
        "notes",
    }

    missing_columns = required_columns.difference(
        sample.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing columns in domain_gold_sample.csv: "
            + ", ".join(sorted(missing_columns))
        )

    print("Manual domain annotation")
    print()
    print_domain_menu()

    annotated_mask = (
        sample["gold_domain"]
        .astype(str)
        .str.strip()
        .ne("")
    )

    annotated_count = int(
        annotated_mask.sum()
    )

    print(
        f"\nExisting progress: "
        f"{annotated_count}/{len(sample)}"
    )

    if annotated_count == len(sample):
        print("The sample is already fully annotated.")
        return

    unannotated_indices = sample.index[
        ~annotated_mask
    ].tolist()

    current_position = unannotated_indices[0]

    while 0 <= current_position < len(sample):
        row = sample.loc[current_position]

        print(
            f"\nSample {row['sample_id']} "
            f"of {len(sample)}"
        )
        print(f"Term: {row['term']}")
        print(
            f"Predicted domain: "
            f"{row['predicted_domain']}"
        )
        print(
            f"Second domain: "
            f"{row['second_domain']}"
        )
        print(f"Confidence: {row['confidence']}")
        print(f"Score: {row['domain_score']}")
        print(f"Reason: {row['reason']}")

        existing_gold = str(
            row["gold_domain"]
        ).strip()

        if existing_gold:
            print(
                f"Current gold domain: "
                f"{existing_gold}"
            )

        answer = input(
            "Domain [1-17, Enter, ?, p, q]: "
        ).strip().casefold()

        if answer == "q":
            save_progress(
                sample,
                sample_path,
            )

            completed = int(
                sample["gold_domain"]
                .astype(str)
                .str.strip()
                .ne("")
                .sum()
            )

            print(
                f"Progress saved: "
                f"{completed}/{len(sample)}"
            )
            return

        if answer == "?":
            print_domain_menu()
            continue

        if answer == "p":
            if current_position > 0:
                current_position -= 1
            else:
                print("This is the first row.")

            continue

        if answer == "":
            selected_domain = str(
                row["predicted_domain"]
            ).strip()

        elif answer in DOMAINS:
            selected_domain = DOMAINS[answer]

        else:
            print(
                "Invalid option. Enter a number from 1 to 17, "
                "p, q, ?, or press Enter."
            )
            continue

        sample.at[
            current_position,
            "gold_domain",
        ] = selected_domain

        update_correctness(
            sample=sample,
            row_index=current_position,
        )

        save_progress(
            sample,
            sample_path,
        )

        current_position += 1

    completed = int(
        sample["gold_domain"]
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )

    print(
        f"Annotation completed: "
        f"{completed}/{len(sample)}"
    )
    print(f"Sample saved to: {sample_path}")


if __name__ == "__main__":
    main()