"""
Create sliced versions of audio_red_round_button_small_train_un_n100 dataset.

- n75 : first  2222 rows
- n50 : first  4697 rows
- n25 : first  7061 rows
"""

from datasets import load_dataset

SOURCE = "ramen-noodels/audio_red_round_button_small_train_un_n100"

splits = {
    "ramen-noodels/audio_red_round_button_small_train_un_n75": 2222,
    "ramen-noodels/audio_red_round_button_small_train_un_n50": 4697,
    "ramen-noodels/audio_red_round_button_small_train_un_n25": 7061,
}

def main():
    print(f"Loading full dataset from {SOURCE} ...")
    ds = load_dataset(SOURCE)["train"]
    print(f"Full dataset length: {len(ds)}")
    print(f"Columns: {ds.column_names}")

    for target_repo, length in splits.items():
        print(f"\n{'='*60}")
        print(f"Creating {target_repo} (length {length}) ...")
        sliced = ds.select(range(length))
        print(f"  Sliced to {len(sliced)} rows.")
        print(f"  Pushing to Hub ...")
        sliced.push_to_hub(target_repo, private=True)
        print(f"  Done: https://huggingface.co/datasets/{target_repo}")

    print("\nAll datasets created successfully!")

if __name__ == "__main__":
    main()