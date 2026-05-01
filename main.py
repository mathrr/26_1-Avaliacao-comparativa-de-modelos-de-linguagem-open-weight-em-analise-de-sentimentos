import pandas as pd
import ollama


def run_test_drive(file_path, sample_size=5):
    # 1. Load a small sample
    # Names based on your file: index, text, label, annotator
    df = pd.read_csv(file_path, header=None, names=['id', 'text', 'label', 'annotator'], nrows=sample_size)

    # 2. Map labels to strings for the LLM
    label_mapping = {1: 'positive', -1: 'negative', 0: 'neutral'}
    df['ground_truth'] = df['label'].map(label_mapping)

    results = []

    print(f"Starting test drive with {sample_size} rows...")

    # 3. Inference Loop
    for index, row in df.iterrows():
        print(f"Processing row {index + 1}/{sample_size}...")

        response = ollama.chat(model='llama3', messages=[
            {
                'role': 'system',
                'content': 'Classify the sentiment of the text as: positive, negative or neutral. Output only the label.'
            },
            {
                'role': 'user',
                'content': row['text'],
            },
        ])

        prediction = response['message']['content'].strip().lower()
        results.append(prediction)

    df['prediction'] = results

    # 4. Quick Evaluation
    print("\n--- Preliminary Results ---")
    print(df[['text', 'ground_truth', 'prediction']])

    # Save to verify
    df.to_csv("test_drive_results.csv", index=False)
    print("\nFile 'test_drive_results.csv' generated.")


if __name__ == "__main__":
    # Ensure Ollama is running before executing this
    run_test_drive('MQD-1465.csv', sample_size=5)