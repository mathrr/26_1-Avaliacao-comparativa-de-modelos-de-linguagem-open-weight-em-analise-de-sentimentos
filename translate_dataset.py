import pandas as pd
from deep_translator import GoogleTranslator
from tqdm import tqdm


def translate_dataset(input_file, output_file):
    # 1. Load the original dataset
    # Using the same column names we defined before
    df = pd.read_csv(input_file, header=None, names=['id', 'text', 'label', 'annotator'])

    translator = GoogleTranslator(source='pt', target='en')

    print(f"Starting translation of {len(df)} rows...")

    # 2. Translation Loop with Progress Bar
    translated_texts = []

    # tqdm creates a nice progress bar in your PyCharm console
    for text in tqdm(df['text'], desc="Translating"):
        try:
            # Basic cleaning: ensure it's a string and not too long
            clean_text = str(text).strip()
            if clean_text:
                translation = translator.translate(clean_text)
                translated_texts.append(translation)
            else:
                translated_texts.append("")
        except Exception as e:
            print(f"\nError translating row: {e}")
            translated_texts.append(text)  # Fallback to original on error

    # 3. Update DataFrame and Save
    df['text_en'] = translated_texts

    # We'll save all columns, but now with the English version
    # You can choose to replace the 'text' column or keep both
    df.to_csv(output_file, index=False)
    print(f"\nSuccess! Translated dataset saved as: {output_file}")


if __name__ == "__main__":
    # Install with: pip install deep-translator tqdm
    translate_dataset('MQD-1465.csv', 'MQD-1465_english.csv')