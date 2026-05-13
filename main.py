import pandas as pd
import asyncio
import time
import os
from ollama import AsyncClient

# --- CONFIGURAÇÃO HARDCODED ---
MODEL_NAME = "mistral-nemo"


async def run_sentiment_analysis():
    file_path = 'MQD-1465.csv'
    if not os.path.exists(file_path):
        print(f"Erro: Arquivo {file_path} não encontrado.")
        return

    df_full = pd.read_csv(file_path, header=None, names=['id', 'text', 'label', 'annotator'])

    print(f"--- Iniciando Experimento com {MODEL_NAME} ---")
    val = int(input(f"Dataset com {len(df_full)} frases. Quantas frases processar? (0 para todas): "))
    prompt_choice = input("Tipo de Prompt: [0] Zero-Shot | [1] Few-Shot: ")

    sample_size = len(df_full) if val <= 0 else min(val, len(df_full))
    data_to_process = df_full.head(sample_size).to_dict('records')

    valid_labels = ['positive', 'negative', 'neutral']

    constraints = (
        "Your response must be exactly one word from the set {positive, negative, neutral}. "
        "Do not include periods, explanations, or any other text. Output must be lowercase."
    )

    if prompt_choice == "1":
        prompt_type = "few-shot"
        system_prompt = (
            f"Act as a sentiment analysis assistant. {constraints}\n\n"
            "Examples:\n"
            "Text: '''This is the best service I have ever had!'''\nLabel: positive\n\n"
            "Text: '''The product arrived broken and customer support was useless.'''\nLabel: negative\n\n"
            "Text: '''The item is okay, it works as expected but nothing special.'''\nLabel: neutral"
        )
    else:
        prompt_type = "zero-shot"
        system_prompt = f"Act as a sentiment analysis assistant. Task: Classify the sentiment. {constraints}"

    clean_model_name = MODEL_NAME.replace(":", "_").replace("-", "_")
    output_filename = f"results_{clean_model_name}_{prompt_type}.csv"

    client = AsyncClient()
    results = [None] * sample_size

    print(f"\n🚀 Executando {prompt_type} em {MODEL_NAME}...")
    start_total = time.time()

    async def process_row(index, row):
        try:
            response = await client.chat(
                model=MODEL_NAME,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': f"'''{row['text']}'''"},
                ],
                options={
                    'temperature': 0.0,
                    'seed': 42,
                    'num_predict': 10  # Aumentado levemente para capturar o erro completo se houver
                }
            )
            content = response['message']['content'].strip().lower()

            # --- VERIFICAÇÃO DE RESPOSTA INVÁLIDA ---
            if content not in valid_labels:
                # Usamos id da linha (index) ou o id do CSV se existir
                id_display = row.get('id', index)
                print(f"⚠️ FORMAT ERROR [Frase {id_display}]: Recebeu '{content}'")

            results[index] = content
        except Exception as e:
            results[index] = f"error: {str(e)}"

    sem = asyncio.Semaphore(10)

    async def safe_process(index, row):
        async with sem:
            await process_row(index, row)
            # Print de progresso normal
            if (index + 1) % 100 == 0 or (index + 1) == sample_size:
                print(f"✅ {index + 1}/{sample_size} frases concluídas...")

    tasks = [safe_process(i, row) for i, row in enumerate(data_to_process)]
    await asyncio.gather(*tasks)

    total_duration = time.time() - start_total

    label_mapping = {1: 'positive', -1: 'negative', 0: 'neutral'}
    df = pd.DataFrame(data_to_process)
    df['ground_truth'] = df['label'].map(label_mapping)
    df['prediction'] = results

    df.to_csv(output_filename, index=False)
    print(f"\nConcluído! Tempo: {total_duration:.2f}s. Arquivo: {output_filename}")


if __name__ == "__main__":
    asyncio.run(run_sentiment_analysis())