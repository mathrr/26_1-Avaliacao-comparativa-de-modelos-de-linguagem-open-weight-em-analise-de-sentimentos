import pandas as pd
import asyncio
import time
from ollama import AsyncClient


async def run_sentiment_analysis():
    file_path = 'MQD-1465.csv'
    df_full = pd.read_csv(file_path, header=None, names=['id', 'text', 'label', 'annotator'])

    val = int(input(f"Dataset com {len(df_full)} frases. Quantas frases? (0 para todas): "))
    sample_size = len(df_full) if val <= 0 else min(val, len(df_full))

    # Otimização: Converter para lista de dicionários é mais rápido que iterrows()
    data_to_process = df_full.head(sample_size).to_dict('records')

    label_mapping = {1: 'positive', -1: 'negative', 0: 'neutral'}
    system_prompt = "[Instruction] Act as a sentiment analysis assistant. Task: classify text in triple backticks into [positive, negative, neutral]. [Constraints] Output ONLY the lowercase word. No periods or explanations."

    client = AsyncClient()
    results = [None] * sample_size
    times = []

    print(f"\nIniciando processamento ASSÍNCRONO de {sample_size} frases...")
    start_total = time.time()

    async def process_row(index, row):
        nonlocal times
        start_phrase = time.time()
        try:
            response = await client.chat(
                model='mistral-nemo',
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': f"'''{row['text']}'''"},
                ],
                options={
                    'temperature': 0.0,
                    'top_p': 1.0,
                    'num_predict': 5  # Otimização de tempo: limita a geração
                }
            )
            results[index] = response['message']['content'].strip()
        except Exception as e:
            results[index] = f"ERROR: {str(e)}"

        duration = time.time() - start_phrase
        times.append(duration)

    # Gerencia o progresso com semáforo para não sobrecarregar a VRAM da GTX 1060
    # 10 tarefas simultâneas é um bom equilíbrio para 6GB de VRAM
    sem = asyncio.Semaphore(10)

    async def safe_process(index, row):
        async with sem:
            await process_row(index, row)
            print(f"Frase {index + 1} processada com sucesso.")

    # Cria todas as tarefas e executa
    tasks = [safe_process(i, row) for i, row in enumerate(data_to_process)]
    await asyncio.gather(*tasks)

    end_total = time.time()
    total_duration = end_total - start_total

    # Finalização
    df = pd.DataFrame(data_to_process)
    df['ground_truth'] = df['label'].map(label_mapping)
    df['prediction'] = results

    print("\n" + "=" * 30)
    print(f"Tempo Total: {total_duration:.2f}s | Média: {total_duration / sample_size:.2f}s/frase")
    df.to_csv(f"results_optimized_{sample_size}.csv", index=False)


if __name__ == "__main__":
    asyncio.run(run_sentiment_analysis())