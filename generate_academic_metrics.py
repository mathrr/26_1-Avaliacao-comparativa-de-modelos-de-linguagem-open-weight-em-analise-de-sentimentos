import pandas as pd
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, f1_score

def generate_academic_metrics(results_csv: str):
    """
    Gera um relatório técnico assumindo que as predições estão limpas (positive, negative, neutral).
    """
    if not os.path.exists(results_csv):
        print(f"Erro: Arquivo {results_csv} não encontrado.")
        return

    # 1. Carregamento dos dados
    df = pd.read_csv(results_csv)
    # Remove apenas linhas onde o ground_truth ou prediction estejam vazios (NaN)
    df = df.dropna(subset=['ground_truth', 'prediction'])

    # Definição fixa das labels (ordem acadêmica padrão)
    labels = ['negative', 'neutral', 'positive']

    y_true = df['ground_truth']
    y_pred = df['prediction']

    # 2. Cálculo das Métricas
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # 3. Construção da Matriz de Confusão
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"Actual_{l}" for l in labels],
                         columns=[f"Pred_{l}" for l in labels])

    # 4. Geração do Arquivo de Saída (_results.txt)
    output_filename = results_csv.replace(".csv", "_results.txt")

    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(f"RELATÓRIO TÉCNICO: EXPERIMENTO DE SENTIMENTO\n")
        f.write(f"Arquivo analisado: {results_csv}\n")
        f.write("-" * 50 + "\n\n")

        f.write(f"MÉTRICAS GERAIS:\n")
        f.write(f"Accuracy:        {acc:.4f}\n")
        f.write(f"Precision Macro: {prec_macro:.4f}\n")
        f.write(f"F1-Score Macro:  {f1_macro:.4f}\n")
        f.write(f"F1-Score Weighted: {f1_weighted:.4f}\n\n")

        f.write(f"MÉTRICAS POR CLASSE (F1-SCORE):\n")
        # labels=labels garante que o relatório siga a ordem definida
        report_dict = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
        for label in labels:
            f.write(f"- {label.capitalize()}: {report_dict[label]['f1-score']:.4f}\n")

        f.write("\nMATRIZ DE CONFUSÃO:\n")
        f.write(cm_df.to_string())
        f.write("\n\n" + "-" * 50 + "\n")
        f.write("Fim do Relatório.")

    print(f"📊 Relatório gerado com sucesso: {output_filename}")

if __name__ == "__main__":
    csv_file = input("Nome do arquivo CSV para processar métricas: ")
    generate_academic_metrics(csv_file)