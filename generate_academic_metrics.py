import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import os


def generate_academic_metrics(results_csv: str, model_name: str):
    """
    Computes metrics and saves a summary table and a heatmap for documentation.
    """
    # 1. Load the results
    df: pd.DataFrame = pd.read_csv(results_csv)
    df = df.dropna(subset=['ground_truth', 'prediction'])

    y_true = df['ground_truth']
    y_pred = df['prediction']
    labels = ['negative', 'neutral', 'positive']

    # 2. Compute Metrics
    acc = accuracy_score(y_true, y_pred)
    # Output_dict=True allows us to manipulate the metrics as a dictionary
    report_dict = classification_report(y_true, y_pred, target_names=labels, labels=labels, output_dict=True,
                                        zero_division=0)

    # 3. Create a Summary Table (The "Documentation Gold")
    summary_data = {
        'Model': [model_name],
        'Overall_Accuracy': [round(acc, 4)],
        'Macro_F1': [round(report_dict['macro avg']['f1-score'], 4)],
        'Weighted_F1': [round(report_dict['weighted avg']['f1-score'], 4)],
        'F1_Negative': [round(report_dict['negative']['f1-score'], 4)],
        'F1_Neutral': [round(report_dict['neutral']['f1-score'], 4)],
        'F1_Positive': [round(report_dict['positive']['f1-score'], 4)]
    }

    summary_df = pd.DataFrame(summary_data)

    # Save/Append to a master results file
    summary_file = 'master_results_table.csv'
    if not os.path.isfile(summary_file):
        summary_df.to_csv(summary_file, index=False)
    else:
        summary_df.to_csv(summary_file, mode='a', header=False, index=False)

    # 4. Visualization (Confusion Matrix)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)

    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')

    image_filename = f'conf_matrix_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(image_filename, dpi=300, bbox_inches='tight')

    print(f"\n--- {model_name} Analysis Complete ---")
    print(f"Metrics saved to: {summary_file}")
    print(f"Chart saved as: {image_filename}")
    print(summary_df.to_string(index=False))  # Show the table in console


if __name__ == "__main__":
    # Example usage:
    generate_academic_metrics('test_drive_results.csv', 'Llama 3')