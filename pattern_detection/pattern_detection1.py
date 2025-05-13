import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
)
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


def predict_with_confidence(df, window_size=48, step=1, model_dir="pattern-classifier-hours-btc-m"):
        df = df.copy()
        df['time'] = pd.to_datetime(df['time'])
        series = df['close'].values
        dates = df['time'].values

        # 1. Формируем окна
        windows, start_times, end_times = [], [], []
        for i in range(0, len(series) - window_size, step):
            window = series[i:i + window_size]
            windows.append(window)
            start_times.append(dates[i])
            end_times.append(dates[i + window_size - 1])

        # 2. Нормализация
        def normalize(seq):
            arr = np.array(seq)
            return (arr - arr.mean()) / (arr.std() + 1e-6)

        texts = [" ".join([f"{v:.3f}" for v in normalize(w)]) for w in windows]
        df_texts = pd.DataFrame({"input_text": texts})

        # 3. Загрузка модели и токенизатора
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        id2label = model.config.id2label

        # 4. Токенизация
        dataset = Dataset.from_pandas(df_texts)
        dataset = dataset.map(lambda x: tokenizer(x["input_text"], truncation=True, padding="max_length", max_length=128), batched=True)
        dataset = dataset.remove_columns(["input_text"])
        dataset.set_format("torch")
        
        # 5. DataLoader
        dataloader = DataLoader(dataset, batch_size=32, collate_fn=DataCollatorWithPadding(tokenizer))

        # 6. Предсказания
        model.eval()
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        all_preds, all_confs, all_probs = [], [], []

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(model.device) for k, v in batch.items()}
                outputs = model(**batch)
                probs = F.softmax(outputs.logits, dim=1)
                confs, preds = torch.max(probs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_confs.extend(confs.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # 7. Сбор результатов
        pred_labels = [id2label[i] for i in all_preds]

        result_df = pd.DataFrame({
            "start": start_times,
            "end": end_times,
            "predicted_label": pred_labels,
            "confidence": all_confs
        })
        result_df['predicted_label_conf']=result_df['predicted_label']
        result_df.loc[result_df['confidence'] <= 0.8, 'predicted_label_conf'] = 'none'
        # Можно добавить все вероятности по классам
        prob_df = pd.DataFrame(all_probs, columns=[f"prob_{id2label[i]}" for i in range(len(id2label))])
        result_df = pd.concat([result_df, prob_df], axis=1)

        return result_df
