import streamlit as st
import pandas as pd
from pattern_detection1 import predict_with_confidence
import matplotlib.pyplot as plt

st.title("Распознавание графических паттернов")

uploaded_file = st.file_uploader("Загрузите CSV-файл с колонками 'time' и 'close'", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['time'] = pd.to_datetime(df['time'])
    result = predict_with_confidence(df)  # возвращает df с колонкой "pattern" или аналогичной
    st.subheader("📉 Примеры распознанных паттернов")

# Уникальные метки
    unique_patterns = result['predicted_label_conf'].unique()

   
    for label in unique_patterns:
        examples = result[result['predicted_label_conf'] == label].head(1)
        
        st.markdown(f"### 🔹 Паттерн: {label}")

        for idx, row in examples.iterrows():
            start = pd.to_datetime(row['start'])
            end = pd.to_datetime(row['end'])

            segment = df[(df['time'] >= start) & (df['time'] <= end)]

            fig, ax = plt.subplots()
            ax.plot(pd.to_datetime(segment['time']), segment['close'], marker='o')
            label = row['predicted_label_conf']
            conf = row['confidence']
            ax.set_title(f"{label} ({conf:.2f}): {start.date()} → {end.date()}")

            ax.set_xlabel("Time")
            ax.set_ylabel("Close Price")
            ax.grid(True)

            st.pyplot(fig)
            plt.close(fig)
    st.success("Паттерны распознаны. Вот фрагмент таблицы:")
    st.dataframe(result.head())

    csv = result.to_csv(index=False).encode('utf-8')
    st.download_button("Скачать размеченный файл", data=csv, file_name="patterns_result.csv", mime="text/csv")

