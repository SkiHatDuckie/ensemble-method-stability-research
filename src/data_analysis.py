# src/data_analysis.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/banknote_authentication.csv"
    df = pd.read_csv(url, header=None)
    df.columns = ["variance", "skewness", "curtosis", "entropy", "target"]

    print(df.head())
    print(df.info())

    for col in df.columns[:-1]:
        plt.figure()
        sns.histplot(data=df, x=col, hue="target", kde=True)
        plt.title(f"{col} distribution by target")
        plt.show()

    for col in df.columns[:-1]:
        plt.figure()
        sns.boxplot(x="target", y=col, data=df)
        plt.title(f"{col} vs target")
        plt.show()

    plt.figure(figsize=(8,6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

    sns.pairplot(df, hue="target")
    plt.show()
