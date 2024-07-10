def cleaner(df, info=True):
    df = df.copy()
    df.columns = df.columns.str.strip()
    df.rename(columns=lambda x: x.lower(), inplace=True)
    df_duplicates = df.duplicated().any()
    df_nan = df.isna().any().any()
    df_empty = (df == "").any().any()

    if info:
        print("All columns empty spaces have been stripped.")
        print("All columns names have been converted to lowercase.\n")
        print(f"Is there any duplicates?: {df_duplicates}")
        print(f"Is there any NaN numbers?: {df_nan}")
        print(f"Is there any empty cells?: {df_empty}")

    return df


def calc_vif(x):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import pandas as pd

    vif = pd.DataFrame()
    vif["variables"] = x.columns
    vif["VIF"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]

    return vif


def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


def reg_formula(model, X):
    coefficients = model.params
    formula = f"y = {coefficients.iloc[0]:.4f}"
    for i in range(1, len(coefficients)):
        formula += f" + {coefficients.iloc[i]:.4f}*{X.columns[i]}"

    return formula


def f_histogram(
    xaxis, bins=20, kde=False, figsize=(6, 4), label=None, xlabel=None, title=None
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=figsize)
    sns.histplot(x=xaxis, bins=bins, label=label, kde=kde)
    plt.xlabel(xlabel)
    plt.title(title, size=14, fontweight="bold", ha="center")
    plt.legend()

    return plt.show()


def f_barplot(xaxis, figsize=(6, 4), xlabel=None, title=None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=figsize)
    sns.set_palette("crest")
    ax = xaxis.plot(kind="bar", width=0.8)
    for container in ax.containers:
        ax.bar_label(container)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(True)
    ax.set_frame_on(False)
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.xlabel(xlabel)
    plt.title(title, size=14, fontweight="bold", ha="center")
    plt.legend()

    return plt.show()


def f_boxplot(
    data=None,
    xaxis=None,
    yaxis=None,
    hue=None,
    figsize=(5, 3),
    showfliers=False,
    ylabel=None,
    title=None,
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=figsize)
    sns.boxplot(
        data=data,
        x=xaxis,
        y=yaxis,
        hue=hue,
        showfliers=showfliers,
        flierprops=dict(markerfacecolor="red", marker="o"),
        width=0.9,
        palette="deep",
    )
    plt.ylabel(ylabel)
    plt.title(title, size=14, fontweight="bold", ha="center")
    plt.tight_layout()
    plt.legend().remove()

    return plt.show()
