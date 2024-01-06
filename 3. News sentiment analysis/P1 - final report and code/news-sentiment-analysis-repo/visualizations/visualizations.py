import ast
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_aspect_sentiment_barplot(
    dataframe,
    by_column="keywords_sentiment",
    percentage=True,
    top_n=10,
    top_based_on="sum",
    based_on_percentage=False,
    min_n_to_consider=1,
):
    # by_column : "keywords_sentiment" or "ner_sentiment"
    sentiments = ["Negative", "Neutral", "Positive"]
    aspects = dict()
    for i, row in dataframe.iterrows():
        for key in row[by_column]:
            if key not in aspects:
                aspects[key] = [0] * len(sentiments)
            aspects[key][sentiments.index(row[by_column][key])] += 1

    aspects = choose_top(
        aspects,
        n=top_n,
        based_on=top_based_on,
        based_on_percentage=based_on_percentage,
        min_n_to_consider=min_n_to_consider,
    )

    labels = list(aspects.keys())
    if percentage:
        data = np.array(list(calculate_width(aspects).values()))
    else:
        data = np.array(list(aspects.values()))
    __plot_bar(data, labels, percentage, sentiments)


def __plot_bar(data, labels, percentage, sentiments):
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap("RdYlGn")(np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    if percentage:
        ax.xaxis.set_visible(False)
        ax.set_xlim(0, np.sum(data, axis=1).max())
    else:
        max_sum = np.sum(data, axis=1).max()
        ax.set_xlim(0, max_sum * 1.05)

    for i, (colname, color) in enumerate(zip(sentiments, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5, label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = "white" if r * g * b < 0.5 else "darkgrey"
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            if percentage:
                text = str(round(c, 1)) + "%" if c > 6 else ""
            else:
                text = str(c) if c > 0.0375 * max_sum else ""
            ax.text(x, y, text, ha="center", va="center", color=text_color)
        ax.legend(
            ncol=len(sentiments),
            bbox_to_anchor=(0, 1),
            loc="lower left",
            fontsize="small",
        )


def plot_sentiment_barplot(
    dataframe, by_column="categories", percentage=True, top_n=None
):
    sentiments = ["Negative", "Neutral", "Positive"]
    if by_column == "categories":
        df = dataframe.explode(by_column)
    elif by_column == "byline":
        df = deepcopy(dataframe)
        df["byline"] = (
            df["byline"].str.split("/", expand=False).apply(lambda x: list(set(x)))
        )
        df = df.explode(by_column)

    df = df[[by_column, "sentiment"]].value_counts().reset_index()
    values = dataframe_counted_to_dict(df, by_column)
    if top_n is not None:
        values = choose_top(values, n=top_n)
    labels = list(values.keys())
    if by_column == "categories":
        labels = map_categories_to_full_names(labels)
    if percentage:
        data = np.array(list(calculate_width(values).values()))
    else:
        data = np.array(list(values.values()))
    __plot_bar(data, labels, percentage, sentiments)


def plot_sentiment_over_time(
    dataframe,
    interval_len=1,
    percentage=True,
    cut_last_interval=False,
    sentiments=["Negative", "Neutral", "Positive"],
):
    # interval_len - number of days
    sentiment_over_time = calculate_sentiment_over_time(
        dataframe,
        interval_len=interval_len,
        sentiments=sentiments,
        cut_last_interval=cut_last_interval,
    )
    labels = list(sentiment_over_time.keys())
    if percentage:
        data = np.array(list(calculate_width(sentiment_over_time).values()))
    else:
        data = np.array(list(sentiment_over_time.values()))
    category_colors = plt.get_cmap("RdYlGn")(np.linspace(0.15, 0.85, data.shape[1]))
    if data.shape[1] == 3:
        category_colors = plt.get_cmap("RdYlGn")(np.array([0.15, 0.35, 0.85]))

    fig, ax = plt.subplots(figsize=(9.2, 5))

    ax.set_xticks(list(range(len(labels))))
    ax.set_xticklabels(labels, rotation=60, ha="right")

    for i, (colname, color) in enumerate(zip(sentiments, category_colors)):
        values = data[:, i]
        ax.plot(labels, values, color=color, marker="o", label=sentiments[i])
        ax.legend(
            ncol=len(sentiments),
            bbox_to_anchor=(0, 1),
            loc="lower left",
            fontsize="small",
        )


def create_date_buckets(date_range, interval_len=1, cut_last_interval=False):
    if interval_len > 1:
        buckets = [
            date_range[idx * interval_len]
            + " - "
            + date_range[(idx + 1) * interval_len - 1]
            for idx in range(len(date_range) // interval_len)
        ]
        buckets_upper_bounds = [
            (idx + 1) * interval_len for idx in range(len(date_range) // interval_len)
        ]
    else:
        buckets = deepcopy(date_range)
        buckets_upper_bounds = [idx + 1 for idx in range(len(buckets))]
    last_idx = len(date_range) // interval_len
    if len(date_range[last_idx * interval_len :]) == 1:
        buckets += [date_range[last_idx * interval_len]]
        buckets_upper_bounds.append(len(date_range))
    elif len(date_range[last_idx * interval_len :]) > 1:
        buckets += [date_range[last_idx * interval_len] + " - " + date_range[-1]]
        buckets_upper_bounds.append(len(date_range))
    if cut_last_interval and len(buckets_upper_bounds) >= 3:
        if (
            buckets_upper_bounds[-1] - buckets_upper_bounds[-2]
            != buckets_upper_bounds[-2] - buckets_upper_bounds[-3]
        ):
            return buckets[:-1], buckets_upper_bounds[:-1]
    return buckets, buckets_upper_bounds


def calculate_sentiment_over_time(
    dataframe,
    interval_len=1,
    sentiments=None,
    cut_last_interval=False,
):
    if sentiments is None:
        sentiments = ["Negative", "Neutral", "Positive"]
    df = deepcopy(dataframe)
    df["day"] = (
        pd.to_datetime(df["versioncreated"], unit="ms", utc=True)
        .map(lambda x: x.tz_convert("Europe/Ljubljana"))
        .dt.strftime("%Y-%m-%d")
    )
    start = df["day"].min()
    end = df["day"].max()
    date_range = pd.date_range(
        datetime.strptime(start, "%Y-%m-%d"),
        datetime.strptime(end, "%Y-%m-%d"),
        freq="d",
    )
    date_range = [d.strftime("%Y-%m-%d") for d in date_range]
    buckets, buckets_upper_bounds = create_date_buckets(
        date_range, interval_len=interval_len, cut_last_interval=cut_last_interval
    )
    sentiment_over_time = dict()
    for i in range(len(buckets)):
        sentiment_over_time[buckets[i]] = [0] * len(sentiments)
        start_day = date_range[buckets_upper_bounds[i - 1]] if i > 0 else date_range[0]
        end_day = date_range[buckets_upper_bounds[i] - 1]
        df_counted = (
            df.loc[(df["day"] >= start_day) & (df["day"] <= end_day), "sentiment"]
            .value_counts()
            .reset_index()
        )
        for j in range(len(sentiments)):
            if sentiments[j] in df_counted["sentiment"].values:
                sentiment_over_time[buckets[i]][j] = df_counted.loc[
                    df_counted["sentiment"] == sentiments[j], "count"
                ].iloc[0]
    return sentiment_over_time


def choose_top(
    dict_with_values,
    n=10,
    based_on="sum",
    based_on_percentage=False,
    min_n_to_consider=1,
):
    # based_on - if int then based on numbers on based_on-th position in dict_with_values values
    if based_on == "sum":
        return dict(sorted(dict_with_values.items(), key=lambda x: -sum(x[1]))[:n])
    else:
        if based_on_percentage:
            if min_n_to_consider > 1:
                dict_with_values = {
                    key: val
                    for key, val in dict_with_values.items()
                    if val[based_on] >= min_n_to_consider
                }
            return dict(
                sorted(
                    dict_with_values.items(),
                    key=lambda x: -(x[1][based_on] / sum(x[1])),
                )[:n]
            )
        return dict(sorted(dict_with_values.items(), key=lambda x: -x[1][based_on])[:n])


def dataframe_counted_to_dict(
    df,
    column_name1,
    order=["Negative", "Neutral", "Positive"],
    column_name2="sentiment",
    column_name3="count",
):
    result = dict()
    for i, row in df.iterrows():
        if row[column_name1] not in result:
            result[row[column_name1]] = [0] * len(order)
        result[row[column_name1]][order.index(row[column_name2])] = row[column_name3]
    return result


def map_categories_to_full_names(categories):
    en_categories = {
        "AD": "Advisory",
        "AC": "Arts and Culture",
        "AS": "Around Slovenia",
        "BE": "Business, finance and economy",
        "HE": "Health, environment, science",
        "PO": "Politics",
        "RU": "Roundup",
        "SE": "Schedule of Events",
        "ST": "Sports",
    }

    return [en_categories[category] for category in categories]


def calculate_width(categories_with_sentiment_counted):
    """
    :param categories_with_sentiment_counted: example:
    {
        "AS": [52, 32, 18],
        "HE": [14, 56, 12],
        "BE": [50, 178, 33],
        "PO": [32, 166, 58],
        "RU": [41, 76, 12]
    }
    :return: example:
    {
        "AS": [50.98, 31.37, 17.65],
        "HE": [17.07, 68.29, 14.63],
        "BE": [19.15, 68.19, 12.64],
        "PO": [12.5, 64.84, 22.65],
        "RU": [31.78, 58.91, 9.31]
    }
    """
    categories_with_widths = dict()
    for key in categories_with_sentiment_counted:
        sentiments_counted = categories_with_sentiment_counted[key]
        categories_with_widths[key] = [
            num / sum(sentiments_counted) * 100 for num in sentiments_counted
        ]
    return categories_with_widths
