# Visualizations

**Usage of functions with example visualizations is presented in the `visualizations.ipynb` file.**

The file visualizations.py provides 3 main functions to generate plots.

1. plot_aspect_sentiment_barplot
```
plot_aspect_sentiment_barplot(
    dataframe,
    by_column="keywords_sentiment",
    percentage=True,
    top_n=10,
    top_based_on="sum",
    based_on_percentage=False,
    min_n_to_consider=1,
)
```
The function plots sentiment by keywords or found named entities - you can specify it with the by_column parameter. You can also decide whether you want to use the number of news or normalize it to 100% with a percentage parameter. It is also possible to select the number of keywords/entities to plot (top_n parameter) and specify how these keywords/entities should be chosen. If top_based_on is "sum", then the top is chosen by the number of articles with a given keyword/entity. If top_based_on is set to int, then the algorithm chooses based on the most negative, neutral or positive sentiments. If you want to find keywords or entities that have, for example, the most often negative sentiment, you can set based_on_percentaed to True, and with parameter min_n_to_consider, you can specify how many at least news should have given keyword/entity.

2. plot_sentiment_barplot

```
plot_sentiment_barplot(
    dataframe,
    by_column="categories",
    percentage=True,
    top_n=None
)
```

The function plot_sentiment_barplot generates plots with overall articles' sentiments and groups them by selected column - for instance, it can be grouped by categories or by the authors. Percentage and top_n parameters work in the same way as for the first described function.

3. plot_sentiment_over_time

```
plot_sentiment_over_time(
    dataframe,
    interval_len=1,
    percentage=True,
    cut_last_interval=False,
    sentiments=["Negative", "Neutral", "Positive"],
)
```
The function plots sentiment within news depending on the time. With parameter interval_len, you can specify how many days should be in one time period (on the x-axis) for which sentiment is aggregated. It happens that the last days are shorter than other intervals, whose lengths are specified by interval_len. If you want to omit such a situation on your plot, you can use the cut_last_interval parameter and set it to True. The percentage parameter allows you to normalize data to 100% for every time period.
