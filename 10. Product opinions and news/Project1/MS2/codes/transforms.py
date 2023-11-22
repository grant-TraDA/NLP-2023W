from data_structures import SentimentAnnotation, AspectAnnotation


def create_aspects(array_, aspect_column, sentiment_column):
    return [
        SentimentAnnotation(text=el[aspect_column], label=el[sentiment_column])
        for id, el in array_.iterrows()
    ]


def create_sentimented(row, text_column):
    return AspectAnnotation(text=row[text_column], aspects=row["Aspects"])


def transform_aspects(
    data_frame, id_column, text_column, aspect_column, sentiment_column
):
    true_labels = (
        data_frame.groupby([id_column, text_column])
        .apply(lambda row: create_aspects(row, aspect_column, sentiment_column))
        .reset_index(name="Aspects")
    )
    return list(true_labels.apply(lambda x: create_sentimented(x, text_column), axis=1))
