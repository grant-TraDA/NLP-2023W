import plotly.express as px
import plotly.graph_objects as go


def prepare_barplot_words(countred_dataframe):
    fig = px.bar(countred_dataframe, x="word", y="count")

    # Check the length of the dataframe
    num_bars = len(countred_dataframe)

    # Set x-axis tick label orientation based on the number of bars
    if num_bars < 15:
        fig.update_layout(
            xaxis=dict(tickangle=0),  # horizontal tick labels
        )
    else:
        fig.update_layout(
            xaxis=dict(tickangle=-90),  # vertical tick labels
        )

    fig.update_layout(
        xaxis_title="Words",
        yaxis_title="Count",
        font=dict(size=18, color="#7f7f7f"),
        xaxis_fixedrange=True,
        yaxis_fixedrange=True,
    )

    fig.update_traces(hovertemplate="Word: <b>%{x}</b><br>Count: <b>%{y}</b><extra></extra>", marker_color="#0499D4")

    return fig
