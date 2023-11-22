import os
import json
import pandas as pd
import datetime
import re
#vader model analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from vader_sentences_processing import SentimentIntensityFromFile, Sentence
#graphs
import matplotlib.pyplot as plt
import seaborn as sns
#wordclouds
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
                 
#custom parameters for visualizations
custom_params = {"axes.spines.right": False, "axes.spines.top": False,
                'axes.facecolor':'whitesmoke'}
sns.set_theme( rc=custom_params)



# Loading the files from Premier League
# We assume that we have all the transcriptions within the 'transcriptions' folder in the same path as the script/notebook. The naming of folders and files is consistent with the original SoccerNet dataset.

file_number=0
files_data = []

for directory, dirnames, filenames in os.walk(r'transcriptions\england_epl'):
    if filenames==[]:
        continue
    if 'other' in directory:
        continue

    for i in filenames:
        if i.endswith('.csv'):
            continue
        if file_number%25==0:
            print(os.path.join(directory,i))

        file_number += 1
        curr_file = SentimentIntensityFromFile(filename=i, path=directory)
        curr_file.load_json()
        curr_file.get_sentiments_from_sentences()
        files_data.append(curr_file)


# From the sentences uploaded from the .json files, we extract the necessary information for data exploration 
# and save it in the structured form of the dataframe.
start_time = []
end_time = []
sentences_text = []
sentiment_positive = []
sentiment_negative = []
sentiment_neutral = []
sentiment_compound = []
folder_names = []
languages_orig = []

for file in files_data:    
    for sent in file.sentences:
        start_time.append(sent.start_time)
        end_time.append(sent.end_time)
        sentences_text.append(sent.text)
        sentiment_positive.append(sent.positive)
        sentiment_negative.append(sent.negative)
        sentiment_neutral.append(sent.neutral)
        sentiment_compound.append(sent.compound)
        folder_names.append(file.folder_name)

df_eng = pd.DataFrame({'MATCH_FOLDER':folder_names, 'START_TIME':start_time,
                    'END_TIME':end_time, 'TEXT':sentences_text,
                    'POSITIVE':sentiment_positive,
                    'NEGATIVE':sentiment_negative,
                    'NEUTRAL':sentiment_neutral,
                    'COMPOUND':sentiment_compound})


#saving the dataframe for future analysis
df_eng.to_csv(r'transcriptions\england_epl\england_epl_sentences_vader.csv', sep=';')

# extracting a column which indicates the sentiment which has the highest intensity
df_eng['SENTIMENT'] = df_eng[['POSITIVE','NEGATIVE','NEUTRAL']].idxmax(axis=1)



# # Premier League - exploration of Vader model performance

# Examples of the comments
# Comments classified as positive with the highest "positive" intensity.
df_eng.loc[df_eng['SENTIMENT']=='POSITIVE'].sort_values(by='POSITIVE', ascending=False).head(30)[['TEXT','POSITIVE','NEUTRAL','NEGATIVE','COMPOUND']]

# Comments classified as positive with the lowest "positive" intensity.
df_eng.loc[df_eng['SENTIMENT']=='POSITIVE'].sort_values(by='POSITIVE', ascending=True).head(30)[['TEXT','POSITIVE','NEUTRAL','NEGATIVE','COMPOUND']]

# Comments classified as negative with the highest "negative" intensity.
df_eng.loc[df_eng['SENTIMENT']=='NEGATIVE'].sort_values(by='NEGATIVE', ascending=False).head(30)[['TEXT','POSITIVE','NEUTRAL','NEGATIVE','COMPOUND']]

# Comments classified as negative with the lowest "negative" intensity.
df_eng.loc[df_eng['SENTIMENT']=='NEGATIVE'].sort_values(by='NEGATIVE', ascending=True).head(30)[['TEXT','POSITIVE','NEUTRAL','NEGATIVE','COMPOUND']]

# Comments classified as neutral with the highest "neutral" intensity.
df_eng.loc[df_eng['SENTIMENT']=='NEUTRAL'].sort_values(by='NEUTRAL', ascending=False).head(15)[['TEXT','POSITIVE','NEUTRAL','NEGATIVE']]




# Histograms of the intensity of each sentiment
fig,axs = plt.subplots(1,3,figsize=(20,6))

axs[0].hist(df_eng['POSITIVE'], bins=20)
axs[0].set_ylabel('Frequency', fontweight='semibold')
axs[0].set_title('Positive value', fontweight='semibold')

axs[1].hist(df_eng['NEGATIVE'], bins=20)
axs[1].set_title('Negative value', fontweight='semibold')

axs[2].hist(df_eng['NEUTRAL'], bins=20)
axs[2].set_title('Neutral value', fontweight='semibold')

plt.suptitle('Histograms of the values returned by VADER', fontsize=14, fontweight='bold')
plt.show()


#Histogram of the intensity of the "Compound" intensity
plt.figure(figsize=(12,6))

plt.hist(df_eng['COMPOUND'], bins=20)
plt.title('Histogram of the Compound values returned by VADER', fontsize=14, fontweight='bold')
plt.show()


# Wordclouds
# Sentences classified as positive
positive_text = ' '.join(df_eng.loc[df_eng['SENTIMENT']=='POSITIVE']['TEXT'].tolist())
plt.figure(figsize=(11,10))
wordcloud = WordCloud(background_color="white", font_path = 'monof55.ttf', width = 3000, height = 2000,
           max_words=500, color_func=lambda *args, **kwargs: "black").generate(positive_text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear') 
plt.axis("off")
plt.show()

# Additional condition for classifying as positive: Compound>=0.5
# Recommended in the documentation: https://vadersentiment.readthedocs.io/en/latest/pages/about_the_scoring.html
positive_text_stronger = ' '.join(df_eng.loc[(df_eng['SENTIMENT']=='POSITIVE') & (df_eng['COMPOUND']>=0.5)]['TEXT'].tolist())

plt.figure(figsize=(11,10))
wordcloud = WordCloud(background_color="white", font_path = 'monof55.ttf', width = 3000, height = 2000,
           max_words=500, color_func=lambda *args, **kwargs: "black").generate(positive_text_stronger)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear') 
plt.axis("off")
plt.show()

# Sentences classified as negative
negative_text = ' '.join(df_eng.loc[df_eng['SENTIMENT']=='NEGATIVE']['TEXT'].tolist())

plt.figure(figsize=(11,10))
wordcloud = WordCloud(background_color="white", font_path = 'monof55.ttf', width = 3000, height = 2000,
           max_words=500, color_func=lambda *args, **kwargs: "black").generate(negative_text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear') 
plt.axis("off")
plt.show()

# Additional condition for classifying as positive: Compound<=-0.5
# Recommended in the documentation: https://vadersentiment.readthedocs.io/en/latest/pages/about_the_scoring.html
df_eng.loc[(df_eng['SENTIMENT']=='NEGATIVE') & (df_eng['COMPOUND']<=-0.5)].shape

negative_text_stronger = ' '.join(df_eng.loc[(df_eng['SENTIMENT']=='NEGATIVE') & (df_eng['COMPOUND']<=-0.5)]['TEXT'].tolist())

plt.figure(figsize=(11,10))
wordcloud = WordCloud(background_color="white", font_path = 'monof55.ttf', width = 3000, height = 2000,
           max_words=500, color_func=lambda *args, **kwargs: "black").generate(negative_text_stronger)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear') 
plt.axis("off")
plt.show()

#number of such sentences
df_eng.loc[(df_eng['SENTIMENT']=='NEGATIVE') & (df_eng['NEGATIVE']>0.6)].shape
