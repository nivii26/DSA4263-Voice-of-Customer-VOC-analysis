# Change in topics over time 
# Change in sentiments over time

# FINAL_OUTPUT_DF = ["TIME", "TEXT", "SENTIMENT", "TOPIC"]
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import numpy as np
from datetime import date
import seaborn as sns
import circlify
import argparse

def generate_wordplot(df):

    # Create container for this graph
    c = st.container()

    # Options in 'Main Topic' filter 
    topics = df['Main Topic'].unique()
    item = 'ALL'
    topics = np.append(topics,item)

    # Topic Filter
    sel_topic = c.selectbox('Select Topic', topics)
    if sel_topic != "ALL": 
        fil_df = df[df['Main Topic'] == sel_topic]  # filter
        data = fil_df['Tokenized_text']
    else:
        data = df['Tokenized_text']

    # remove the quotes and commas from text   
    data = data.str.replace(r"'", "")
    data = data.str.replace(r",", "")

    all_words = ' '.join(data)
    word_freq = pd.Series(all_words.split()).value_counts()
    # Slider
    n = c.slider('Select number for top-N words per topic:', 10, word_freq.size, 20)

    # Slider select
    top_words = word_freq[:n]
    wordcloud = WordCloud(width=800, height=500, background_color='white', stopwords=None, min_font_size=10, colormap='winter').generate_from_frequencies(top_words)
    
    # Plot wordcloud for selected topic
    plt.figure(figsize=(5,5), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    c.pyplot()
    

def generate_sentiment_frequency(df):
    c2 = st.container()

    # df['Time'] = pd.to_datetime(df['Time'], infer_datetime_format=True)
    # Options in 'Sentiment' filter 
    topics = df['Sentiment'].unique()
    item = 'BOTH'
    topics = np.append(topics,item)

    sel_sentiment = c2.selectbox('Select Sentiment', topics)

    if sel_sentiment != 'BOTH':
        fil_df = df[df['Sentiment'] == sel_sentiment]  # filter
        df = fil_df
        
        if sel_sentiment == 'positive':
            col = '#77DD77' 
        else:
            col = '#FF0000'
    else:
        col = 'blue'

    count_by_topic = df.groupby(['Main Topic'])['Main Topic'].count()
    count_by_topic = count_by_topic.to_frame()
    
    c2.write(count_by_topic)
    plt.figure(figsize=(8,8), facecolor=None)
    count_by_topic.plot(kind='barh', rot = 0, color = col)
    plt.xlabel('Count')
    plt.show()
    c2.pyplot()

    bubble(df,col)


def sentiment_distribution(df):
    c3 = st.container()

    # Convert datatype to datetime
    df['Time'] = pd.to_datetime(df['Time'], dayfirst = 'True')
    df['year'] = pd.DatetimeIndex(df['Time']).year

    year_range = df['year'].unique()
    item = 'ALL'
    year_range = np.append(year_range,item)
    
    # Select year
    sel_year = c3.selectbox('Select Year', year_range)
    
    if sel_year != 'ALL':
        fil_df = df[df['year'] == int(sel_year)]  # filter
        df = fil_df 
    
    sents = df['Sentiment'].unique().tolist()
    count = df['Sentiment'].value_counts().to_list()
    colors = ['#77DD77', '#FF0000']
    explode = (0.03,) * len(sents)
    
    # Pie plot
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))
    ax1.pie(count, colors=colors, labels=sents, autopct='%1.1f%%', pctdistance=0.75, explode=explode)

    center_circle = plt.Circle((0,0), 0.75, fc='white')
    ax1.add_artist(center_circle)

    # Bar plot
    df['Sentiment'].value_counts().plot(kind = 'bar', color = colors, rot = 0)
    fig.tight_layout()
    plt.show()
    c3.pyplot()

def generate_wordplot_per_sentiment(df):

    # Create container for this graph
    c5 = st.container()

    # Options in 'Main Topic' filter 
    topics = df['Sentiment'].unique()
    item = 'ALL'
    topics = np.append(topics,item)

    # Topic Filter
    sel_topic = c5.selectbox('Select Sentiment', topics)
    if sel_topic != "ALL": 
        fil_df = df[df['Sentiment'] == sel_topic]  # filter
        data = fil_df['Tokenized_text']

        if sel_topic == 'positive':
            c = 'YlGn'
        else:
            c = 'YlOrRd'
        
    else:
        data = df['Tokenized_text']
        c = 'BuPu'

    # remove the quotes and commas from text   
    data = data.str.replace(r"'", "")
    data = data.str.replace(r",", "")

    all_words = ' '.join(data)
    word_freq = pd.Series(all_words.split()).value_counts()
    
    # Slider
    n = c5.slider('Select number for top-N words:', 20, word_freq.size, (20,word_freq.size-20), 20)

    # Slider select
    top_words = word_freq[n[0]:n[1]]
    wordcloud = WordCloud(width=800, height=300, background_color='white', stopwords=None, min_font_size=10, colormap=c).generate_from_frequencies(top_words)
    
    # Plot wordcloud for selected topic
    plt.figure(figsize=(5,5), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    c5.pyplot()

def topics_over_time(df):
    c4 = st.container()
    
    # Convert datatype to datetime
    df['Time'] = pd.to_datetime(df['Time'], dayfirst = 'True')
    df['year'] = pd.DatetimeIndex(df['Time']).year
    df['month'] = pd.DatetimeIndex(df['Time']).month
    df['day'] = pd.DatetimeIndex(df['Time']).day
    df['year'] = df['year'].astype(str)
    
    year_options = df['year'].unique()
    
    # Year Filter
    sel_year = c4.multiselect('Select Year(s)', year_options, default = year_options)
    fil_df = df[df['year'].isin(sel_year)]  # filter
    df = fil_df
    
   # Group the data by month and Main Topic and count the number of rows in each group
    monthly_topic_count = df.groupby(['month', 'Main Topic']).size().reset_index(name='count')

    # Get a list of the unique Main Topics
    topics = monthly_topic_count['Main Topic'].unique()
    
    plt.figure(figsize=(9, 6))
    
    # Create a line graph for each Main Topic
    for topic in topics:
        topic_data = monthly_topic_count[monthly_topic_count['Main Topic'] == topic]
        plt.plot(topic_data['month'], topic_data['count'], label=topic)
        

    # Add axis labels and title
    plt.xlabel('Month')
    plt.ylabel('Number of Reviews')
    #plt.title('Reviews over the Months')
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    plt.xticks(range(1, 13), months, rotation = 45)
    # Move the legend to the right of the graph
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
    
    # Show the plot
    plt.show()
    c4.pyplot()

def bubble(data, col):
    c5 = st.container()
    
        # Get the counts for each Main Topic
    counts = data['Main Topic'].value_counts().tolist()

    # Create a hierarchy of circles
    circles = circlify.circlify(
        counts, 
        show_enclosure=True, 
        target_enclosure=circlify.Circle(x=0, y=0, r=0.9),

    )
    #color = ['red', 'blue', 'pink', 'purple', 'orange', 'yellow']

    # Create a bubble chart
    fig, ax = plt.subplots(figsize=(8, 8))
    for circle, label in zip(circles, data['Main Topic'].unique()):
        x, y, r = circle.x, circle.y, circle.r
        ax.add_patch(plt.Circle((x, y), r*1, alpha=0.4, color = col))
        ax.text(x, y, label, ha='center', va='center', fontsize=r*15)

    # Set plot limits
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    # Add labels and title
    plt.axis('off')

    # Show the plot
    plt.show()
    c5.pyplot()

# Main function
def main(file_path):
    # Load the tokenized texts from a CSV file
    #data = pd.read_csv('/Users/nnerella/Downloads/lsa_7.csv')
    data = pd.read_csv(file_path)

    # Define Streamlit app
    st.title('The Voice of Customer (VoC) Analytics Dashboard')
    st.markdown('---')

    ## Graph 1
    st.header('Sentiment Distribution')
    st.markdown('This plot displays percentage and count distribution of sentiments over the years as selected.')
    sentiment_distribution(data)
    

    ## Graph 2
    st.header('Word Distribution per Sentiment')
    st.markdown('This plot displays a wordplot for the selected sentiemnt')
    generate_wordplot_per_sentiment(data)


    ## Graph 3: Distribution of Topics
    st.header('Topic Distribution')
    st.markdown('This plot displays distribution of reviews over the topics for the sentiment selected.')
    generate_sentiment_frequency(data)

    ## Graph 4: Word Plot
    st.header('Word Distribution per Topic')
    st.markdown('This plot displays the top N words in the selected topic of interest. Larger the word size in the word plot, the more frequently the word appears in the reviews under this topic. ')
    # Generate the word plot
    generate_wordplot(data)

    ## Graph 5
    st.header('Trends in Topics over the Years')
    st.markdown('This plot displays the distribution of topics over the months of selected year(s)')
    topics_over_time(data)


if __name__ == '__main__':
    # Define and parse command-line arguments
    parser = argparse.ArgumentParser(description="Process a CSV file for analysis.")
    parser.add_argument("file_path", type=str, help="Path to CSV file.")
    args = parser.parse_args()

    # Call the main function with the file path argument
    main(args.file_path)
