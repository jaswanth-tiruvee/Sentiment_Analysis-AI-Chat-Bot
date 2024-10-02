import pandas as pd
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from afinn import Afinn
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


afinn = Afinn()
analyzer = SentimentIntensityAnalyzer()

st.title('Sentimental Analysis ChatBot', anchor=None, help=None)

st.subheader('This bot uses  VADER algorithm ', divider='rainbow')


with st.expander("Let's analyze the text:"):
    text = st.text_input("Add your text to analyze:", key="text_input_key_vader")
    if text:
        scores = analyzer.polarity_scores(text)

        polarity = scores['compound']
        positive = scores['pos']
        neutral = scores['neu']
        negative = scores['neg']

       
        if polarity >= 0.05:
            category = "Positive"
        elif polarity <= -0.05:
            category = "Negative"
        else:
            category = "Neutral"

        
        st.markdown("### Analysis Results")
        st.markdown(f"**Text:** {text}")
        st.markdown(f"**Polarity Score:** {polarity}")
        st.markdown(f"**Positive:**  {positive} |  **Neutral:** {neutral}  |**Negative:** {negative}")
        st.markdown(f"**Sentiment Category:** {category}")


st.subheader('This bot uses  AFINN algorithm ', divider='rainbow')

with st.expander("Let's analyze the text:"):
    text = st.text_input("Add your text to analyze:",key="text_input_key_afinn")
    if text:
        score = afinn.score(text)
        def categorize_sentiment(score):
            if score > 5:
                return "Joy"
            elif score > 0:
                return "Positive"
            elif score == 0:
                return "Neutral"
            elif score < -5:
                return "Anger"
            else:
                return "Negative"

        category = categorize_sentiment(score)

        
        st.markdown("### Analysis Results")
        st.markdown(f"**Text:** {text}")
        st.markdown(f"**Sentiment Score:** {score}")
        st.markdown(f"**Sentiment Category:** {category}")




def sentiment_scores(sentence):
    sentiment_dict = analyzer.polarity_scores(sentence)
    return sentiment_dict


def overall_sentiment(compound):
    if compound >= 0.05:
        return 'Positive'
    elif compound <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'


def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)


def main():
    st.title('Sentiment Analysis and Visualizations of Comments')
    st.subheader("Note:", divider='rainbow')
    st.write("Asigurați-vă că fișierul este în format CSV și conține un cap de tabel.Coloana cu textul de analizat trebuie să fie denumită 'comment'.")
    st.write(" Make sure the file is in CSV format and contains a table header. The column with the text to be analysed must be named 'comment'.")
    
  
    st.subheader("Add a CSV File to Analyze:")
    file_upload = st.file_uploader("Upload a CSV File")
    
    if file_upload is not None:
        try:
            df = pd.read_csv(file_upload, sep=';', encoding="latin1")
            analyzer = SentimentIntensityAnalyzer()
            
            # Calculate sentiment scores
            df['sentiment_dict'] = df['comment'].apply(sentiment_scores)
            df['neg'] = df['sentiment_dict'].apply(lambda x: x['neg'])
            df['neu'] = df['sentiment_dict'].apply(lambda x: x['neu'])
            df['pos'] = df['sentiment_dict'].apply(lambda x: x['pos'])
            df['compound'] = df['sentiment_dict'].apply(lambda x: x['compound'])
            
            df = df.drop(columns=['sentiment_dict'])
            
         
            df['sentiment'] = df['compound'].apply(overall_sentiment)
            
            
            st.subheader('Processed Data')
            st.dataframe(df)
            
           
            csv = df.to_csv(index=False, sep=';')
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name='VaderLexicon.csv',
                mime='text/csv'
            )
            
            
            st.subheader('Visualizations')
    
            st.subheader('Sentiment Score Distribution')
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(df['compound'], bins=20, kde=True, ax=ax)
            st.pyplot(fig)
            sentiment_counts = df['sentiment'].value_counts()
            
           
            st.subheader('Sentiment Pie Chart')
            
            st.write(sentiment_counts)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  
            st.pyplot(fig)

         
            st.subheader('Word Cloud')
            all_text = ' '.join(df['comment'].tolist())
            generate_wordcloud(all_text)
            
          
           
            
        except Exception as e:
            st.error(f"Error: {e}")
    
if __name__ == "__main__":
    main()
# from textblob import TextBlob
# from dataclasses import dataclass


# @dataclass
# class Mood:
#     emoji : str
#     sentiment: float

# def get_mood(input_text: str, *, threshold: float) -> Mood:
#     sentiment : float = TextBlob(input_text).sentiment.polarity


#     friendly_threshhold: float = threshold
#     hostile_threshhold: float = -threshold


#     if sentiment >= friendly_threshhold:
#         return Mood('🙂', sentiment)
#     elif sentiment <= hostile_threshhold:
#         return Mood('☹️', sentiment)
#     else:
#          return Mood('😐', sentiment)
 

# if __name__ == '__main__':
#     while True:
#         text: str = input("Text: ")
#         # if (text.lower == "stop"):
#         #     print("Thank you for using this bot and have a nice day!")
#         #     break
#         mood: Mood = get_mood(text, threshold=0.5)
#         print(f'{mood.emoji} ({mood.sentiment})') 

