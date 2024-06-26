import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import io

# Downloading necessary NLTK data
nltk.download('vader_lexicon')

class SentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def transform_scale(self, score):
        return 5 * score + 5  # Convert the sentiment score from -1 to 1 scale to 0 to 10 scale

    def calculate_overall_sentiment(self, reviews):
        compound_scores = [self.sia.polarity_scores(str(review))["compound"] for review in reviews if isinstance(review, str)]
        overall_sentiment = sum(compound_scores) / len(compound_scores) if compound_scores else 0
        return self.transform_scale(overall_sentiment)

    def analyze_sentiment(self, reviews):
        sentiments = [{'compound': self.transform_scale(self.sia.polarity_scores(str(review))["compound"]),
                       'pos': self.sia.polarity_scores(str(review))["pos"],
                       'neu': self.sia.polarity_scores(str(review))["neu"],
                       'neg': self.sia.polarity_scores(str(review))["neg"]}
                      for review in reviews if isinstance(review, str)]
        return sentiments

    def analyze_periodic_sentiment(self, reviews, period):
        period_reviews = [' '.join(reviews[i:i + period]) for i in range(0, len(reviews), period)]
        return self.analyze_sentiment(period_reviews)

    def interpret_sentiment(self, sentiments):
        avg_sentiment = sum([sentiment['compound'] for sentiment in sentiments]) / len(sentiments) if sentiments else 0
        if avg_sentiment >= 6.5:
            description = "Excellent progress, keep up the good work!"
        elif avg_sentiment >= 6.2:
            description = "Good progress, continue to work hard!"
        else:
            description = "Needs improvement, stay motivated and keep trying!"

        trend = "No change"
        if len(sentiments) > 1:
            first_half_avg = sum([sentiment['compound'] for sentiment in sentiments[:len(sentiments)//2]]) / (len(sentiments)//2)
            second_half_avg = sum([sentiment['compound'] for sentiment in sentiments[len(sentiments)//2:]]) / (len(sentiments)//2)
            if second_half_avg > first_half_avg:
                trend = "Improving"
            elif second_half_avg < first_half_avg:
                trend = "Declining"

        return description, trend

# Streamlit UI setup
st.title("Student Review Sentiment Analysis")

# Upload CSV file
csv_file = st.file_uploader("Upload your CSV file")

if csv_file:
    df = pd.read_csv(io.BytesIO(csv_file.read()), encoding='utf-8')
    st.write(df.head())  # Debug statement to check the loaded data

    if "Branch" in df.columns:
        # Existing logic for handling dataset with 'Branch' column
        branches = df["Branch"].unique().tolist()
        selected_branch = st.selectbox("Select a Branch:", ["All Branches"] + branches)
        review_period = st.selectbox("Review Period:", [1, 4])

        if selected_branch != "All Branches":
            # Processing for individual branch
            branch_data = df[df["Branch"] == selected_branch]
            st.write(branch_data)  # Debug statement to check the branch_data
            reviews = branch_data.iloc[:, 5:].values.flatten().tolist()  # Assuming feedback starts from column index 5

            analyzer = SentimentAnalyzer()

            if review_period == 1:
                sentiments = analyzer.analyze_sentiment(reviews)
            else:
                sentiments = analyzer.analyze_periodic_sentiment(reviews, review_period)

            overall_sentiment = analyzer.calculate_overall_sentiment(reviews)
            st.subheader(f"Overall Sentiment for {selected_branch}: {overall_sentiment:.2f}")
            st.subheader("Sentiment Analysis")

            # Plotting sentiment
            weeks = list(range(1, len(sentiments) + 1))
            sentiment_scores = [sentiment['compound'] for sentiment in sentiments]
            pos_scores = [sentiment['pos'] for sentiment in sentiments]
            neu_scores = [sentiment['neu'] for sentiment in sentiments]
            neg_scores = [sentiment['neg'] for sentiment in sentiments]

            fig, ax = plt.subplots()
            ax.plot(weeks, sentiment_scores, label="Overall", color="blue")
            ax.fill_between(weeks, sentiment_scores, color="blue", alpha=0.1)
            ax.plot(weeks, pos_scores, label="Positive", color="green")
            ax.plot(weeks, neu_scores, label="Neutral", color="gray")
            ax.plot(weeks, neg_scores, label="Negative", color="red")

            ax.set_xlabel('Week')
            ax.set_ylabel('Sentiment Score')
            ax.set_title(f'Sentiment Analysis for {selected_branch}')
            ax.legend()
            st.pyplot(fig)

            description, trend = analyzer.interpret_sentiment(sentiments)
            st.subheader("Progress Description")
            st.write(f"Sentiment Trend: {trend}")
            st.write(f"Description: {description}")

            # Breakdown of analysis
            st.subheader("Breakdown of Analysis")
            breakdown_df = pd.DataFrame(sentiments, index=list(range(1, len(sentiments) + 1)))
            st.write(breakdown_df)

        else:
            # Processing for all branches
            analyzer = SentimentAnalyzer()
            all_branches_data = []

            for branch in branches:
                branch_reviews = df[df["Branch"] == branch].iloc[:, 5:].values.flatten().tolist()  # Assuming feedback starts from column index 5
                overall_sentiment = analyzer.calculate_overall_sentiment(branch_reviews)
                description, trend = analyzer.interpret_sentiment(analyzer.analyze_sentiment(branch_reviews))
                
                branch_data = {
                    "Branch": branch,
                    "Overall Sentiment": overall_sentiment,
                    "Sentiment Description": description,
                    "Sentiment Trend": trend
                }
                all_branches_data.append(branch_data)

            all_branches_df = pd.DataFrame(all_branches_data)
            st.subheader("All Branches Sentiment Analysis")
            st.write(all_branches_df)

            # Function to convert DataFrame to CSV (for download)
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')

            csv_data = convert_df_to_csv(all_branches_df)

            # Download button
            st.download_button(
                label="Download data as CSV",
                data=csv_data,
                file_name='branches_sentiment_analysis.csv',
                mime='text/csv',
            )
        
    elif "Student" in df.columns:
        student_names = df["Student"].unique().tolist()
        selected_student = st.selectbox("Select a Student:", ["All Students"] + student_names)
        review_period = st.selectbox("Review Period:", [1, 4])
    
        if selected_student != "All Students":
            # Processing for individual student
            student_data = df[df["Student"] == selected_student]
            st.write(student_data)  # Debug statement to check the student_data
            reviews = student_data.iloc[:, 1:].values.flatten().tolist()  # Assuming feedback starts from column index 1
    
            analyzer = SentimentAnalyzer()
    
            if review_period == 1:
                sentiments = analyzer.analyze_sentiment(reviews)
            else:
                sentiments = analyzer.analyze_periodic_sentiment(reviews, review_period)
    
            overall_sentiment = analyzer.calculate_overall_sentiment(reviews)
            st.subheader(f"Overall Sentiment for {selected_student}: {overall_sentiment:.2f}")
            st.subheader("Sentiment Analysis")
    
            # Plotting sentiment
            weeks = list(range(1, len(sentiments) + 1))
            sentiment_scores = [sentiment['compound'] for sentiment in sentiments]
            pos_scores = [sentiment['pos'] for sentiment in sentiments]
            neu_scores = [sentiment['neu'] for sentiment in sentiments]
            neg_scores = [sentiment['neg'] for sentiment in sentiments]
    
            fig, ax = plt.subplots()
            ax.plot(weeks, sentiment_scores, label="Overall", color="blue")
            ax.fill_between(weeks, sentiment_scores, color="blue", alpha=0.1)
            ax.plot(weeks, pos_scores, label="Positive", color="green")
            ax.plot(weeks, neu_scores, label="Neutral", color="gray")
            ax.plot(weeks, neg_scores, label="Negative", color="red")
    
            ax.set_xlabel('Week')
            ax.set_ylabel('Sentiment Score')
            ax.set_title(f'Sentiment Analysis for {selected_student}')
            ax.legend()
            st.pyplot(fig)
    
            description, trend = analyzer.interpret_sentiment(sentiments)
            st.subheader("Progress Description")
            st.write(f"Sentiment Trend: {trend}")
            st.write(f"Description: {description}")
    
            # Breakdown of analysis
            st.subheader("Breakdown of Analysis")
            breakdown_df = pd.DataFrame(sentiments, index=list(range(1, len(sentiments) + 1)))
            st.write(breakdown_df)
    
        else:
            # Processing for all students
            analyzer = SentimentAnalyzer()
            all_students_data = []
    
            for student in student_names:
                student_reviews = df[df["Student"] == student].iloc[:, 1:].values.flatten().tolist()  # Assuming feedback starts from column index 1
                overall_sentiment = analyzer.calculate_overall_sentiment(student_reviews)
                description, trend = analyzer.interpret_sentiment(analyzer.analyze_sentiment(student_reviews))
                
                student_data = {
                    "Student": student,
                    "Overall Sentiment": overall_sentiment,
                    "Sentiment Description": description,
                    "Sentiment Trend": trend
                }
                all_students_data.append(student_data)
    
            all_students_df = pd.DataFrame(all_students_data)
            st.subheader("All Students Sentiment Analysis")
            st.write(all_students_df)
    
            # Function to convert DataFrame to CSV (for download)
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')
    
            csv_data = convert_df_to_csv(all_students_df)
    
            # Download button
            st.download_button(
                label="Download data as CSV",
                data=csv_data,
                file_name='students_sentiment_analysis.csv',
                mime='text/csv',
            )

    else:
        # Assuming the new dataset structure without a 'Branch' column
        review_columns = df.columns[1::2]  # Adjust column selection to match the provided dataset structure
        reviews = df[review_columns].values.flatten().tolist()

        analyzer = SentimentAnalyzer()

        review_period = st.selectbox("Review Period:", [1, 4])

        if review_period == 1:
            sentiments = analyzer.analyze_sentiment(reviews)
        else:
            sentiments = analyzer.analyze_periodic_sentiment(reviews, review_period)

        overall_sentiment = analyzer.calculate_overall_sentiment(reviews)
        st.subheader(f"Overall Sentiment: {overall_sentiment:.2f}")
        st.subheader("Sentiment Analysis")

        # Plotting sentiment
        weeks = list(range(1, len(sentiments) + 1))
        sentiment_scores = [sentiment['compound'] for sentiment in sentiments]
        pos_scores = [sentiment['pos'] for sentiment in sentiments]
        neu_scores = [sentiment['neu'] for sentiment in sentiments]
        neg_scores = [sentiment['neg'] for sentiment in sentiments]

        fig, ax = plt.subplots()
        ax.plot(weeks, sentiment_scores, label="Overall", color="blue")
        ax.fill_between(weeks, sentiment_scores, color="blue", alpha=0.1)
        ax.plot(weeks, pos_scores, label="Positive", color="green")
        ax.plot(weeks, neu_scores, label="Neutral", color="gray")
        ax.plot(weeks, neg_scores, label="Negative", color="red")

        ax.set_xlabel('Week')
        ax.set_ylabel('Sentiment Score')
        ax.set_title('Sentiment Analysis')
        ax.legend()
        st.pyplot(fig)

        description, trend = analyzer.interpret_sentiment(sentiments)
        st.subheader("Progress Description")
        st.write(f"Sentiment Trend: {trend}")
        st.write(f"Description: {description}")

        # Breakdown of analysis
        st.subheader("Breakdown of Analysis")
        breakdown_df = pd.DataFrame(sentiments, index=list(range(1, len(sentiments) + 1)))
        st.write(breakdown_df)
        
