import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt 
import joblib

# Load pre-trained model
pipe_lr = joblib.load("pipe_lr.joblib")

# Emotion emojis dictionary
emotions_emoji_dict = { 
    "anger": "😠",  
    "fear": "😨",  
    "joy": "😊😂",  
    "love": "😍",  
    "sadness": "😔", 
    "surprise": "😲" 
}

# Function to predict emotion
def predict_emotion(docx):
    results = pipe_lr.predict([docx])
    return results[0]

# Function to get prediction probabilities
def get_prediction_proba(docx):    
    results = pipe_lr.predict_proba([docx])
    return results

# Streamlit app layout
def main():
    st.title("Text Emotion Detection")
    st.subheader("Detect Emotion In Text")

    with st.form(key='my_form'):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label='Submit')

    # Execute only if form is submitted
    if submit_text:
        # Define col1 and col2 inside the if block
        col1, col2 = st.columns(2)

        # Predictions
        prediction = predict_emotion(raw_text)
        probability = get_prediction_proba(raw_text)

        with col1:
            st.success("Original Text")
            st.write(raw_text)

            st.success("Prediction")
            emoji_icon = emotions_emoji_dict.get(prediction, "❓")  # Handle unknown predictions
            st.write(f"{prediction}: {emoji_icon}")
            st.write(f"Confidence: {np.max(probability):.2f}")

        with col2:
            st.success("Prediction Probability")
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
            st.altair_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()
