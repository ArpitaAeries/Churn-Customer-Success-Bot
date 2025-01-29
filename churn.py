import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load your trained model
model = joblib.load('/content/churn_model.pkl')

# Define a function to make predictions
def predict_churn(data):
    df = pd.DataFrame(data, index=[0])
    
    # Apply the same feature engineering steps
    df['Total_Usage_Duration'] = df['Comment_Usage_Duration'] + df['Like_Usage_Duration'] + df['Share_Usage_Duration']
    df['Profile_Content_Interaction'] = df['Profile_Update_Usage_Count'] * df['Content_Download_Usage_Count']
    df['Feedback_Ratio'] = df['User_Feedback_Positive'] / (df['User_Feedback_Negative'] + 1)
    df['Occupation_Group'] = df['Occupation_Designer'] + df['Occupation_Engineer'] + df['Occupation_Manager']
    df['Log_Comment_Usage_Duration'] = np.log1p(df['Comment_Usage_Duration'])
    
    # Ensure the order of columns matches the training data
    df = df[[
        'Search_successfully', 'Content_Download_successfully', 'Profile_Update_Usage_Count', 
        'Content_Download_Usage_Count', 'Comment_Usage_Duration', 'Like_Usage_Duration', 
        'Share_Usage_Duration', 'Occupation_Designer', 'Occupation_Doctor', 'Occupation_Engineer', 
        'Occupation_Manager', 'Occupation_Teacher', 'User_Feedback_Negative', 'User_Feedback_Positive', 
        'Support_Interaction_No', 'Support_Interaction_Yes', 'A/B_Test_Group_A', 'A/B_Test_Group_B', 
        'A/B_Test_Result_Negative', 'A/B_Test_Result_Positive', 'Total_Usage_Duration', 
        'Profile_Content_Interaction', 'Feedback_Ratio', 'Occupation_Group', 'Log_Comment_Usage_Duration'
    ]]
    
    prediction = model.predict(df)
    return prediction[0]

# Streamlit app
st.title("Churn Prediction App")

# Create input fields for each feature
search_successfully = st.number_input('Search Successfully')
content_download_successfully = st.number_input('Content Download Successfully')
profile_update_usage_count = st.number_input('Profile Update Usage Count')
content_download_usage_count = st.number_input('Content Download Usage Count')
comment_usage_duration = st.number_input('Comment Usage Duration')
like_usage_duration = st.number_input('Like Usage Duration')
share_usage_duration = st.number_input('Share Usage Duration')
occupation_designer = st.selectbox('Occupation Designer', [0, 1])
occupation_doctor = st.selectbox('Occupation Doctor', [0, 1])
occupation_engineer = st.selectbox('Occupation Engineer', [0, 1])
occupation_manager = st.selectbox('Occupation Manager', [0, 1])
occupation_teacher = st.selectbox('Occupation Teacher', [0, 1])
user_feedback_negative = st.number_input('User Feedback Negative')
user_feedback_positive = st.number_input('User Feedback Positive')
support_interaction_no = st.selectbox('Support Interaction No', [0, 1])
support_interaction_yes = st.selectbox('Support Interaction Yes', [0, 1])
ab_test_group_a = st.selectbox('A/B Test Group A', [0, 1])
ab_test_group_b = st.selectbox('A/B Test Group B', [0, 1])
ab_test_result_negative = st.selectbox('A/B Test Result Negative', [0, 1])
ab_test_result_positive = st.selectbox('A/B Test Result Positive', [0, 1])

# Create a dictionary to hold the input data
input_data = {
    'Search_successfully': search_successfully,
    'Content_Download_successfully': content_download_successfully,
    'Profile_Update_Usage_Count': profile_update_usage_count,
    'Content_Download_Usage_Count': content_download_usage_count,
    'Comment_Usage_Duration': comment_usage_duration,
    'Like_Usage_Duration': like_usage_duration,
    'Share_Usage_Duration': share_usage_duration,
    'Occupation_Designer': occupation_designer,
    'Occupation_Doctor': occupation_doctor,
    'Occupation_Engineer': occupation_engineer,
    'Occupation_Manager': occupation_manager,
    'Occupation_Teacher': occupation_teacher,
    'User_Feedback_Negative': user_feedback_negative,
    'User_Feedback_Positive': user_feedback_positive,
    'Support_Interaction_No': support_interaction_no,
    'Support_Interaction_Yes': support_interaction_yes,
    'A/B_Test_Group_A': ab_test_group_a,
    'A/B_Test_Group_B': ab_test_group_b,
    'A/B_Test_Result_Negative': ab_test_result_negative,
    'A/B_Test_Result_Positive': ab_test_result_positive
}

# Button to make predictions
if st.button('Predict Churn'):
    result = predict_churn(input_data)
    st.write(f'The predicted churn status is: {"Churned" if result == 1 else "Not Churned"}')
