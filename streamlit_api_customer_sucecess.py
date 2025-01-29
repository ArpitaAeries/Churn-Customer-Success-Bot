    
#load important libraries
import pandas as pd
import numpy as np
import random
import string
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display
import streamlit as st

#For LLM
import torch
import transformers
from transformers import AutoTokenizer
from  langchain import LLMChain, HuggingFacePipeline, PromptTemplate

import streamlit as st
import pandas as pd
import numpy as np
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt

#Part-1

# Load the Excel file into a dataframe
df1 = pd.read_excel('User_Specific_Data.xlsx')

# Calculate total usage count for each feature
usage_counts = df1[['Login_Usage_Count', 'Profile_Update_Usage_Count', 'Messaging_Usage_Count',
                    'Search_Usage_Count', 'Content_Upload_Usage_Count', 'Content_Download_Usage_Count',
                    'Comment_Usage_Count', 'Like_Usage_Count', 'Share_Usage_Count',
                    'Follow_Unfollow_Usage_Count']].sum()

# Calculate total usage duration for each feature
usage_durations = df1[['Login_Usage_Duration', 'Profile_Update_Usage_Duration',
                        'Messaging_Usage_Duration', 'Search_Usage_Duration',
                        'Content_Upload_Usage_Duration', 'Content_Download_Usage_Duration',
                        'Comment_Usage_Duration', 'Like_Usage_Duration', 'Share_Usage_Duration',
                        'Follow_Unfollow_Usage_Duration']].sum()

# Sort features by usage count and duration
most_used_features = usage_counts.sort_values(ascending=False)
most_used_durations = usage_durations.sort_values(ascending=False)

# Calculate success rate for each feature
success_rates = df1[['Login_successfully', 'Profile_Update_successfully', 'Search_successfully',
                     'Messaging_successfully', 'Content_Upload_successfully', 'Content_Download_successfully',
                     'Comment_successfully', 'Like_successfully', 'Share_successfully',
                     'Follow_Unfollow_successfully']].mean()

# Sort features by success rate
most_successful_features = success_rates.sort_values(ascending=False)

# Streamlit app
st.title("Advisory of Customer Success Team")
st.header("Customer Behavior Analysis:")

st.subheader("Most Used Features by Usage Count:")
st.write(most_used_features.head(1))

st.subheader("Most Used Features by Duration:")
st.write(most_used_durations.head(1))

# Plot for Most Used Features by Usage Count
st.subheader("Plot: Most Used Features by Usage Count")
fig, ax = plt.subplots(figsize=(10, 6))
most_used_features.plot(kind='bar', ax=ax)
plt.title('Most Used Features by Usage Count')
plt.xlabel('Features')
plt.ylabel('Total Usage Count')
st.pyplot(fig)

# Plot for Most Used Features by Duration
st.subheader("Plot: Most Used Features by Duration")
fig, ax = plt.subplots(figsize=(10, 6))
most_used_durations.plot(kind='bar', color='orange', ax=ax)
plt.title('Most Used Features by Duration')
plt.xlabel('Features')
plt.ylabel('Total Usage Duration')
st.pyplot(fig)

st.subheader("Highest Success Rate by Usage:")
st.write(most_successful_features.head(1))


st.header("Demographic Behaviour of the Users:")

# Load the CSV file into a dataframe
demographic_usage_stats = pd.read_csv('demographic_usage_stats.csv')

# Create dropdown widgets for Gender, Age_Group, and Location
gender_dropdown = st.selectbox('Select Gender:', demographic_usage_stats['Gender'].unique())
age_group_dropdown = st.selectbox('Select Age Group:', demographic_usage_stats['Age_Group'].unique())
location_dropdown = st.selectbox('Select Location:', demographic_usage_stats['Location'].unique())

# Function to filter the DataFrame based on selected dropdown values and display usage stats
def show_usage_stats(gender, age_group, location):
    filtered_df = demographic_usage_stats[(demographic_usage_stats['Gender'] == gender) &
                                          (demographic_usage_stats['Age_Group'] == age_group) &
                                          (demographic_usage_stats['Location'] == location)]

    # Check if the filtered DataFrame is not empty
    if not filtered_df.empty:
        # Select only numeric columns for calculating the mean
        numeric_columns = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
        sorted_features = filtered_df[numeric_columns].mean().sort_values(ascending=False)

        # Get the most popular feature name
        most_popular_feature = sorted_features.index[0]

        # Generate the output statement
        output_statement = f"The most popular feature among {gender}s of age group '{age_group}' in {location} is '{most_popular_feature}' with an average usage count of {sorted_features.iloc[0]:.2f}."
        st.write(output_statement)
    else:
        st.write("No data available for the selected criteria.")

# Display the usage stats based on dropdown selections
show_usage_stats(gender_dropdown, age_group_dropdown, location_dropdown)

st.header("Insights from the Feedbacks given by the Users:")

import streamlit as st
import pandas as pd
import numpy as np
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the CSV file into a dataframe
df3 = pd.read_csv('feature_comments.csv')

# Initialize an empty string to store the combined information
text = ""

# Iterate through each row in the dataframe
for index, row in df3.iterrows():
    feature = row['Feature']
    comment = row['Comment']
    text += f"{feature}: {comment}\n"
#st.write(text)

# Create a multiline text input for the template
template_input = st.text_area("Enter your LLM template", height=100)

# Create a button to generate the output
if st.button("Generate Output"):
    # Replace the `{text}` placeholder in the template with the actual combined text
    #template_input = template_input.replace("{text}", text)

    # LLM model setup
    model = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        max_length=10000,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )

    # Generate the LLM output
    llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})
    prompt = PromptTemplate(template=template_input, input_variables=["text"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    llm_output=llm_chain.run(text)
    
    #llm_output = pipeline(template_input)[0]['generated_text']

    # Find the index of "FINANCIAL REPORT:"
    index = llm_output.find("CUSTOMER REPORT:")

    # Extract the part of the string after "FINANCIAL REPORT:"
    output_after = llm_output[index + len("CUSTOMER REPORT:"):]

    # Display the generated output
    st.write("**Detailed Customer Report:**")
    st.write(output_after)
