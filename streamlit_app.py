import openai 
import streamlit as st
#pip install streamlit-chat  
from streamlit_chat import message
import numpy as np
import pandas as pd
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity
st.set_page_config(layout="wide")


openai.api_key = ""
df=pd.read_csv('processed/embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

toggle_value = st.sidebar.selectbox('Select an Engine:', ['Davinci-GPT-3','ChatGPT-3'])



# Check the toggle value and do something different based on the selection
if toggle_value == 'ChatGPT-3':
    st.sidebar.markdown('Now you are using <span style="color: red;">ChatGPT-3</span> engine', unsafe_allow_html=True)
    # Do something for apple
else:
    st.sidebar.markdown('You are using <span style="color: red;">Davinci GPT-3</span> engine', unsafe_allow_html=True)
    # Do something for orange


def create_context(
    question, df, max_len=2900, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')


    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        
        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4
        
        # If the context is too long, break
        if cur_len > max_len:
            break
        
        # Else add it to the text that is being returned
        returns.append(row["text"])
#     print("\n\n###\n\n".join(returns))
    # Return the context
    return "\n\n###\n\n".join(returns)

def answer_question(
    df,
    model="text-davinci-003",
    question="Am I allowed to publish model outputs to Twitter, without a human review?",
    max_len=3000,
    size="ada",
    debug=False,
    max_tokens=350,
    stop_sequence=None
    
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        if model == 'Davinci-GPT-3':
            model = "text-davinci-003"
            # Create a completions using the question and context
            response = openai.Completion.create(
                prompt=f"You are helpful assistant. Answer the question (in your own words) based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
                temperature=0.1,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=stop_sequence,
                model=model,
            )
            return response["choices"][0]["text"].strip()
        
        else:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Answer the question (in your own words) mostly based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:"},
                ],
                 temperature = 0.5
                )
            return response["choices"][0]['message']['content']
    
    except Exception as e:
        print(e)
        return ""


def generate_response(prompt):
    completions = openai.Completion.create(
        engine = "text-davinci-003",
        prompt = prompt,
        max_tokens = 1024,
        n = 1,
        stop = None,
        temperature=0.5,
    )
    message = completions.choices[0].text
    return message

# st.title("Document Search/Q&A using GPT-3")
st.markdown("<h1 style='font-size:30px'>Document Search/Q&A using GPT-3</h1>", unsafe_allow_html=True)
st.markdown("<span style='font-size: 12px;'>I am in beta testing mode, please report any issues to the admin. Use natural language to ask questions below. If you encounter an error, try refreshing the page or select a different engine</span>", unsafe_allow_html=True)
st.markdown("<span style='font-size: 12px;'>This application utilizes a collection of open source text corpus related to Victory Capital, including data from the Annual 10-K report. The stored text is vectorized using GPT text-embedding-ada-002 engine to generate embeddings, and either the text-davinci-003 or ChatGPT-3 engine is employed to answer your questions based on semantic match using natural language. To get started, simply type your question and hit enter.</span>", unsafe_allow_html=True)

# st.markdown("I am in beta testing mode, please report any issues to the admin. Use natural language to ask questions below. If you encounter an error, try refreshing the page or select a different engine")
# st.markdown("This app uses a corpus of open source Victory Capital information (e.g. from Annual 10-K report). This app creates embeddings of the stored text using GPT3-ada engine and uses either text-davinci-003 or ChatGPT-3 engine for answring the question based on sematic match using natural language.")

# Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []



# We will get the user's input by calling the get_text function
def get_text():
    input_text = st.text_input("You: ","", key="input")
    return input_text

#dataframe with questions
questions_df = pd.read_csv('questions_df.csv')

# user question is generated and stored in a csv file
user_input = get_text()

with st.sidebar:
    st.text("\n\n")
# Define custom CSS styles
    custom_css = f"""
        .stButton {{
            padding-top: 0.2rem;
            padding-bottom: 0.2rem;
        }}
    """

    # Add custom CSS styles to the page
    st.markdown(f'<style>{custom_css}</style>', unsafe_allow_html=True)

    # st.text("\n\n")
    st.markdown("<span style='font-size: 24px;'>Not sure what to ask?</span>", unsafe_allow_html=True)
    if st.button("Click here to generate a question"):
        x = np.random.choice(range(0,37))
        st.markdown("<span style='font-size: 14px;'>{}</span>".format(questions_df.iloc[x].values[0]), unsafe_allow_html=True)
        user_input = questions_df.iloc[x].values[0]
    # st.markdown("Not sure what to ask?")
    # if st.button("Click here to generate a question"):
    #     x = np.random.choice(range(0,37))
    #     st.markdown(questions_df.iloc[x].values[0])
    #     user_input = questions_df.iloc[x].values[0]

qbank = pd.read_csv('qbank.csv')
qbank = pd.concat([qbank, pd.DataFrame([user_input], columns = ['question'])], ignore_index = True)
qbank = qbank.dropna().drop_duplicates().reset_index(drop= True)
qbank.to_csv('qbank.csv', index = False)


with st.sidebar:
    st.text("\n\n\n")
    # Add a heading with a reduced margin-top and margin-bottom
    st.markdown("<h3 style='margin-top: 0.0rem; margin-bottom: 0.25rem;'>Curious what others are asking?</h3>", unsafe_allow_html=True)
    # Get the maximum index available in the dataframe
    max_index = questions_df.shape[0] - 1

    # Generate a list of unique random indices
    y = np.random.choice(range(0, max_index), 5)

    # Get the questions from the dataframe
    questions = [questions_df.iloc[y[i]].values[0] for i in range(len(y))]

    # Create a div container with reduced padding
    st.markdown("""
    <div style='padding: 0.25rem;'>
        <ul style='margin-top: 0.25rem;'>
            <li style='font-size: 11px;'>{}</li>
            <li style='font-size: 11px;'>{}</li>
            <li style='font-size: 11px;'>{}</li>
            <li style='font-size: 11px;'>{}</li>
            <li style='font-size: 11px;'>{}</li>
        </ul>
    </div>
    """.format(*questions), unsafe_allow_html=True)
    
    st.text("\n\n")
    st.markdown("<h5 style='margin-top: 0.0rem; margin-bottom: 0.0rem; font-size: 0.8rem;'>Click the button to see more questions</h5>", unsafe_allow_html=True)
    if st.button('Show More question', key='refresh_button', help="Click to get a new set of questions."):
        y = np.random.choice(range(0, max_index), 5)
        questions = [questions_df.iloc[y[i]].values[0] for i in range(len(y))]
        # st.markdown("<h3 style='margin-top: 0.0rem; margin-bottom: 0.25rem;'>Curious what others are asking?</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div style='padding: 0.25rem;'>
            <ul style='margin-top: 0.25rem;'>
                <li style='font-size: 11px;'>{}</li>
                <li style='font-size: 11px;'>{}</li>
                <li style='font-size: 11px;'>{}</li>
                <li style='font-size: 11px;'>{}</li>
                <li style='font-size: 11px;'>{}</li>
            </ul>
        </div>
        """.format(*questions), unsafe_allow_html=True)


# with st.sidebar:
#     st.text("\n\n\n")
#     # Add a heading with a reduced margin-top and margin-bottom
#     # st.markdown("<h1 style='margin-top: 0.0rem; margin-bottom: 0.25rem; font-size: 1.2rem;'>Curious what others are asking?</h1>", unsafe_allow_html=True)
#     # Get the maximum index available in the dataframe
#     max_index = questions_df.shape[0] - 1

#     # Generate a list of unique random indices
#     y = np.unique(np.random.choice(range(0, max_index), 6))

#     # y = np.unique(np.random.choice(range(0,qbank.shape[0]-1),6))
#     # Get the questions from the dataframe
#     questions = [questions_df.iloc[y[i]].values[0] for i in range(len(y))]
#     # Create a div container with reduced padding
#     st.markdown("""
#     <div style='padding: 0.25rem;'>
#         <h1 style='margin-top: 0.0rem; margin-bottom: 0.25rem; font-size: 1.2rem;'>Curious what others are asking?</h1>
#         <ul style='margin-top: 0.25rem;'>
#             <li style='font-size: 11px;'>{}</li>
#             <li style='font-size: 11px;'>{}</li>
#             <li style='font-size: 11px;'>{}</li>
#             <li style='font-size: 11px;'>{}</li>
#             <li style='font-size: 11px;'>{}</li>
#         </ul>
#     </div>
#     """.format(*[questions_df.iloc[y[i]].values[0] for i in range(len(y))]), unsafe_allow_html=True)
    


if user_input:
    
    output = answer_question(df, model = toggle_value, question = user_input)
    # store the output 
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state['generated']:
    
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')