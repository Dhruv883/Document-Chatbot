import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


def get_conversational_chain():
    prompt_template = """Answer the question comprehensively using only the information provided in the context. Ensure that all relevant details are included. If the necessary information is not available, respond with "Cannot Answer this question", 
    don't provide the wrong answer\n\n Context:\n {context}?\n Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    res = chain(
        {"input_documents": docs, "question": user_question}
        , return_only_outputs=True)

    return res["output_text"]


def main():
    st.header("Chat with your Documents")

    if 'history' not in st.session_state:
        st.session_state.history = []

    prompt = st.text_input("Enter your Query")
    if st.button("Ask"):
        if prompt:
            response = user_input(prompt)
            st.session_state.history.append((prompt, response))

    for question, answer in reversed(st.session_state.history):
        st.write(f"**Q.** {question}")
        st.write(f"**Ans.** {answer}")
        st.write("---")


if __name__ == "__main__":
    main()
