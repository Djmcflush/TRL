import streamlit as st
from transformers import pipeline
import argparse




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="scripts/training/task_configs/summarization/t5_ppo.yml", help="Path for config")
    args = parser.parse_args()


    model = pipeline("summarization")

    def summarize_text(text):
        result = model(text, max_length=100, min_length=30)[0]['summary_text']
        return result

    def run_python_script_with_flags(file_path, flags):
        with open(file_path, "r") as file:
            code = file.read()
            exec(compile(code, file_path, 'exec'), flags.__dict__)
    header_text = """
    \n \n
    Author: DeMarcus Edwards  \n
    Advisor: Danda B. Rawat \n
    Lab: Center of Excellence in AI/ML \n
    University: Howard University \n
    Department : Department of Electrical Engineering & Computer Science \n
    """
    st.write("", header=header_text, unsafe_allow_html=True)

    
    st.title("How It Works: \n ")
    st.title("Reinforcement Learning for Language Models")

    st.write("To Start Here's the Base Model Output for this Large Language Model (T5) " )
    text = st.text_area("Enter some text to summarize", "")

    if text:
        summary = summarize_text(text)
        st.write("Summary: ", summary)

    explanation = """
    Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by performing actions in an environment to maximize a reward signal. 

    In RL, an agent interacts with an environment by taking actions, receiving observations, and obtaining rewards. The agent's goal is to learn a policy that maps observations to actions, so as to maximize the expected cumulative reward over time. 

    The learning process in RL typically involves trial and error, where the agent takes actions and receives feedback in the form of rewards. The agent uses this feedback to update its policy, gradually improving its decision-making over time. 

    RL can be applied to a wide range of problems, including control problems, games, and robotics. It has been used to solve complex tasks, such as playing Atari games, controlling robots, and optimizing financial portfolios.
    """

    st.write(explanation)

    reinforcement = st.button("Retrain Model with Reinforcement Learning")
    if reinforcement:
        run_python_script_with_flags("scripts/training/train_text_generation.py", args)
        summary = summarize_text(text)
        new_output = st.button("RL new Output")
        if new_output:
            st.write("Summary: ", summary)

    


