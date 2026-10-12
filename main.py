"""Command-line demo of the HR Policy Assistant.
Run with: python main.py
"""

from hr_assistant.pipeline import ask, build_hr_assistant

def main():
    print("Building the HR POlicy Assistant")
    agent = build_hr_assistant()
    print("Assistant is ready!\n")

    demo_questions = [
        "How many paid annual leaves do I get?",
        "What is the notice period during probation?",
        "Can I work from home every day?"
    ]

    for question in demo_questions:
        print("="*60)
        print("QUESTION - \n", question)
        print("-"*60)
        answer = ask(agent, question)
        print("ANSWER - \n", answer)        
        print("="*60)

if __name__ == "__main__":
    main()