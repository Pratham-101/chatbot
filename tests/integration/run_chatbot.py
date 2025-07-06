import asyncio
from chatbot.rag_chatbot import RAGChatbot

async def main():
    chatbot = RAGChatbot()
    print("Chatbot CLI. Type 'exit' to quit.")
    while True:
        question = input("Enter your question: ")
        if question.lower() == "exit":
            break
        print("Processing your question, please wait...")
        answer = await chatbot.generate_answer(question)
        print(f"Answer:\n{answer}\n")

if __name__ == "__main__":
    asyncio.run(main())
