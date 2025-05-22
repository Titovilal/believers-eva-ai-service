from src.retrievers import ChromaDBRetriever
from src.agents.agent_factory import AgentFactory
import chainlit as cl


@cl.on_chat_start
async def start_chat():
    cl.user_session.set("messages", [])
    retriever = ChromaDBRetriever(
        persist_directory="notebooks/chromadb_index_notebook01"
    )
    retriever.load("notebooks/chromadb_index_notebook01")
    if retriever.count() == 0:
        raise ValueError(
            "No documents found in the index. Please check the index path."
        )
    else:
        print(f"Number of documents in the index: {retriever.count()}")
    agent = AgentFactory.create_agent(retriever=retriever)
    cl.user_session.set("agent", agent)


# What is the definition of carbon credit in the ESG efrag context?


async def inference(query: str):
    messages = cl.user_session.get("messages")
    # messages.append({"role": "user", "content": query})
    agent = cl.user_session.get("agent")

    result = agent.run_sync(query, message_history=messages)

    print(result.all_messages())
    cl.user_session.set("messages", result.all_messages())

    retrieved_chunks = ""
    for i, step in enumerate(result.all_messages(), 1):
        for part in step.parts:
            if part.part_kind == "tool-return":
                if part.tool_name == "retrieve_from_documents":
                    retrieved_chunks = part.content
            # if part.part_kind == "tool-call":
            #    st.json({"Tool": part.tool_name, "Arguments": part.args})
            #    elif part.part_kind == "tool-return":
            #        if part.tool_name == "get_schema_view":
            #            st.code(part.content, language="sql", line_numbers=True)
            #        else:
            #            st.write(part.content)
            #    elif hasattr(part, "content"):
            #        st.write(part.content)
    # Send a response back to the user
    await cl.Message(
        content=result.output,
        # content=f"{result.output}\n\n{retrieved_chunks}",
        author="Gemini",
    ).send()

    await cl.Message(
        content="",
        elements=[
            cl.Text(
                name="Retrieved chunks:", content=retrieved_chunks, display="inline"
            )
        ],
        author="Gemini",
    ).send()


@cl.on_message
async def chat(message: cl.Message):
    await inference(message.content)
