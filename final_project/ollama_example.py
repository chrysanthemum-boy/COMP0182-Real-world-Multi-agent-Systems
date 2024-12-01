import ollama
from ollama import chat
from ollama import ChatResponse

# response: ChatResponse = chat(model='llama3.2', messages=[
#   {
#     'role': 'user',
#     'content': 'Why is the sky blue?',
#   },
# ])

# print(response['message']['content'])
# # or access fields directly from the response object
# print(response.message.content)

def resolve_conflict_with_llm(conflict):
    prompt = f"""
    Conflict detected:
    - Time: {conflict['time']}
    - Agent 1: {conflict['agent_1']} at location {conflict['location_1']}
    - Agent 2: {conflict['agent_2']} at location {conflict['location_2']}

    Suggest a resolution to this conflict without altering their end goals.
    """

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "system", "content": "You are a conflict resolution expert for multi-agent systems."},
                  {"role": "user", "content": prompt}]
    )

    return response['message']['content']


# answer = ollama.generate(model='llama3.2', prompt='Why is the sky blue?')
answer = resolve_conflict_with_llm({'time': 10, 'agent_1': 'A', 'location_1': '', 'agent_2': 'B', 'location_2': 'B1'})
print(answer)