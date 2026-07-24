def run_agent(messages, tools, max_steps=12):
    for step in range(max_steps):
        reply = model(messages, tools)
        messages.append(reply)

        if not reply.tool_calls:
            return reply.text

        for call in reply.tool_calls:
            result = dispatch(call)
            messages.append(tool_message(result))

    raise StepLimitReached(messages)
