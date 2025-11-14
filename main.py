from agent.agent import create_agent

def read_multiline_input():
    print("You (Finish input with an empty line):")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break

        if line.strip() == "":
            break
        lines.append(line)

    return "\n".join(lines).strip()


def main():
    agent = create_agent()
    print("🤖 Agent ready! Type 'exit' to quit.\n")

    while True:
        msg = read_multiline_input()
        if msg.lower() in ["exit", "quit"]:
            break
        if not msg:
            continue

        try:
            result = agent.invoke({"input": msg})
            print("\nAI:\n", result if isinstance(result, str) else getattr(result, "content", result), "\n")
        except Exception as e:
            print("\n[Error]:", e, "\n")

#Main
if __name__ == "__main__":
    main()
