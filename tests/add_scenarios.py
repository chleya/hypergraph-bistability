# Script to add new scenarios

new_scenarios = '''
    # ========== Step 2: New stress scenarios ==========
    EvalScenario(
        name="paraphrase_heavy_conflict",
        description="The agent should recognize the same conflict through heavily paraphrased questions.",
        tier="stress",
        turns=[
            EvalTurn("Remember that there's a debate about whether to use PostgreSQL or MongoDB for the user profile service. The team prefers PostgreSQL but the lead engineer prefers MongoDB for flexibility."),
            EvalTurn("What's for lunch today?"),  # distraction
            EvalTurn("I need a summary of the architectural decision we discussed.",  # paraphrase 1
                expected_retrievals=["PostgreSQL", "MongoDB", "user profile service"],
                expected_response_signals=["PostgreSQL", "MongoDB"],
            ),
            EvalTurn("Give me the key points from our database technology discussion.",  # paraphrase 2
                expected_retrievals=["PostgreSQL", "MongoDB", "team prefers"],
                expected_response_signals=["database", "PostgreSQL", "MongoDB"],
            ),
            EvalTurn("What did we decide about storing user data?",  # paraphrase 3
                expected_retrievals=["user profile service", "PostgreSQL", "MongoDB"],
                expected_response_signals=["user data", "decision"],
            ),
            EvalTurn("Tell me the status of our storage technology choice.",  # paraphrase 4
                expected_retrievals=["storage technology", "PostgreSQL vs MongoDB"],
                expected_response_signals=["storage", "technology"],
            ),
        ],
        tags=["conflict", "paraphrase", "memory"],
    ),
    EvalScenario(
        name="long_interruption_resume",
        description="The agent should recover task context after a long interruption sequence.",
        tier="stress",
        turns=[
            EvalTurn("I'm working on implementing user authentication for the new API."),
            EvalTurn("The auth should use JWT tokens with refresh token rotation."),
            EvalTurn("Let's take a break and talk about dinner."),
            EvalTurn("What's a good recipe for pasta?"),
            EvalTurn("Actually, let's get some coffee first."),
            EvalTurn("How do you make a good latte?"),
            EvalTurn("Okay, back to work. What was I implementing?",
                expected_retrievals=["user authentication", "JWT tokens", "refresh token rotation"],
                expected_response_signals=["authentication", "JWT", "refresh token"],
            ),
            EvalTurn("What are the security considerations for JWT?",
                expected_retrievals=["JWT", "security", "refresh token rotation"],
                expected_response_signals=["JWT", "security"],
            ),
            EvalTurn("Continue with the implementation. What do I need to add next?",
                expected_retrievals=["JWT tokens", "refresh token rotation", "auth"],
                expected_response_signals=["implementation", "next steps"],
            ),
        ],
        tags=["interruption", "task_resume", "long_context"],
    ),
'''

# Read existing scenarios
with open(r'src\hypergraph_bistability\evals\scenarios.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the last occurrence of '],' to insert after it
import re
# Find position before the final ]
last_bracket = content.rfind(']')
if last_bracket > 0:
    new_content = content[:last_bracket] + new_scenarios + '\n' + content[last_bracket:]
    
    with open(r'src\hypergraph_bistability\evals\scenarios.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print('Added 2 new scenarios!')
else:
    print('Could not find insertion point')
