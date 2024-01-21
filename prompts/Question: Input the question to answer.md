Question: Input the question to answer.

Thought: Outline your decision-finding process as a plan, considering previous and upcoming steps.

Action: Use a JSON blob with "action" (tool name) and "action_input" (tool input) keys. Only use one action per $JSON_BLOB.

   Example:
   ```json
   {{
     "action": "$TOOL_NAME",
     "action_input": "$INPUT"
   }}
   ```

Observation: the result of the action.

Repeat the Thought/Action/Observation steps as needed.

Final Thought: "I know what to respond"

Final Action:
   ```json
   {{
     "action": "Final Answer",
     "action_input": "Final response to human"
   }}
   ```