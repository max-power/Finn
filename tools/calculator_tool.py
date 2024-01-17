from langchain_core.tools import BaseTool
from tools.tools import handle_tool_error

import math
import numexpr
import re

class CalculatorTool(BaseTool):
    name        = "Calculator"
    description = "Useful for when you need to answer questions about math."
    handle_tool_error=handle_tool_error
        
    def _run(self, expression: str):
        try:
            output = str(
                numexpr.evaluate(
                    expression.strip(),
                    global_dict={},  # restrict access to globals
                    local_dict={"pi": math.pi, "e": math.e}
                )
            )
        except Exception as e:
            raise ValueError(
                f'calculate("{expression}") raised error: {e}.'
                " Please try again with a valid numerical expression"
            )

        # Remove any leading and trailing brackets from the output
        return re.sub(r"^\[|\]$", "", output)
