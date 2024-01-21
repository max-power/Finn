from typing import Optional, Type
from langchain_core.tools import BaseTool
from tools.tools import handle_tool_error

from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.runnables.config import run_in_executor

import math
import numexpr
import re
import json

class CalculatorTool(BaseTool):
    name        = "Calculator"
    description = "Useful for when you need to answer questions about math. Examples: '2**8', '1230/12 + 99.95'"
    handle_tool_error=handle_tool_error
        
    def _run(
        self, expression: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
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
        return json.dumps({'result': re.sub(r"^\[|\]$", "", output) })

    # async def _arun(
    #     self, expression: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    # ) -> str:
    #     """Use the tool asynchronously."""
    #     return await run_in_executor(None, self._run, expression)
