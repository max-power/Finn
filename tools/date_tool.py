from pydantic.v1 import BaseModel, Field
from typing import Optional
from langchain_core.tools import BaseTool
from datetime import datetime
import moment
from tools.tools import handle_tool_error

class DateToolSchema(BaseModel):
    """Input for DateTool."""
    input: str = Field(
        title="Datetime string",
        description="Input is any date or temporal string.", 
        default="now",
        examples=["December 18, 2012", "2012-12-18", 1355875153626, "now", "2 days ago", "in 10 years", "yesterday"],
    )
    format: Optional[str] = Field(
        title="Format",
        description="Output format of the date or time. Use strftime formatting string.",
        default="%A, %B %d, %Y %H:%M:%S",
        examples=["%A, %B %d, %Y %H:%M:%S", "%H:%M:%S", "%A", "%Y"],
    )

class DateTool(BaseTool):
    name = "Moment"
    description = "Useful for when you are need to find the date and or time. Remember JSON input keys 'input' and 'format', both optional."
    args_schema = DateToolSchema
    handle_tool_error=handle_tool_error
        
    def _run(self, input="now", format="%A, %B %d, %Y %H:%M:%S"):
        return moment.date(input).strftime(format)
