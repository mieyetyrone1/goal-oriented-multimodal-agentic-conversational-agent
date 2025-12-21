class SimpleCalculatorTool:
    """
    Minimal example tool.
    Demonstrates agent -> tool -> result flow.
    """

    name = "calculator"
    description = "Performs basic arithmetic expressions."

    def run(self, expression: str) -> str:
        try:
            result = eval(expression, {"__builtins__": {}})
            return f"Result: {result}"
        except Exception as e:
            return f"Error evaluating expression: {str(e)}"
