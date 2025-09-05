"""
AI Client for Keynote MCP Server Integration

This client demonstrates how to integrate an MCP server with a Large Language Model
(Gemini 2.0 Flash) to create an intelligent automation system. The client handles
the communication between the AI model and the MCP server, managing tool calls,
error handling, and state management.

Key Features:
- Asynchronous MCP server communication
- LLM integration with Google Gemini
- Intelligent tool selection and parameter parsing
- Multi-phase execution (computation + visualization)
- Robust error handling and timeout management
- State management for complex workflows

Author: Senior AI Engineer Portfolio Project
License: MIT
"""

import os
import asyncio
import logging
from typing import Dict, List, Any, Optional
from concurrent.futures import TimeoutError
from functools import partial

# Environment and configuration
from dotenv import load_dotenv

# MCP Framework
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

# Google AI
from google import genai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class KeynoteAIClient:
    """
    AI Client for Keynote MCP Server Integration.
    
    This class manages the communication between a Large Language Model
    and the Keynote MCP server, providing intelligent automation capabilities.
    """
    
    def __init__(self, api_key: Optional[str] = None, max_iterations: int = 6):
        """
        Initialize the AI client.
        
        Args:
            api_key: Google Gemini API key (if None, will load from environment)
            max_iterations: Maximum number of iterations for problem solving
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        self.client = genai.Client(api_key=self.api_key)
        self.max_iterations = max_iterations
        
        # State management
        self.reset_state()
        
        # Visualization workflow steps
        self.visualization_steps = ["open_keynote_presentation", "draw_rectangle_in_keynote", "add_text_to_keynote"]
        
        logger.info("KeynoteAIClient initialized successfully")
    
    def reset_state(self) -> None:
        """Reset all state variables to their initial values."""
        self.last_response = None
        self.iteration = 0
        self.iteration_responses = []
        self.current_phase = "computation"  # "computation" or "visualization"
        self.current_visualization_step = 0
        
        logger.debug("Client state reset")
    
    async def generate_with_timeout(self, prompt: str, timeout: int = 10) -> str:
        """
        Generate content from the LLM with timeout protection.
        
        Args:
            prompt: Input prompt for the LLM
            timeout: Maximum time to wait for response (seconds)
            
        Returns:
            Generated text from the LLM
            
        Raises:
            TimeoutError: If generation takes longer than timeout
            Exception: If generation fails
        """
        logger.info("Starting LLM generation with timeout protection")
        
        try:
            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None, 
                    lambda: self.client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=prompt
                    )
                ),
                timeout=timeout
            )
            
            logger.info("LLM generation completed successfully")
            return response.text.strip()
            
        except TimeoutError:
            logger.error("LLM generation timed out")
            raise
        except Exception as e:
            logger.error(f"Error in LLM generation: {e}")
            raise
    
    def create_system_prompt(self, tools: List[Any]) -> str:
        """
        Create a comprehensive system prompt with available tools.
        
        Args:
            tools: List of available MCP tools
            
        Returns:
            Formatted system prompt string
        """
        logger.info(f"Creating system prompt with {len(tools)} available tools")
        
        try:
            tools_description = []
            for i, tool in enumerate(tools):
                try:
                    # Extract tool information
                    params = tool.inputSchema
                    description = getattr(tool, 'description', 'No description available')
                    name = getattr(tool, 'name', f'tool_{i}')
                    
                    # Format parameters
                    if 'properties' in params:
                        param_details = []
                        for param_name, param_info in params['properties'].items():
                            param_type = param_info.get('type', 'unknown')
                            param_details.append(f"{param_name}: {param_type}")
                        params_str = ', '.join(param_details)
                    else:
                        params_str = 'no parameters'
                    
                    tool_desc = f"{i+1}. {name}({params_str}) - {description}"
                    tools_description.append(tool_desc)
                    logger.debug(f"Added tool description: {tool_desc}")
                    
                except Exception as e:
                    logger.warning(f"Error processing tool {i}: {e}")
                    tools_description.append(f"{i+1}. Error processing tool")
            
            tools_description_str = "\n".join(tools_description)
            
            system_prompt = f"""You are an advanced AI agent specialized in mathematical computation and presentation automation. You have access to a comprehensive set of tools for both mathematical operations and Keynote presentation creation.

Available Tools:
{tools_description_str}

RESPONSE FORMAT:
You must respond with EXACTLY ONE line in one of these formats (no additional text):

1. For function calls:
FUNCTION_CALL: function_name|param1|param2|...

2. For final answers:
FINAL_ANSWER: [number] | ReasoningType:[type] | SelfCheck:[pass/fail]

3. For errors or uncertainty:
ERROR: [description of issue] | SuggestedNextStep:[action]

IMPORTANT RULES:
- Always reason step-by-step internally before responding
- Clearly tag reasoning type (Arithmetic, Logic, Lookup, etc.)
- Perform a quick self-check before finalizing any answer (e.g., check order of magnitude, recompute critical step)
- When a function returns multiple values, process all of them
- Do not repeat function calls with the same parameters
- For array parameters, use square brackets: [value1,value2,value3]
- You can only make ONE function call at a time
- Always wait for function completion before proceeding
- If a function fails, output an ERROR with a suggested next step
- In a multi-turn conversation, continue from the last completed step without repeating earlier steps

WORKFLOW:
1. COMPUTATION PHASE:
   - Complete all mathematical computations using available tools
   - When computation is complete, output FINAL_ANSWER with reasoning type and self-check status
   - This marks the end of the computation phase

2. VISUALIZATION PHASE:
   - After FINAL_ANSWER, proceed to visualization
   - Call these tools ONE AT A TIME in this exact order:
     a. FUNCTION_CALL: open_keynote_presentation
     b. FUNCTION_CALL: draw_rectangle_in_keynote|"left"
     c. FUNCTION_CALL: add_text_to_keynote|"The answer is: [your actual answer]"
   - Wait for each function to complete before calling the next one
   - Always include your actual computed answer in the text

3. ERROR HANDLING:
   - If you are unsure, if the computation result fails sanity checks, or if a tool call fails, respond with:
     ERROR: [description] | SuggestedNextStep:[retry with different input / clarify / recompute]

EXAMPLE WORKFLOW:
Query: "Find ASCII values of 'HELLO' and sum their exponentials, then visualize in Keynote"

Step 1: FUNCTION_CALL: string_to_ascii_values|HELLO
Step 2: FUNCTION_CALL: exponential_sum|[72,69,76,76,79]
Step 3: FINAL_ANSWER: [1.234e+15] | ReasoningType:Arithmetic | SelfCheck:pass
Step 4: FUNCTION_CALL: open_keynote_presentation
Step 5: FUNCTION_CALL: draw_rectangle_in_keynote|"left"
Step 6: FUNCTION_CALL: add_text_to_keynote|"The answer is: 1.234e+15"

CRITICAL: Your entire response must be a single line starting with FUNCTION_CALL:, FINAL_ANSWER:, or ERROR:. No explanations or additional text.
"""

            logger.info("System prompt created successfully")
            return system_prompt
            
        except Exception as e:
            logger.error(f"Error creating system prompt: {e}")
            return "Error loading tools"
    
    async def execute_tool_call(self, session: ClientSession, tools: List[Any], 
                              function_info: str) -> Dict[str, Any]:
        """
        Execute a tool call with proper parameter parsing and error handling.
        
        Args:
            session: MCP client session
            tools: List of available tools
            function_info: Function call information from LLM
            
        Returns:
            Dictionary with execution results
        """
        logger.info(f"Executing tool call: {function_info}")
        
        try:
            # Parse function information
            parts = [p.strip() for p in function_info.split("|")]
            func_name, params = parts[0], parts[1:]
            
            logger.debug(f"Function: {func_name}, Parameters: {params}")
            
            # Find the matching tool
            tool = next((t for t in tools if t.name == func_name), None)
            if not tool:
                available_tools = [t.name for t in tools]
                raise ValueError(f"Unknown tool: {func_name}. Available: {available_tools}")
            
            logger.debug(f"Found tool: {tool.name}")
            
            # Prepare arguments according to tool schema
            arguments = {}
            schema_properties = tool.inputSchema.get('properties', {})
            
            for param_name, param_info in schema_properties.items():
                if not params:
                    raise ValueError(f"Not enough parameters provided for {func_name}")
                
                value = params.pop(0)
                param_type = param_info.get('type', 'string')
                
                logger.debug(f"Converting {param_name}={value} to type {param_type}")
                
                # Type conversion
                if param_type == 'integer':
                    arguments[param_name] = int(value)
                elif param_type == 'number':
                    arguments[param_name] = float(value)
                elif param_type == 'array':
                    if isinstance(value, str):
                        value = value.strip('[]').split(',')
                    arguments[param_name] = [int(x.strip()) for x in value]
                else:
                    arguments[param_name] = str(value)
            
            logger.debug(f"Final arguments: {arguments}")
            
            # Execute the tool
            result = await session.call_tool(func_name, arguments=arguments)
            logger.info(f"Tool {func_name} executed successfully")
            
            # Process result
            if hasattr(result, 'content'):
                if isinstance(result.content, list):
                    iteration_result = [
                        item.text if hasattr(item, 'text') else str(item)
                        for item in result.content
                    ]
                else:
                    iteration_result = str(result.content)
            else:
                iteration_result = str(result)
            
            # Format response
            if isinstance(iteration_result, list):
                result_str = f"[{', '.join(iteration_result)}]"
            else:
                result_str = str(iteration_result)
            
            return {
                "success": True,
                "result": iteration_result,
                "formatted_result": result_str,
                "function_name": func_name,
                "arguments": arguments
            }
            
        except Exception as e:
            logger.error(f"Error executing tool call: {e}")
            return {
                "success": False,
                "error": str(e),
                "function_name": func_name if 'func_name' in locals() else "unknown"
            }
    
    async def run_query(self, query: str) -> Dict[str, Any]:
        """
        Execute a complete query with computation and visualization.
        
        Args:
            query: User query to process
            
        Returns:
            Dictionary with execution results and final answer
        """
        logger.info(f"Starting query execution: {query}")
        self.reset_state()
        
        try:
            # Establish MCP server connection
            logger.info("Establishing connection to MCP server")
            server_params = StdioServerParameters(
                command="python",
                args=["src/keynote_mcp_server.py"]
            )
            
            async with stdio_client(server_params) as (read, write):
                logger.info("MCP connection established")
                
                async with ClientSession(read, write) as session:
                    logger.info("MCP session created")
                    await session.initialize()
                    
                    # Get available tools
                    tools_result = await session.list_tools()
                    tools = tools_result.tools
                    logger.info(f"Retrieved {len(tools)} available tools")
                    
                    # Create system prompt
                    system_prompt = self.create_system_prompt(tools)
                    
                    # Main execution loop
                    current_query = query
                    
                    while self.iteration < self.max_iterations:
                        logger.info(f"--- Iteration {self.iteration + 1} ---")
                        logger.info(f"Current phase: {self.current_phase}")
                        
                        # Prepare query for current iteration
                        if self.last_response is None:
                            current_query = query
                        else:
                            if self.current_phase == "computation":
                                current_query = query + "\n\n" + " ".join(self.iteration_responses)
                                current_query += " What should I do next?"
                            else:  # visualization phase
                                current_query = f"Proceed with visualization step {self.current_visualization_step + 1}: {self.visualization_steps[self.current_visualization_step]}"
                        
                        # Get LLM response
                        logger.info("Generating LLM response")
                        prompt = f"{system_prompt}\n\nQuery: {current_query}"
                        
                        try:
                            response_text = await self.generate_with_timeout(prompt)
                            logger.info(f"LLM Response: {response_text}")
                            
                            # Extract the relevant line from response
                            for line in response_text.split('\n'):
                                line = line.strip()
                                if line.startswith("FUNCTION_CALL:") or line.startswith("FINAL_ANSWER:"):
                                    response_text = line
                                    break
                            
                        except Exception as e:
                            logger.error(f"Failed to get LLM response: {e}")
                            break
                        
                        # Process response
                        if response_text.startswith("FUNCTION_CALL:"):
                            _, function_info = response_text.split(":", 1)
                            
                            # Execute tool call
                            result = await self.execute_tool_call(session, tools, function_info)
                            
                            if result["success"]:
                                # Record successful execution
                                self.iteration_responses.append(
                                    f"In iteration {self.iteration + 1}, you called {result['function_name']} "
                                    f"with parameters {result['arguments']}, and the function returned {result['formatted_result']}."
                                )
                                self.last_response = result["result"]
                                
                                # Update visualization step if in visualization phase
                                if self.current_phase == "visualization":
                                    self.current_visualization_step += 1
                                    if self.current_visualization_step >= len(self.visualization_steps):
                                        logger.info("Visualization complete!")
                                        break
                            else:
                                logger.error(f"Tool execution failed: {result['error']}")
                                self.iteration_responses.append(f"Error in iteration {self.iteration + 1}: {result['error']}")
                                break
                        
                        elif response_text.startswith("FINAL_ANSWER:"):
                            logger.info("=== Computation Complete ===")
                            if self.current_phase == "computation":
                                self.current_phase = "visualization"
                                logger.info("Transitioning to visualization phase")
                            self.last_response = response_text
                            continue
                        
                        self.iteration += 1
                    
                    return {
                        "success": True,
                        "final_answer": self.last_response,
                        "iterations": self.iteration_responses,
                        "total_iterations": self.iteration
                    }
                    
        except Exception as e:
            logger.error(f"Error in query execution: {e}")
            return {
                "success": False,
                "error": str(e),
                "iterations": self.iteration_responses
            }
        finally:
            self.reset_state()

async def main():
    """
    Main function demonstrating the AI client usage.
    """
    logger.info("Starting Keynote AI Client Demo")
    
    try:
        # Initialize client
        client = KeynoteAIClient()
        
        # Example query
        query = """Find the ASCII values of characters in 'AI' and then return the sum of exponentials of those values, then visualize the results in Keynote"""
        
        logger.info(f"Executing query: {query}")
        
        # Execute query
        result = await client.run_query(query)
        
        if result["success"]:
            logger.info("Query executed successfully!")
            logger.info(f"Final answer: {result['final_answer']}")
            logger.info(f"Total iterations: {result['total_iterations']}")
            
            print("\n" + "="*50)
            print("EXECUTION SUMMARY")
            print("="*50)
            print(f"Query: {query}")
            print(f"Final Answer: {result['final_answer']}")
            print(f"Total Iterations: {result['total_iterations']}")
            print("\nIteration Details:")
            for i, iteration in enumerate(result['iterations'], 1):
                print(f"{i}. {iteration}")
        else:
            logger.error(f"Query execution failed: {result['error']}")
            print(f"Error: {result['error']}")
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"Fatal error: {e}")

if __name__ == "__main__":
    asyncio.run(main())