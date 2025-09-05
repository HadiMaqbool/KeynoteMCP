"""
Demo Usage Examples for Keynote MCP Server

This script demonstrates various ways to use the Keynote MCP Server
for mathematical computations and presentation automation.

Author: Senior AI Engineer Portfolio Project
License: MIT
"""

import asyncio
import logging
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ai_client import KeynoteAIClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_basic_mathematics():
    """Demonstrate basic mathematical operations."""
    print("\n" + "="*60)
    print("DEMO 1: Basic Mathematical Operations")
    print("="*60)
    
    client = KeynoteAIClient()
    
    queries = [
        "Calculate 15 + 27 and visualize the result in Keynote",
        "Find the factorial of 6 and create a presentation",
        "Compute the square root of 144 and show in Keynote"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n--- Query {i} ---")
        print(f"Query: {query}")
        
        try:
            result = await client.run_query(query)
            if result["success"]:
                print(f"✅ Success! Final answer: {result['final_answer']}")
                print(f"Total iterations: {result['total_iterations']}")
            else:
                print(f"❌ Failed: {result['error']}")
        except Exception as e:
            print(f"❌ Error: {e}")


async def demo_advanced_mathematics():
    """Demonstrate advanced mathematical operations."""
    print("\n" + "="*60)
    print("DEMO 2: Advanced Mathematical Operations")
    print("="*60)
    
    client = KeynoteAIClient()
    
    queries = [
        "Find ASCII values of 'HELLO' and sum their exponentials, then visualize in Keynote",
        "Generate first 8 Fibonacci numbers and create a presentation",
        "Calculate sine, cosine, and tangent of π/4 and show results in Keynote"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n--- Query {i} ---")
        print(f"Query: {query}")
        
        try:
            result = await client.run_query(query)
            if result["success"]:
                print(f"✅ Success! Final answer: {result['final_answer']}")
                print(f"Total iterations: {result['total_iterations']}")
            else:
                print(f"❌ Failed: {result['error']}")
        except Exception as e:
            print(f"❌ Error: {e}")


async def demo_string_processing():
    """Demonstrate string processing and ASCII operations."""
    print("\n" + "="*60)
    print("DEMO 3: String Processing and ASCII Operations")
    print("="*60)
    
    client = KeynoteAIClient()
    
    queries = [
        "Convert 'AI' to ASCII values and sum their exponentials, then visualize in Keynote",
        "Process the word 'MATH' and create a presentation with the results",
        "Analyze 'KEYNOTE' character by character and show in presentation"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n--- Query {i} ---")
        print(f"Query: {query}")
        
        try:
            result = await client.run_query(query)
            if result["success"]:
                print(f"✅ Success! Final answer: {result['final_answer']}")
                print(f"Total iterations: {result['total_iterations']}")
            else:
                print(f"❌ Failed: {result['error']}")
        except Exception as e:
            print(f"❌ Error: {e}")


async def demo_custom_queries():
    """Demonstrate custom user queries."""
    print("\n" + "="*60)
    print("DEMO 4: Custom User Queries")
    print("="*60)
    
    client = KeynoteAIClient()
    
    # Interactive mode
    print("\nEnter your own mathematical queries (type 'quit' to exit):")
    
    while True:
        try:
            query = input("\nYour query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            print(f"Processing: {query}")
            result = await client.run_query(query)
            
            if result["success"]:
                print(f"✅ Success! Final answer: {result['final_answer']}")
                print(f"Total iterations: {result['total_iterations']}")
                
                if result['iterations']:
                    print("\nExecution details:")
                    for i, iteration in enumerate(result['iterations'], 1):
                        print(f"  {i}. {iteration}")
            else:
                print(f"❌ Failed: {result['error']}")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


async def demo_performance_test():
    """Demonstrate performance with multiple operations."""
    print("\n" + "="*60)
    print("DEMO 5: Performance Test")
    print("="*60)
    
    client = KeynoteAIClient()
    
    import time
    
    queries = [
        "Calculate 2^10 and visualize in Keynote",
        "Find factorial of 7 and create presentation",
        "Compute square root of 256 and show in Keynote",
        "Calculate 100 + 200 and visualize result",
        "Find ASCII values of 'TEST' and sum exponentials"
    ]
    
    total_time = 0
    successful_queries = 0
    
    for i, query in enumerate(queries, 1):
        print(f"\n--- Performance Test {i}/5 ---")
        print(f"Query: {query}")
        
        start_time = time.time()
        
        try:
            result = await client.run_query(query)
            end_time = time.time()
            execution_time = end_time - start_time
            total_time += execution_time
            
            if result["success"]:
                successful_queries += 1
                print(f"✅ Success in {execution_time:.2f} seconds")
                print(f"Final answer: {result['final_answer']}")
            else:
                print(f"❌ Failed: {result['error']}")
                
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            total_time += execution_time
            print(f"❌ Error: {e}")
    
    print(f"\n--- Performance Summary ---")
    print(f"Total queries: {len(queries)}")
    print(f"Successful queries: {successful_queries}")
    print(f"Success rate: {(successful_queries/len(queries)*100):.1f}%")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Average time per query: {(total_time/len(queries)):.2f} seconds")


async def main():
    """Main demo function."""
    print("🎯 Keynote MCP Server - Demo Suite")
    print("=" * 60)
    print("This demo showcases the AI-powered presentation automation system.")
    print("Make sure you have:")
    print("1. Google Gemini API key configured")
    print("2. Keynote installed on macOS")
    print("3. All dependencies installed")
    print("=" * 60)
    
    try:
        # Run all demos
        await demo_basic_mathematics()
        await demo_advanced_mathematics()
        await demo_string_processing()
        await demo_performance_test()
        
        # Interactive demo
        await demo_custom_queries()
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        logger.exception("Demo execution failed")


if __name__ == "__main__":
    asyncio.run(main())