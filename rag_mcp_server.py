from mcp.server.fastmcp import FastMCP
from rag_service import ask_question

mcp = FastMCP("CollegeRAGServer")

@mcp.tool()
def query_college_documents(question: str) -> str:
    """Ask a question about college regulations, exams, or fees."""
    result = ask_question(question)
    sources = "\n".join([f"- {s['source']} (Page {s['page']})" for s in result['sources']])
    return f"Answer: {result['answer']}\n\nSources:\n{sources}"

if __name__ == "__main__":
    mcp.run()