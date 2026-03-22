from research_agent.pipeline.research_orchestrator import ResearchOrchestrator

if __name__ == "__main__":
    agent = ResearchOrchestrator()
    final_report = agent.run("Invention of Umbrella")
    print("\n=== FINAL REPORT ===\n")
    print(final_report)