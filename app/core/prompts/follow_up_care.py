from app.core.prompts.live_prompt import LivePromptObject


system_prompt:str = LivePromptObject("follow_up_care/system_prompt.txt")
neo4j_query_prompt:str = LivePromptObject("follow_up_care/neo4j_query_prompt.txt")
graph_rag_prompt:str = LivePromptObject("follow_up_care/graph_rag_prompt.txt")
