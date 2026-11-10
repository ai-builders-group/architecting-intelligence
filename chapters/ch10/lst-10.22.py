# --8<-- [start:lst-10.22]
# architecting_intelligence/ch09/community_detector.py
from typing import Protocol
from pydantic import BaseModel, Field

class CommunityResult(BaseModel):
    community_id: int
    member_count: int
    modularity: float

async def run_leiden_gds(
    driver, graph_name: str = "knowledge_graph") -> list[CommunityResult]:
    """
    Execute Leiden algorithm inside the Neo4j GDS kernel.
    Do NOT pull the graph into Python memory.
    """
    query = """
    CALL gds.leiden.write($graph_name, {
        writeProperty: 'community_id',
        includeIntermediateCommunities: false,
        gamma: 1.0,
        theta: 0.01
    })
    YIELD communityCount, modularity, writeMillis
    RETURN communityCount, modularity
    """
    async with driver.session() as session:
        result = await session.run(query, graph_name=graph_name)
        record = await result.single()
        return CommunityResult(
            community_id=-1,  # Batch result
            member_count=record["communityCount"],
            modularity=record["modularity"]
        )

# --8<-- [end:lst-10.22]
