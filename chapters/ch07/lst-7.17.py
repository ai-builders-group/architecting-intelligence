# --8<-- [start:lst-7.17]
# src/agent/notifiers/slack.py
import aiohttp
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class ProposalContext:
    action: str
    risk_level: str
    reasoning: str
    workflow_id: str
    estimated_cost: float | None = None

class HumanNotifier(ABC):
    @abstractmethod
    async def send_proposal(
        self,
        proposal: ProposalContext,
        callback_url: str,
    ) -> str:
        ...

class SlackNotifier(HumanNotifier):
    """
    Sends approval requests to a Slack channel via webhook.
    """
    
    def __init__(self, webhook_url: str, channel: str):
        self.webhook_url = webhook_url
        self.channel = channel
    
    async def send_proposal(
        self,
        proposal: ProposalContext,
        callback_url: str,
    ) -> str:
        """
        Post an interactive Slack message with Approve/Reject buttons.
        """
        message = {
            "channel": self.channel,
            "text": f"🤖 Agent Approval Request: {proposal.action}",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            f"*Proposed Action:* {proposal.action}\n"
                            f"*Risk Level:* {proposal.risk_level.upper()}\n"
                            f"*Estimated Cost:* ${proposal.estimated_cost or 0:.2f}\n\n"
                            f"*Reasoning:*\n{proposal.reasoning}"
                        ),
                    },
                },
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "✅ Approve"},
                            "style": "primary",
                            "value": proposal.workflow_id,
                            "action_id": "approve",
                        },
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "❌ Reject"},
                            "style": "danger",
                            "value": proposal.workflow_id,
                            "action_id": "reject",
                        },
                    ],
                },
            ],
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.webhook_url, json=message) as resp:
                resp.raise_for_status()
                result = await resp.json()
                return result["ts"]  # Slack message timestamp as ID
# --8<-- [end:lst-7.17]
