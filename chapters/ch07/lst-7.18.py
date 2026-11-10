# --8<-- [start:lst-7.18]
# src/agent/notifiers/email.py
import aiosmtplib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from email.mime.text import MIMEText

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

class EmailNotifier(HumanNotifier):
    """
    Sends approval requests via email with a magic link.
    """
    
    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        from_addr: str,
        to_addrs: list[str],
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.from_addr = from_addr
        self.to_addrs = to_addrs
    
    async def send_proposal(
        self,
        proposal: ProposalContext,
        callback_url: str,
    ) -> str:
        """
        Send a plaintext email with a link to the authenticated review page.
        """
        review_url = callback_url
        
        body = f"""
Agent Approval Request

Action: {proposal.action}
Risk Level: {proposal.risk_level.upper()}
Estimated Cost: ${proposal.estimated_cost or 0:.2f}

Reasoning:
{proposal.reasoning}

To approve or reject this action, sign in and open:
{review_url}

The decision is submitted from that page as an authenticated POST;
opening the link alone changes nothing.

This request will expire in 1 hour.
        """
        
        message = MIMEText(body)
        message["Subject"] = f"[Agent] Approval Required: {proposal.action[:50]}"
        message["From"] = self.from_addr
        message["To"] = ", ".join(self.to_addrs)
        
        await aiosmtplib.send(
            message,
            hostname=self.smtp_host,
            port=self.smtp_port,
        )
        
        return proposal.workflow_id  # Use workflow ID as notification ID
# --8<-- [end:lst-7.18]
