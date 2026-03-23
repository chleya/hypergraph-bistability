"""LLM Transport Adapters.

This module provides abstraction over different LLM APIs (OpenAI, Anthropic, MiniMax).
Extracted from hypergraph_agent.py to reduce file size.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


class LLMTransport:
    """Base class for LLM API transports."""
    
    def __init__(self, agent):
        self.agent = agent
    
    def call(self, messages: List[Dict[str, str]]) -> str:
        raise NotImplementedError


class OpenAIClientTransport(LLMTransport):
    """OpenAI client-based transport."""
    
    def call(self, messages: List[Dict[str, str]]) -> str:
        return self.agent._call_llm_via_client(messages)


class AnthropicHTTPTransport(LLMTransport):
    """Anthropic HTTP transport."""
    
    def call(self, messages: List[Dict[str, str]]) -> str:
        return self.agent._call_llm_via_anthropic_http(messages)


class PowerShellTransport(LLMTransport):
    """PowerShell-based transport for Windows compatibility."""
    
    def call(self, messages: List[Dict[str, str]], transport: str = "openai") -> str:
        return self.agent._call_llm_via_powershell_transport(messages)


class MiniMaxPowerShellTransport(LLMTransport):
    """MiniMax-specific PowerShell transport."""
    
    def call(self, messages: List[Dict[str, str]]) -> str:
        return self.agent._call_llm_via_powershell_anthropic_transport(messages)


class LLMTransportManager:
    """Manages LLM transport selection and configuration."""
    
    TRANSPORTS = {
        "openai_client": OpenAIClientTransport,
        "anthropic_http": AnthropicHTTPTransport,
        "powershell": PowerShellTransport,
        "minimax_powershell": MiniMaxPowerShellTransport,
    }
    
    def __init__(self, agent):
        self.agent = agent
        self._transport_cache: Dict[str, LLMTransport] = {}
    
    def get_transport(self, transport_name: str) -> LLMTransport:
        """Get or create a transport instance."""
        if transport_name not in self._transport_cache:
            transport_class = self.TRANSPORTS.get(transport_name)
            if transport_class is None:
                raise ValueError(f"Unknown transport: {transport_name}")
            self._transport_cache[transport_name] = transport_class(self.agent)
        return self._transport_cache[transport_name]
    
    def detect_transport(self, base_url: Optional[str] = None) -> str:
        """Detect appropriate transport based on configuration."""
        return self.agent._detect_llm_transport(base_url)
    
    def should_use_powershell(self, error: Exception) -> bool:
        """Determine if PowerShell fallback should be used."""
        return self.agent._should_use_powershell_transport(error)


def create_llm_transport_manager(agent) -> LLMTransportManager:
    """Factory function to create LLM transport manager."""
    return LLMTransportManager(agent)
