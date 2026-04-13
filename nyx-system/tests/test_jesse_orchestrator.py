"""
TDD Tests — Agent 5: JesseOrchestrator (Meta)

Rôle: agrège les 4 agents en décision finale BUY/SELL/WAIT.
  - Tous passed → BUY ou SELL selon direction
  - Un agent block → WAIT
  - Sizing: P>0.75→1.5x, P>0.65→1.2x, P<0.60→0.75x

Output: OrchestratorDecision
"""
import pytest
import numpy as np
from src.agents.contracts import AgentResult, OrchestratorDecision


def make_agent_result(agent: str, state: str, score: float, passed: bool,
                      **metadata) -> AgentResult:
    return AgentResult(
        agent=agent, state=state, score=score, passed=passed,
        reason=f"{agent} {state}", metadata=metadata,
    )


class TestOrchestratorContract:
    def test_returns_orchestrator_decision(self):
        from src.ml.jesse_agents import JesseOrchestrator
        orch = JesseOrchestrator()
        ctx = make_agent_result('context', 'bullish', 0.8, True, p_bull=0.8, p_bear=0.1, p_neutral=0.1)
        reg = make_agent_result('regime', 'trend_plus', 0.7, True, p_trend=0.7, dominant_state='trend_plus', adx=0.4)
        stp = make_agent_result('setup', 'valid_setup', 0.65, True, p_setup=0.65, context_score=0.8, regime_score=0.7)
        ent = make_agent_result('entry', 'ready', 0.75, True, p_up=0.7, p_down=0.1, p_neutral=0.2, direction=1)
        decision = orch.decide(ctx, reg, stp, ent)
        assert isinstance(decision, OrchestratorDecision)

    def test_action_is_valid(self):
        from src.ml.jesse_agents import JesseOrchestrator
        orch = JesseOrchestrator()
        ctx = make_agent_result('context', 'bullish', 0.8, True, p_bull=0.8, p_bear=0.1, p_neutral=0.1)
        reg = make_agent_result('regime', 'trend_plus', 0.7, True, p_trend=0.7, dominant_state='trend_plus', adx=0.4)
        stp = make_agent_result('setup', 'valid_setup', 0.65, True, p_setup=0.65, context_score=0.8, regime_score=0.7)
        ent = make_agent_result('entry', 'ready', 0.75, True, p_up=0.7, p_down=0.1, p_neutral=0.2, direction=1)
        decision = orch.decide(ctx, reg, stp, ent)
        assert decision.action in ('BUY', 'SELL', 'WAIT')


class TestOrchestratorLogic:
    def test_all_passed_long_direction_is_buy(self):
        """All agents pass + direction=1 → BUY."""
        from src.ml.jesse_agents import JesseOrchestrator
        orch = JesseOrchestrator()
        ctx = make_agent_result('context', 'bullish', 0.8, True, p_bull=0.8, p_bear=0.1, p_neutral=0.1)
        reg = make_agent_result('regime', 'trend_plus', 0.7, True, p_trend=0.7, dominant_state='trend_plus', adx=0.4)
        stp = make_agent_result('setup', 'valid_setup', 0.65, True, p_setup=0.65, context_score=0.8, regime_score=0.7)
        ent = make_agent_result('entry', 'ready', 0.75, True, p_up=0.7, p_down=0.1, p_neutral=0.2, direction=1)
        decision = orch.decide(ctx, reg, stp, ent)
        assert decision.action == 'BUY'
        assert len(decision.blocked_by) == 0

    def test_all_passed_short_direction_is_sell(self):
        """All agents pass + direction=-1 → SELL."""
        from src.ml.jesse_agents import JesseOrchestrator
        orch = JesseOrchestrator()
        ctx = make_agent_result('context', 'bearish', 0.3, True, p_bull=0.1, p_bear=0.8, p_neutral=0.1)
        reg = make_agent_result('regime', 'trend_minus', 0.7, True, p_trend=0.7, dominant_state='trend_minus', adx=0.5)
        stp = make_agent_result('setup', 'valid_setup', 0.6, True, p_setup=0.6, context_score=0.3, regime_score=0.7)
        ent = make_agent_result('entry', 'ready', 0.7, True, p_up=0.1, p_down=0.7, p_neutral=0.2, direction=-1)
        decision = orch.decide(ctx, reg, stp, ent)
        assert decision.action == 'SELL'

    def test_one_agent_blocks_is_wait(self):
        """If any agent blocks → WAIT."""
        from src.ml.jesse_agents import JesseOrchestrator
        orch = JesseOrchestrator()
        ctx = make_agent_result('context', 'bearish', 0.3, False, p_bull=0.2, p_bear=0.7, p_neutral=0.1)  # BLOCKED
        reg = make_agent_result('regime', 'trend_plus', 0.7, True, p_trend=0.7, dominant_state='trend_plus', adx=0.4)
        stp = make_agent_result('setup', 'valid_setup', 0.65, True, p_setup=0.65, context_score=0.3, regime_score=0.7)
        ent = make_agent_result('entry', 'ready', 0.75, True, p_up=0.7, p_down=0.1, p_neutral=0.2, direction=1)
        decision = orch.decide(ctx, reg, stp, ent)
        assert decision.action == 'WAIT'
        assert 'context' in decision.blocked_by

    def test_multiple_blocks_lists_all(self):
        """Multiple blockers should all appear in blocked_by."""
        from src.ml.jesse_agents import JesseOrchestrator
        orch = JesseOrchestrator()
        ctx = make_agent_result('context', 'bearish', 0.3, False, p_bull=0.2, p_bear=0.7, p_neutral=0.1)
        reg = make_agent_result('regime', 'squeeze', 0.4, False, p_trend=0.3, dominant_state='squeeze', adx=0.1)
        stp = make_agent_result('setup', 'no_setup', 0.3, False, p_setup=0.3, context_score=0.3, regime_score=0.4)
        ent = make_agent_result('entry', 'ready', 0.75, True, p_up=0.7, p_down=0.1, p_neutral=0.2, direction=1)
        decision = orch.decide(ctx, reg, stp, ent)
        assert decision.action == 'WAIT'
        assert len(decision.blocked_by) >= 2


class TestOrchestratorSizing:
    def test_high_score_gets_1_5x(self):
        """Average score > 0.75 → size_factor = 1.5."""
        from src.ml.jesse_agents import JesseOrchestrator
        orch = JesseOrchestrator()
        ctx = make_agent_result('context', 'bullish', 0.9, True, p_bull=0.9, p_bear=0.05, p_neutral=0.05)
        reg = make_agent_result('regime', 'trend_plus', 0.85, True, p_trend=0.85, dominant_state='trend_plus', adx=0.6)
        stp = make_agent_result('setup', 'valid_setup', 0.8, True, p_setup=0.8, context_score=0.9, regime_score=0.85)
        ent = make_agent_result('entry', 'ready', 0.9, True, p_up=0.85, p_down=0.05, p_neutral=0.1, direction=1)
        decision = orch.decide(ctx, reg, stp, ent)
        assert decision.risk_analysis['size_factor'] == 1.5

    def test_low_score_gets_0_75x(self):
        """Average score < 0.60 → size_factor = 0.75."""
        from src.ml.jesse_agents import JesseOrchestrator
        orch = JesseOrchestrator()
        ctx = make_agent_result('context', 'bullish', 0.55, True, p_bull=0.55, p_bear=0.2, p_neutral=0.25)
        reg = make_agent_result('regime', 'range', 0.5, True, p_trend=0.4, dominant_state='range', adx=0.2)
        stp = make_agent_result('setup', 'valid_setup', 0.55, True, p_setup=0.55, context_score=0.55, regime_score=0.5)
        ent = make_agent_result('entry', 'ready', 0.5, True, p_up=0.5, p_down=0.2, p_neutral=0.3, direction=1)
        decision = orch.decide(ctx, reg, stp, ent)
        assert decision.risk_analysis['size_factor'] == 0.75

    def test_components_preserved(self):
        """Decision must contain all 4 agent results."""
        from src.ml.jesse_agents import JesseOrchestrator
        orch = JesseOrchestrator()
        ctx = make_agent_result('context', 'bullish', 0.7, True, p_bull=0.7, p_bear=0.15, p_neutral=0.15)
        reg = make_agent_result('regime', 'trend_plus', 0.7, True, p_trend=0.7, dominant_state='trend_plus', adx=0.4)
        stp = make_agent_result('setup', 'valid_setup', 0.65, True, p_setup=0.65, context_score=0.7, regime_score=0.7)
        ent = make_agent_result('entry', 'ready', 0.7, True, p_up=0.65, p_down=0.15, p_neutral=0.2, direction=1)
        decision = orch.decide(ctx, reg, stp, ent)
        assert 'context' in decision.components
        assert 'regime' in decision.components
        assert 'setup' in decision.components
        assert 'entry' in decision.components
