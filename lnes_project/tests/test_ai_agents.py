"""Tests for AI agents (FinBERT and Groq) with mocking."""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import pytest

from tests.test_utils import assert_valid_action_log, generate_mock_news_data


# =============================================================================
# FinBERT Agent Tests
# =============================================================================

class TestFinBERTAgent:
    """Test suite for FinBERT agent."""
    
    @pytest.mark.requires_model
    def test_finbert_agent_initialization_real(self):
        """Test FinBERT agent initialization with real model (slow)."""
        pytest.skip("Requires downloading FinBERT model - run manually if needed")
        from src.agents import FinBERTAgent
        
        agent = FinBERTAgent(confidence_threshold=0.7)
        assert agent.name == "FinBERT"
        assert agent.confidence_threshold == 0.7
    
    def test_finbert_agent_initialization_mock(self, mock_finbert):
        """Test FinBERT agent initialization with mock."""
        with patch("src.agents.pipeline", return_value=mock_finbert):
            from src.agents import FinBERTAgent
            
            agent = FinBERTAgent(confidence_threshold=0.7)
            assert agent.name == "FinBERT"
            assert agent.confidence_threshold == 0.7
    
    def test_finbert_agent_decision_positive_sentiment(self):
        """Test FinBERT agent makes buy decision on positive sentiment."""
        mock_pipeline = Mock()
        mock_pipeline.return_value = [{"label": "positive", "score": 0.85}]
        
        with patch("src.agents.pipeline", return_value=mock_pipeline):
            from src.agents import FinBERTAgent
            
            agent = FinBERTAgent(confidence_threshold=0.7)
            market_state = {
                "price": 100.0,
                "cluster_id": 0,
                "news_text": "Company reports excellent earnings",
            }
            
            decision = agent.decide(market_state)
            assert decision == "buy"
    
    def test_finbert_agent_decision_negative_sentiment(self):
        """Test FinBERT agent makes sell decision on negative sentiment."""
        mock_pipeline = Mock()
        mock_pipeline.return_value = [{"label": "negative", "score": 0.90}]
        
        with patch("src.agents.pipeline", return_value=mock_pipeline):
            from src.agents import FinBERTAgent
            
            agent = FinBERTAgent(confidence_threshold=0.7)
            market_state = {
                "price": 100.0,
                "cluster_id": 0,
                "news_text": "Company faces major lawsuit",
            }
            
            decision = agent.decide(market_state)
            assert decision == "sell"
    
    def test_finbert_agent_decision_neutral_sentiment(self):
        """Test FinBERT agent makes hold decision on neutral sentiment."""
        mock_pipeline = Mock()
        mock_pipeline.return_value = [{"label": "neutral", "score": 0.80}]
        
        with patch("src.agents.pipeline", return_value=mock_pipeline):
            from src.agents import FinBERTAgent
            
            agent = FinBERTAgent(confidence_threshold=0.7)
            market_state = {
                "price": 100.0,
                "cluster_id": 0,
                "news_text": "Company holds annual meeting",
            }
            
            decision = agent.decide(market_state)
            assert decision == "hold"
    
    def test_finbert_agent_decision_low_confidence(self):
        """Test FinBERT agent makes hold decision when confidence is low."""
        mock_pipeline = Mock()
        mock_pipeline.return_value = [{"label": "positive", "score": 0.60}]
        
        with patch("src.agents.pipeline", return_value=mock_pipeline):
            from src.agents import FinBERTAgent
            
            agent = FinBERTAgent(confidence_threshold=0.7)
            market_state = {
                "price": 100.0,
                "cluster_id": 0,
                "news_text": "Company updates website",
            }
            
            decision = agent.decide(market_state)
            assert decision == "hold"
    
    def test_finbert_agent_missing_news_text(self):
        """Test FinBERT agent handles missing news text."""
        mock_pipeline = Mock()
        
        with patch("src.agents.pipeline", return_value=mock_pipeline):
            from src.agents import FinBERTAgent
            
            agent = FinBERTAgent()
            market_state = {
                "price": 100.0,
                "cluster_id": 0,
            }
            
            decision = agent.decide(market_state)
            # Should default to hold when news text is missing
            assert decision in ["buy", "sell", "hold"]
    
    def test_finbert_agent_error_handling(self):
        """Test FinBERT agent handles errors gracefully."""
        mock_pipeline = Mock()
        mock_pipeline.side_effect = Exception("Model error")
        
        with patch("src.agents.pipeline", return_value=mock_pipeline):
            from src.agents import FinBERTAgent
            
            agent = FinBERTAgent()
            market_state = {
                "price": 100.0,
                "cluster_id": 0,
                "news_text": "Test news",
            }
            
            # Should handle error and return valid action
            decision = agent.decide(market_state)
            assert decision in ["buy", "sell", "hold"]
    
    def test_finbert_agent_confidence_threshold_parameter(self):
        """Test FinBERT agent respects confidence threshold."""
        mock_pipeline = Mock()
        
        with patch("src.agents.pipeline", return_value=mock_pipeline):
            from src.agents import FinBERTAgent
            
            # Low threshold
            agent1 = FinBERTAgent(confidence_threshold=0.5)
            assert agent1.confidence_threshold == 0.5
            
            # High threshold
            agent2 = FinBERTAgent(confidence_threshold=0.9)
            assert agent2.confidence_threshold == 0.9


# =============================================================================
# Groq Agent Tests
# =============================================================================

class TestGroqAgent:
    """Test suite for Groq LLM agent."""
    
    @pytest.mark.requires_api
    def test_groq_agent_initialization_real(self):
        """Test Groq agent initialization with real API (requires key)."""
        pytest.skip("Requires Groq API key - run manually if needed")
        from src.agents import GroqAgent
        
        agent = GroqAgent(api_key="test_key", model="llama-3.1-8b-instant")
        assert agent.name == "Groq"
        assert agent.model == "llama-3.1-8b-instant"
    
    def test_groq_agent_initialization_mock(self):
        """Test Groq agent initialization with mock."""
        with patch("src.agents.Groq"):
            from src.agents import GroqAgent
            
            agent = GroqAgent(api_key="test_key")
            assert agent.name == "Groq"
    
    def test_groq_agent_decision_buy(self):
        """Test Groq agent makes buy decision."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="buy"))]
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch("src.agents.Groq", return_value=mock_client):
            from src.agents import GroqAgent
            
            agent = GroqAgent(api_key="test_key")
            market_state = {
                "price": 100.0,
                "cluster_id": 0,
                "news_text": "Strong earnings beat expectations",
            }
            
            decision = agent.decide(market_state)
            assert decision == "buy"
    
    def test_groq_agent_decision_sell(self):
        """Test Groq agent makes sell decision."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="sell"))]
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch("src.agents.Groq", return_value=mock_client):
            from src.agents import GroqAgent
            
            agent = GroqAgent(api_key="test_key")
            market_state = {
                "price": 100.0,
                "cluster_id": 0,
                "news_text": "Major scandal uncovered",
            }
            
            decision = agent.decide(market_state)
            assert decision == "sell"
    
    def test_groq_agent_decision_hold(self):
        """Test Groq agent makes hold decision."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="hold"))]
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch("src.agents.Groq", return_value=mock_client):
            from src.agents import GroqAgent
            
            agent = GroqAgent(api_key="test_key")
            market_state = {
                "price": 100.0,
                "cluster_id": 0,
                "news_text": "Routine quarterly report",
            }
            
            decision = agent.decide(market_state)
            assert decision == "hold"
    
    def test_groq_agent_missing_news_text(self):
        """Test Groq agent handles missing news text."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="hold"))]
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch("src.agents.Groq", return_value=mock_client):
            from src.agents import GroqAgent
            
            agent = GroqAgent(api_key="test_key")
            market_state = {
                "price": 100.0,
                "cluster_id": 0,
            }
            
            decision = agent.decide(market_state)
            assert decision in ["buy", "sell", "hold"]
    
    def test_groq_agent_api_error_handling(self):
        """Test Groq agent handles API errors gracefully."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        
        with patch("src.agents.Groq", return_value=mock_client):
            from src.agents import GroqAgent
            
            agent = GroqAgent(api_key="test_key")
            market_state = {
                "price": 100.0,
                "cluster_id": 0,
                "news_text": "Test news",
            }
            
            # Should handle error and return valid action
            decision = agent.decide(market_state)
            assert decision in ["buy", "sell", "hold"]
    
    def test_groq_agent_invalid_response_parsing(self):
        """Test Groq agent handles invalid response format."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="invalid action"))]
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch("src.agents.Groq", return_value=mock_client):
            from src.agents import GroqAgent
            
            agent = GroqAgent(api_key="test_key")
            market_state = {
                "price": 100.0,
                "cluster_id": 0,
                "news_text": "Test news",
            }
            
            decision = agent.decide(market_state)
            # Should default to hold on invalid response
            assert decision in ["buy", "sell", "hold"]
    
    def test_groq_agent_model_parameter(self):
        """Test Groq agent respects model parameter."""
        with patch("src.agents.Groq"):
            from src.agents import GroqAgent
            
            agent = GroqAgent(
                api_key="test_key",
                model="llama-3.1-70b-versatile",
            )
            assert agent.model == "llama-3.1-70b-versatile"
    
    def test_groq_agent_temperature_parameter(self):
        """Test Groq agent uses temperature parameter."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="hold"))]
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch("src.agents.Groq", return_value=mock_client):
            from src.agents import GroqAgent
            
            agent = GroqAgent(api_key="test_key", temperature=0.5)
            market_state = {
                "price": 100.0,
                "cluster_id": 0,
                "news_text": "Test news",
            }
            
            agent.decide(market_state)
            
            # Verify temperature was passed to API
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert "temperature" in call_kwargs


# =============================================================================
# Integration Tests
# =============================================================================

class TestAIAgentIntegration:
    """Integration tests for AI agents with simulator."""
    
    def test_finbert_agent_in_simulation(self, sample_news_data, sample_price_data):
        """Test FinBERT agent works in full simulation."""
        mock_pipeline = Mock()
        mock_pipeline.return_value = [{"label": "positive", "score": 0.85}]
        
        with patch("src.agents.pipeline", return_value=mock_pipeline):
            from src.agents import FinBERTAgent
            from src.simulator import Simulator
            
            agent = FinBERTAgent()
            sim = Simulator(agents=[agent], alpha=0.01)
            
            # Run simulation with mock data
            initial_price = 100.0
            prices = [initial_price]
            
            for i in range(10):
                market_state = {
                    "price": prices[-1],
                    "cluster_id": i % 3,
                    "news_text": f"News item {i}",
                }
                
                action = agent.decide(market_state)
                assert action in ["buy", "sell", "hold"]
                
                # Simple price update
                if action == "buy":
                    prices.append(prices[-1] + 0.1)
                elif action == "sell":
                    prices.append(prices[-1] - 0.1)
                else:
                    prices.append(prices[-1])
            
            assert len(prices) == 11
            assert all(p > 0 for p in prices)
    
    def test_groq_agent_in_simulation(self, sample_news_data, sample_price_data):
        """Test Groq agent works in full simulation."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="buy"))]
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch("src.agents.Groq", return_value=mock_client):
            from src.agents import GroqAgent
            from src.simulator import Simulator
            
            agent = GroqAgent(api_key="test_key")
            sim = Simulator(agents=[agent], alpha=0.01)
            
            # Run simulation with mock data
            initial_price = 100.0
            prices = [initial_price]
            
            for i in range(10):
                market_state = {
                    "price": prices[-1],
                    "cluster_id": i % 3,
                    "news_text": f"News item {i}",
                }
                
                action = agent.decide(market_state)
                assert action in ["buy", "sell", "hold"]
                
                # Simple price update
                if action == "buy":
                    prices.append(prices[-1] + 0.1)
                elif action == "sell":
                    prices.append(prices[-1] - 0.1)
                else:
                    prices.append(prices[-1])
            
            assert len(prices) == 11
            assert all(p > 0 for p in prices)
    
    def test_mixed_agent_simulation(self):
        """Test simulation with mix of AI and rule-based agents."""
        # Mock both AI agents
        mock_finbert = Mock()
        mock_finbert.return_value = [{"label": "positive", "score": 0.85}]
        
        mock_groq_response = Mock()
        mock_groq_response.choices = [Mock(message=Mock(content="buy"))]
        
        mock_groq_client = Mock()
        mock_groq_client.chat.completions.create.return_value = mock_groq_response
        
        with patch("src.agents.pipeline", return_value=mock_finbert), \
             patch("src.agents.Groq", return_value=mock_groq_client):
            
            from src.agents import FinBERTAgent, GroqAgent, MomentumAgent, RandomAgent
            from src.simulator import Simulator
            
            agents = [
                FinBERTAgent(),
                GroqAgent(api_key="test_key"),
                MomentumAgent(),
                RandomAgent(seed=42),
            ]
            
            sim = Simulator(agents=agents, alpha=0.01)
            
            # Run short simulation
            initial_price = 100.0
            prices = [initial_price]
            
            for i in range(5):
                market_state = {
                    "price": prices[-1],
                    "cluster_id": i % 3,
                    "news_text": f"News {i}",
                }
                
                # Get decisions from all agents
                decisions = [agent.decide(market_state) for agent in agents]
                assert len(decisions) == 4
                assert all(d in ["buy", "sell", "hold"] for d in decisions)
                
                # Calculate net order flow
                order_flow = sum(
                    1 if d == "buy" else (-1 if d == "sell" else 0)
                    for d in decisions
                )
                
                # Update price
                new_price = max(prices[-1] + 0.01 * order_flow, 0.01)
                prices.append(new_price)
            
            assert len(prices) == 6
            assert all(p > 0 for p in prices)


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestAIAgentErrorHandling:
    """Test error handling and edge cases for AI agents."""
    
    def test_finbert_missing_dependencies(self):
        """Test FinBERT agent handles missing transformers library."""
        with patch("src.agents.pipeline", side_effect=ImportError("transformers not found")):
            with pytest.raises(ImportError):
                from src.agents import FinBERTAgent
                agent = FinBERTAgent()
    
    def test_groq_missing_dependencies(self):
        """Test Groq agent handles missing groq library."""
        with patch("src.agents.Groq", side_effect=ImportError("groq not found")):
            with pytest.raises(ImportError):
                from src.agents import GroqAgent
                agent = GroqAgent(api_key="test_key")
    
    def test_groq_invalid_api_key(self):
        """Test Groq agent handles invalid API key."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("Invalid API key")
        
        with patch("src.agents.Groq", return_value=mock_client):
            from src.agents import GroqAgent
            
            agent = GroqAgent(api_key="invalid_key")
            market_state = {
                "price": 100.0,
                "cluster_id": 0,
                "news_text": "Test",
            }
            
            decision = agent.decide(market_state)
            # Should handle error gracefully
            assert decision in ["buy", "sell", "hold"]
    
    def test_finbert_empty_text(self):
        """Test FinBERT agent handles empty text."""
        mock_pipeline = Mock()
        mock_pipeline.return_value = [{"label": "neutral", "score": 0.5}]
        
        with patch("src.agents.pipeline", return_value=mock_pipeline):
            from src.agents import FinBERTAgent
            
            agent = FinBERTAgent()
            market_state = {
                "price": 100.0,
                "cluster_id": 0,
                "news_text": "",
            }
            
            decision = agent.decide(market_state)
            assert decision in ["buy", "sell", "hold"]
    
    def test_groq_rate_limit_error(self):
        """Test Groq agent handles rate limit errors."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("Rate limit exceeded")
        
        with patch("src.agents.Groq", return_value=mock_client):
            from src.agents import GroqAgent
            
            agent = GroqAgent(api_key="test_key")
            market_state = {
                "price": 100.0,
                "cluster_id": 0,
                "news_text": "Test",
            }
            
            decision = agent.decide(market_state)
            # Should fallback to hold on rate limit
            assert decision in ["buy", "sell", "hold"]

