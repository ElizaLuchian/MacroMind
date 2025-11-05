from src.agents import ContrarianAgent, MomentumAgent, NewsReactiveAgent, RandomAgent


def test_momentum_agent_buy_sell_logic():
    agent = MomentumAgent(name="momentum")
    assert agent.decide({"price": 2.0, "prev_price": 1.0}, None) == "buy"
    assert agent.decide({"price": 1.0, "prev_price": 2.0}, None) == "sell"


def test_contrarian_inverts_momentum():
    agent = ContrarianAgent(name="contrarian")
    assert agent.decide({"price": 2.0, "prev_price": 1.0}, None) == "sell"


def test_news_reactive_agent_uses_sentiment():
    agent = NewsReactiveAgent(name="news", cluster_sentiment={0: 0.3, 1: -0.4})
    assert agent.decide({}, 0) == "buy"
    assert agent.decide({}, 1) == "sell"
    assert agent.decide({}, 2) == "hold"


def test_random_agent_outputs_valid_action():
    agent = RandomAgent(name="random", seed=0)
    assert agent.decide({}, None) in {"buy", "sell", "hold"}


