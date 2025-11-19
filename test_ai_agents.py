"""Quick test to verify AI agents work."""

print("🧪 Testing AI Agents...")

# Test 1: FinBERT (should work if transformers installed)
print("\n1️⃣ Testing FinBERT Agent (FREE, local)...")
try:
    from lnes_project.src.agents import FinBERTAgent

    agent = FinBERTAgent(name="test_finbert")
    decision = agent.decide(
        market_state={"price": 100.0, "prev_price": 99.0},
        cluster_id=0,
        news_text="Company reports strong earnings and positive outlook",
    )
    print(f"   ✅ FinBERT works! Decision: {decision}")
except ImportError as e:
    print(f"   ❌ Install transformers: pip install transformers torch")
    print(f"   Error: {e}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: Groq (requires API key)
print("\n2️⃣ Testing Groq Agent (FREE tier)...")
try:
    from lnes_project.src.agents import GroqAgent
    import os

    if not os.getenv("GROQ_API_KEY"):
        print("   ⚠️  GROQ_API_KEY not set")
        print("   📝 Get free key at: https://console.groq.com/keys")
        print("   🔧 Set with: export GROQ_API_KEY='your-key'")
    else:
        agent = GroqAgent(name="test_groq")
        decision = agent.decide(
            market_state={"price": 100.0, "prev_price": 99.0},
            cluster_id=0,
            news_text="Market crashes due to economic concerns",
        )
        print(f"   ✅ Groq works! Decision: {decision}")
except ImportError:
    print(f"   ❌ Install groq: pip install groq")
except ValueError as e:
    print(f"   ❌ {e}")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 70)
print("📚 Next steps:")
print("   1. Install AI deps: pip install transformers torch groq")
print("   2. Run basic test: python lnes_project/scripts/run_ai_experiment.py")
print("   3. Get Groq key: https://console.groq.com/keys")
print("   4. Read guide: lnes_project/AI_AGENTS_GUIDE.md")
print("=" * 70)



