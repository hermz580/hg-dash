#!/bin/bash
# HF Generate CLI Demo Script
# Run this to see the capabilities of your new CLI tool

echo "🎨 Welcome to HF Generate Demo!"
echo "================================"

echo "🎭 Step 1: Display the beautiful banner"
python hf_generate.py splash

echo ""
echo "🧠 Step 2: Show available models"
python hf_generate.py models

echo ""
echo "⚙️ Step 3: Check configuration"
python hf_generate.py config

echo ""
echo "📚 Step 4: Show generation history (empty initially)"
python hf_generate.py history

echo ""
echo "🎯 Demo complete!"
echo ""
echo "Next steps:"
echo "1. Load a model: python hf_generate.py load-model gpt2"
echo "2. Try interactive mode: python hf_generate.py interactive"
echo "3. Generate text: python hf_generate.py generate 'Your prompt here'"
echo "4. Batch process: python hf_generate.py batch test_prompts.txt"
echo ""
echo "Happy generating! 🚀"