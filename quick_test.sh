#!/bin/bash

# 🎨 HF Generate - Quick Test Script
echo "🚀 Testing HF Generate CLI Tool..."
echo ""

# Test 1: Generate a simple text
echo "📝 Test 1: Simple generation with default settings"
python hf_generate.py generate "Once upon a time in a land far away" --max-length 60

echo ""
echo "✅ Test completed!"
echo ""

# Test 2: Generate with specific model and style
echo "📝 Test 2: Generation with custom parameters"  
python hf_generate.py generate "The secret to happiness is" --model distilgpt2 --style precise --max-length 80

echo ""
echo "✅ All tests completed! 🎉"
echo ""
echo "🎯 Try these commands:"
echo "  • Interactive mode: python hf_generate.py interactive"
echo "  • View history: python hf_generate.py history"
echo "  • Show stats: python hf_generate.py stats"