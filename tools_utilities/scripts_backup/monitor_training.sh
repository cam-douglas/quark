#!/bin/bash
# Budget Wikipedia Training Monitoring Script

echo "🔍 MONITORING WIKIPEDIA TRAINING"
echo "================================"

# Check namespace
echo "📊 Checking training namespace..."
kubectl get namespace quark-training

# Check pods
echo ""
echo "🖥️  Checking training pods..."
kubectl get pods -n quark-training

# Check pod details
echo ""
echo "📋 Pod details..."
kubectl describe pods -n quark-training

# Check logs
echo ""
echo "📝 Recent logs (last 50 lines)..."
kubectl logs -n quark-training -l app=wikipedia-training --tail=50

echo ""
echo "🔄 To monitor in real-time, run:"
echo "kubectl logs -n quark-training -l app=wikipedia-training -f"

echo ""
echo "💰 Estimated costs so far:"
echo "⏱️  Running time: Use 'kubectl get pods -n quark-training -o wide' to check start time"
echo "💸 Cost rate: ~$0.20-0.40 per hour for 2x g4dn.2xlarge spot instances"

echo ""
echo "🛑 To stop training and save costs:"
echo "kubectl delete namespace quark-training"
