// Base calibration diversity pool — ported from main2.py + expanded

export const calibrationPool = {
  technical: [
    "Large language models use transformer architectures to process sequential data efficiently.",
    "Neural network compression techniques include pruning, quantization, and knowledge distillation.",
    "The attention mechanism allows models to focus on relevant parts of the input sequence.",
    "Edge computing enables AI inference on resource-constrained mobile devices.",
    "Quantization reduces model precision from 32-bit floating point to 8-bit integers.",
    "Batch normalization helps stabilize training by normalizing intermediate activations.",
    "Gradient checkpointing trades compute for memory during backpropagation.",
    "Mixed-precision training uses FP16 for forward pass and FP32 for loss scaling.",
  ],
  reasoning: [
    "If all mammals are warm-blooded and whales are mammals, then whales must be warm-blooded.",
    "The fastest way to solve this problem is to break it down into smaller subproblems.",
    "Logical reasoning requires careful analysis of premises and valid inference patterns.",
    "Given that A implies B, and B implies C, we can conclude that A implies C by transitivity.",
    "The probability of two independent events both occurring equals the product of their individual probabilities.",
    "To prove a statement by contradiction, assume its negation leads to an impossible conclusion.",
  ],
  general: [
    "The solar system consists of the sun and eight planets orbiting around it.",
    "Photosynthesis is the process by which plants convert sunlight into chemical energy.",
    "The Industrial Revolution transformed manufacturing and transportation in the 18th century.",
    "Water molecules consist of two hydrogen atoms bonded to one oxygen atom.",
    "The speed of light in a vacuum is approximately 299,792 kilometers per second.",
    "DNA carries genetic information using sequences of four nucleotide bases.",
    "The human brain contains approximately 86 billion neurons connected by trillions of synapses.",
    "Plate tectonics explains the movement of Earth's lithospheric plates over geological time.",
  ],
  conversational: [
    "Hello! How can I help you today with your questions?",
    "That's a great question. Let me explain the concept in detail.",
    "I understand your concern. Here are some possible solutions to consider.",
    "Could you provide more context about what you're trying to accomplish?",
    "Let me summarize what we've discussed so far to make sure we're aligned.",
    "Thanks for your patience. I'll look into this and get back to you shortly.",
  ],
  creative: [
    "Once upon a time, in a distant galaxy, there lived a curious explorer.",
    "The sunset painted the sky in brilliant shades of orange and purple.",
    "Innovation drives progress and opens new possibilities for the future.",
    "She opened the old journal and found a map drawn in faded ink.",
    "The city hummed with energy as the festival lights flickered to life.",
    "In the quiet of the forest, every sound told a story of survival.",
  ],
  instructions: [
    "To solve this task, first identify the key requirements and constraints.",
    "Step by step, we can approach this problem systematically and efficiently.",
    "Let's analyze the situation carefully before making any decisions.",
    "Begin by gathering all relevant data, then organize it by priority.",
    "Follow these three rules to ensure consistent and reproducible results.",
    "Document your assumptions clearly so others can validate your approach.",
  ],
};

export function getPoolCounts() {
  return Object.fromEntries(
    Object.entries(calibrationPool).map(([k, v]) => [k, v.length])
  );
}

export function getTotalBaseCount() {
  return Object.values(calibrationPool).reduce((sum, arr) => sum + arr.length, 0);
}

export function getAllBaseSamples() {
  return Object.values(calibrationPool).flat();
}
