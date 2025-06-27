const express = require('express');
const app = express();
const port = 8003;

app.use(express.json());

// Store for evaluations
const evaluationsStore = [];

// Tool: Evaluate an output
app.post('/evaluate_output', (req, res) => {
    const { output, score, issues, context, timestamp } = req.body;
    if (!output || score === undefined) {
        return res.status(400).json({ error: 'Output and score are required' });
    }
    
    const evaluation = {
        output,
        score,
        issues: issues || [],
        context: context || {},
        timestamp: timestamp || new Date().toISOString()
    };
    evaluationsStore.push(evaluation);
    console.log(`Evaluated output with score: ${score}`);
    res.json({
        status: 'success',
        evaluation_id: evaluationsStore.length - 1,
        message: `Output evaluated with score ${score}`
    });
});

// Tool: Refine a prompt
app.post('/refine_prompt', (req, res) => {
    const { current_prompt, issues, strategy } = req.body;
    if (!current_prompt) {
        return res.status(400).json({ error: 'Current prompt is required' });
    }
    
    // Simple refinement logic (expand as needed)
    let refined_prompt = current_prompt;
    if (issues && issues.includes("Output too short") || strategy === "few-shot") {
        refined_prompt += "\n\nBeispiel: Bitte geben Sie eine ausführliche Antwort mit mindestens 100 Wörtern.";
    }
    
    console.log(`Refined prompt for issues: ${issues}`);
    res.json({
        status: 'success',
        refined_prompt,
        message: 'Prompt refined successfully'
    });
});

// Tool: Switch model based on performance
app.post('/switch_model', (req, res) => {
    const { current_model, performance } = req.body;
    if (!current_model) {
        return res.status(400).json({ error: 'Current model is required' });
    }
    
    // Simple model switching logic (expand as needed)
    let new_model = current_model;
    if (performance && performance.latency > 2.0) {
        new_model = 'ollama-gptq-v2-low-latency';
    }
    
    console.log(`Switched model from ${current_model} to ${new_model}`);
    res.json({
        status: 'success',
        name: new_model,
        message: `Model switched to ${new_model}`
    });
});

// Start the server
app.listen(port, () => {
    console.log(`Optimizer MCP Server running on port ${port}`);
});
