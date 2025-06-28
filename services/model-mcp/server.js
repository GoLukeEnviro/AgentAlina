const express = require('express');
const app = express();
const port = 8001;

app.use(express.json());

// Mock data for models
const models = [
    { name: 'ollama-gptq-v1', version: '1.0.0', quantized: true, size: '4bit' },
    { name: 'ollama-gptq-v2', version: '2.0.0', quantized: true, size: '8bit' },
    { name: 'llama-index', version: '1.1.0', quantized: false }
];

// Tool: Fetch a model by name
app.post('/fetch_model', (req, res) => {
    const { name } = req.body;
    if (!name) {
        return res.status(400).json({ error: 'Model name is required' });
    }
    
    const model = models.find(m => m.name === name);
    if (!model) {
        return res.status(404).json({ error: `Model ${name} not found` });
    }
    
    console.log(`Fetched model: ${name}`);
    res.json({
        status: 'success',
        model: model,
        message: `Model ${name} fetched successfully`
    });
});

// Tool: List all available model versions
app.get('/list_versions', (req, res) => {
    console.log('Listing all model versions');
    res.json({
        status: 'success',
        models: models.map(m => ({ name: m.name, version: m.version, quantized: m.quantized }))
    });
});

// Tool: Quantize a model
app.post('/quantize_model', (req, res) => {
    const { name, bits } = req.body;
    if (!name || !bits) {
        return res.status(400).json({ error: 'Model name and quantization bits are required' });
    }
    
    const modelIndex = models.findIndex(m => m.name === name);
    if (modelIndex === -1) {
        return res.status(404).json({ error: `Model ${name} not found` });
    }
    
    models[modelIndex].quantized = true;
    models[modelIndex].size = `${bits}bit`;
    console.log(`Quantized model: ${name} to ${bits}bit`);
    
    res.json({
        status: 'success',
        model: models[modelIndex],
        message: `Model ${name} quantized to ${bits}bit`
    });
});

// Start the server
app.listen(port, () => {
    console.log(`Model MCP Server running on port ${port}`);
});
