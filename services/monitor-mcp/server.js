const express = require('express');
const app = express();
const port = 8002;

app.use(express.json());

// Store for metrics and alerts
const metricsStore = [];
const alertsStore = [];

// Tool: Push a metric to the store
app.post('/push_metric', (req, res) => {
    const { name, value, labels } = req.body;
    if (!name || value === undefined) {
        return res.status(400).json({ error: 'Metric name and value are required' });
    }
    
    const metric = {
        name,
        value,
        labels: labels || {},
        timestamp: new Date().toISOString()
    };
    metricsStore.push(metric);
    console.log(`Pushed metric: ${name} = ${value}`);
    res.json({
        status: 'success',
        message: `Metric ${name} pushed successfully`
    });
});

// Tool: Define an alert
app.post('/define_alert', (req, res) => {
    const { type, message, timestamp } = req.body;
    if (!type || !message) {
        return res.status(400).json({ error: 'Alert type and message are required' });
    }
    
    const alert = {
        type,
        message,
        timestamp: timestamp || new Date().toISOString()
    };
    alertsStore.push(alert);
    console.log(`Defined alert: ${type} - ${message}`);
    res.json({
        status: 'success',
        message: `Alert ${type} defined successfully`
    });
});

// Tool: Get all alerts
app.get('/get_alerts', (req, res) => {
    console.log('Fetching all alerts');
    res.json({
        status: 'success',
        alerts: alertsStore
    });
});

// Start the server
app.listen(port, () => {
    console.log(`Monitor MCP Server running on port ${port}`);
});
