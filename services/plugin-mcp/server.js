const express = require('express');
const app = express();
const port = 8004;

app.use(express.json());

// Mock data for plugins
const plugins = [
    { name: 'data-pipeline', version: '1.0.0', auto_install: false },
    { name: 'alert-notifier', version: '2.1.0', auto_install: true },
    { name: 'performance-metrics', version: '1.2.3', auto_install: false }
];

// Tool: Install a plugin
app.post('/install_plugin', (req, res) => {
    const { name, version } = req.body;
    if (!name) {
        return res.status(400).json({ error: 'Plugin name is required' });
    }
    
    const plugin = plugins.find(p => p.name === name);
    if (!plugin) {
        return res.status(404).json({ error: `Plugin ${name} not found` });
    }
    
    if (version && version !== plugin.version) {
        plugin.version = version;
    }
    
    console.log(`Installed plugin: ${name} version ${plugin.version}`);
    res.json({
        status: 'success',
        plugin: { name: plugin.name, version: plugin.version },
        message: `Plugin ${name} installed successfully`
    });
});

// Tool: List all available plugins
app.get('/list_plugins', (req, res) => {
    console.log('Listing all plugins');
    res.json({
        status: 'success',
        plugins
    });
});

// Tool: Update a plugin
app.post('/update_plugin', (req, res) => {
    const { name, version } = req.body;
    if (!name) {
        return res.status(400).json({ error: 'Plugin name is required' });
    }
    
    const pluginIndex = plugins.findIndex(p => p.name === name);
    if (pluginIndex === -1) {
        return res.status(404).json({ error: `Plugin ${name} not found` });
    }
    
    if (version) {
        plugins[pluginIndex].version = version;
    }
    
    console.log(`Updated plugin: ${name} to version ${plugins[pluginIndex].version}`);
    res.json({
        status: 'success',
        plugin: plugins[pluginIndex],
        message: `Plugin ${name} updated successfully`
    });
});

// Start the server
app.listen(port, () => {
    console.log(`Plugin MCP Server running on port ${port}`);
});
