import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  ListResourcesRequestSchema,
  ReadResourceRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import { z } from 'zod';

// Create server instance
const server = new Server(
  {
    name: 'plugin-mcp',
    version: '0.1.0',
  },
  {
    capabilities: {
      tools: {},
      resources: {},
    },
  }
);

// Store for plugins
let pluginsStore = [];
let executionHistory = [];
let pluginRegistry = new Map();

// Define input schemas
const LoadPluginSchema = z.object({
  name: z.string().describe('Plugin name'),
  path: z.string().describe('Path to plugin file'),
  config: z.record(z.any()).optional().describe('Plugin configuration'),
  autoStart: z.boolean().optional().describe('Auto-start plugin after loading'),
});

const ExecutePluginSchema = z.object({
  pluginName: z.string().describe('Name of the plugin to execute'),
  method: z.string().describe('Method to call on the plugin'),
  args: z.array(z.any()).optional().describe('Arguments to pass to the method'),
  timeout: z.number().optional().describe('Execution timeout in milliseconds'),
});

const GetPluginsSchema = z.object({
  status: z.enum(['all', 'loaded', 'active', 'error']).optional().describe('Filter by status'),
  category: z.string().optional().describe('Filter by plugin category'),
});

const UnloadPluginSchema = z.object({
  pluginName: z.string().describe('Name of the plugin to unload'),
  force: z.boolean().optional().describe('Force unload even if plugin is active'),
});

// List available tools
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: 'load_plugin',
        description: 'Load a plugin from file system',
        inputSchema: LoadPluginSchema,
      },
      {
        name: 'execute_plugin',
        description: 'Execute a method on a loaded plugin',
        inputSchema: ExecutePluginSchema,
      },
      {
        name: 'get_plugins',
        description: 'List all loaded plugins and their status',
        inputSchema: GetPluginsSchema,
      },
      {
        name: 'unload_plugin',
        description: 'Unload a plugin from memory',
        inputSchema: UnloadPluginSchema,
      },
    ],
  };
});

// Handle tool calls
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  switch (name) {
    case 'load_plugin': {
      const { name: pluginName, path, config, autoStart } = LoadPluginSchema.parse(args);
      
      // Check if plugin already loaded
      const existingPlugin = pluginsStore.find(p => p.name === pluginName);
      if (existingPlugin) {
        return {
          content: [
            {
              type: 'text',
              text: `Plugin '${pluginName}' is already loaded. Status: ${existingPlugin.status}`,
            },
          ],
        };
      }
      
      const plugin = {
        id: Date.now(),
        name: pluginName,
        path,
        config: config || {},
        status: 'loaded',
        timestamp: new Date().toISOString(),
        version: '1.0.0',
        category: detectPluginCategory(path),
        methods: generatePluginMethods(pluginName),
        autoStart: autoStart || false,
      };
      
      pluginsStore.push(plugin);
      pluginRegistry.set(pluginName, plugin);
      
      if (autoStart) {
        plugin.status = 'active';
      }
      
      return {
        content: [
          {
            type: 'text',
            text: `Plugin loaded successfully:\n\nName: ${pluginName}\nPath: ${path}\nCategory: ${plugin.category}\nStatus: ${plugin.status}\nMethods: ${plugin.methods.join(', ')}\nAuto-start: ${plugin.autoStart}\n\nPlugin ID: ${plugin.id}`,
          },
        ],
      };
    }

    case 'execute_plugin': {
      const { pluginName, method, args: methodArgs, timeout } = ExecutePluginSchema.parse(args);
      
      const plugin = pluginRegistry.get(pluginName);
      if (!plugin) {
        throw new Error(`Plugin '${pluginName}' not found. Load it first using load_plugin.`);
      }
      
      if (!plugin.methods.includes(method)) {
        throw new Error(`Method '${method}' not available in plugin '${pluginName}'. Available methods: ${plugin.methods.join(', ')}`);
      }
      
      const execution = {
        id: Date.now(),
        pluginName,
        method,
        args: methodArgs || [],
        timeout: timeout || 5000,
        startTime: new Date().toISOString(),
        status: 'running',
      };
      
      // Simulate plugin execution
      const executionTime = Math.random() * 2000 + 100;
      const success = Math.random() > 0.1; // 90% success rate
      
      setTimeout(() => {
        execution.status = success ? 'completed' : 'failed';
        execution.endTime = new Date().toISOString();
        execution.executionTime = executionTime;
        execution.result = success 
          ? generateExecutionResult(pluginName, method, methodArgs)
          : 'Execution failed: Plugin error occurred';
      }, 100);
      
      executionHistory.push(execution);
      
      return {
        content: [
          {
            type: 'text',
            text: `Plugin execution initiated:\n\nPlugin: ${pluginName}\nMethod: ${method}\nArguments: ${JSON.stringify(methodArgs || [])}\nTimeout: ${execution.timeout}ms\nExecution ID: ${execution.id}\n\nStatus: ${execution.status}`,
          },
        ],
      };
    }

    case 'get_plugins': {
      const { status = 'all', category } = GetPluginsSchema.parse(args);
      
      let results = [...pluginsStore];
      
      if (status !== 'all') {
        results = results.filter(plugin => plugin.status === status);
      }
      
      if (category) {
        results = results.filter(plugin => plugin.category === category);
      }
      
      const summary = {
        total: results.length,
        byStatus: {
          loaded: results.filter(p => p.status === 'loaded').length,
          active: results.filter(p => p.status === 'active').length,
          error: results.filter(p => p.status === 'error').length,
        },
        byCategory: {},
      };
      
      // Count by category
      results.forEach(plugin => {
        summary.byCategory[plugin.category] = (summary.byCategory[plugin.category] || 0) + 1;
      });
      
      return {
        content: [
          {
            type: 'text',
            text: `Plugin Registry Summary:\n\nTotal Plugins: ${summary.total}\nLoaded: ${summary.byStatus.loaded}\nActive: ${summary.byStatus.active}\nError: ${summary.byStatus.error}\n\nBy Category:\n${Object.entries(summary.byCategory).map(([cat, count]) => `• ${cat}: ${count}`).join('\n')}\n\nPlugins:\n${results.map(p => 
              `• ${p.name} (${p.category}) - ${p.status} - ${p.methods.length} methods`
            ).join('\n')}`,
          },
        ],
      };
    }

    case 'unload_plugin': {
      const { pluginName, force } = UnloadPluginSchema.parse(args);
      
      const pluginIndex = pluginsStore.findIndex(p => p.name === pluginName);
      if (pluginIndex === -1) {
        throw new Error(`Plugin '${pluginName}' not found.`);
      }
      
      const plugin = pluginsStore[pluginIndex];
      
      if (plugin.status === 'active' && !force) {
        throw new Error(`Plugin '${pluginName}' is currently active. Use force=true to unload.`);
      }
      
      pluginsStore.splice(pluginIndex, 1);
      pluginRegistry.delete(pluginName);
      
      return {
        content: [
          {
            type: 'text',
            text: `Plugin '${pluginName}' unloaded successfully.\n\nPrevious status: ${plugin.status}\nForced: ${force || false}`,
          },
        ],
      };
    }

    default:
      throw new Error(`Unknown tool: ${name}`);
  }
});

// List available resources
server.setRequestHandler(ListResourcesRequestSchema, async () => {
  return {
    resources: [
      {
        uri: "plugin://registry",
        name: "Plugin Registry",
        description: "Complete registry of all loaded plugins",
        mimeType: "application/json",
      },
      {
        uri: "plugin://execution-history",
        name: "Execution History",
        description: "History of all plugin executions",
        mimeType: "application/json",
      },
    ],
  };
});

// Handle resource reads
server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
  const { uri } = request.params;

  switch (uri) {
    case "plugin://registry":
      return {
        contents: [
          {
            uri,
            mimeType: "application/json",
            text: JSON.stringify({
              plugins: pluginsStore,
              summary: {
                totalPlugins: pluginsStore.length,
                activePlugins: pluginsStore.filter(p => p.status === 'active').length,
                categories: [...new Set(pluginsStore.map(p => p.category))],
              }
            }, null, 2),
          },
        ],
      };

    case "plugin://execution-history":
      return {
        contents: [
          {
            uri,
            mimeType: "application/json",
            text: JSON.stringify(executionHistory, null, 2),
          },
        ],
      };

    default:
      throw new Error(`Unknown resource: ${uri}`);
  }
});

// Helper functions
function detectPluginCategory(path) {
  if (path.includes('ai') || path.includes('ml')) return 'AI/ML';
  if (path.includes('data') || path.includes('db')) return 'Data';
  if (path.includes('web') || path.includes('http')) return 'Web';
  if (path.includes('util') || path.includes('tool')) return 'Utility';
  return 'General';
}

function generatePluginMethods(pluginName) {
  const commonMethods = ['init', 'execute', 'cleanup'];
  const specificMethods = {
    'ai': ['predict', 'train', 'evaluate'],
    'data': ['process', 'transform', 'validate'],
    'web': ['request', 'parse', 'scrape'],
    'util': ['format', 'convert', 'validate'],
  };
  
  const category = pluginName.toLowerCase();
  for (const [key, methods] of Object.entries(specificMethods)) {
    if (category.includes(key)) {
      return [...commonMethods, ...methods];
    }
  }
  
  return commonMethods;
}

function generateExecutionResult(pluginName, method, args) {
  const results = {
    'init': `Plugin ${pluginName} initialized successfully`,
    'execute': `Executed ${method} with ${args?.length || 0} arguments`,
    'cleanup': `Plugin ${pluginName} cleaned up resources`,
    'predict': `Prediction completed with confidence: ${(Math.random() * 100).toFixed(2)}%`,
    'train': `Training completed with accuracy: ${(Math.random() * 100).toFixed(2)}%`,
    'process': `Processed ${Math.floor(Math.random() * 1000)} items`,
    'request': `HTTP request completed with status: 200`,
  };
  
  return results[method] || `Method ${method} executed successfully`;
}

// Main function to start the server
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error('Plugin MCP server running on stdio');
}

// Start the server
main().catch(console.error);
