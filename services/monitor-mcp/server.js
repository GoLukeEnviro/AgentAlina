import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  ListResourcesRequestSchema,
  ReadResourceRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";

// Store for metrics and alerts
const metricsStore = [];
const alertsStore = [];

// Create MCP server
const server = new Server(
  {
    name: "monitor-mcp",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
      resources: {},
    },
  }
);

// Define tool schemas
const PushMetricSchema = z.object({
  name: z.string().describe("Name of the metric"),
  value: z.number().describe("Numeric value of the metric"),
  labels: z.record(z.string()).optional().describe("Optional labels for the metric"),
});

const DefineAlertSchema = z.object({
  type: z.string().describe("Type of alert (info, warning, error, critical)"),
  message: z.string().describe("Alert message"),
  timestamp: z.string().optional().describe("Optional timestamp (ISO string)"),
});

// List available tools
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "push_metric",
        description: "Push a metric to the monitoring store",
        inputSchema: PushMetricSchema,
      },
      {
        name: "define_alert",
        description: "Define an alert in the monitoring system",
        inputSchema: DefineAlertSchema,
      },
      {
        name: "get_metrics",
        description: "Retrieve all stored metrics",
        inputSchema: z.object({}),
      },
      {
        name: "get_alerts",
        description: "Retrieve all stored alerts",
        inputSchema: z.object({}),
      },
    ],
  };
});

// Handle tool calls
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  switch (name) {
    case "push_metric": {
      const { name: metricName, value, labels } = PushMetricSchema.parse(args);
      
      const metric = {
        name: metricName,
        value,
        labels: labels || {},
        timestamp: new Date().toISOString(),
      };
      
      metricsStore.push(metric);
      console.error(`Pushed metric: ${metricName} = ${value}`);
      
      return {
        content: [
          {
            type: "text",
            text: `Metric ${metricName} pushed successfully with value ${value}`,
          },
        ],
      };
    }

    case "define_alert": {
      const { type, message, timestamp } = DefineAlertSchema.parse(args);
      
      const alert = {
        type,
        message,
        timestamp: timestamp || new Date().toISOString(),
      };
      
      alertsStore.push(alert);
      console.error(`Defined alert: ${type} - ${message}`);
      
      return {
        content: [
          {
            type: "text",
            text: `Alert ${type} defined successfully: ${message}`,
          },
        ],
      };
    }

    case "get_metrics": {
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(metricsStore, null, 2),
          },
        ],
      };
    }

    case "get_alerts": {
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(alertsStore, null, 2),
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
        uri: "monitor://metrics",
        name: "Current Metrics",
        description: "All stored metrics in the monitoring system",
        mimeType: "application/json",
      },
      {
        uri: "monitor://alerts",
        name: "Current Alerts",
        description: "All stored alerts in the monitoring system",
        mimeType: "application/json",
      },
    ],
  };
});

// Handle resource reads
server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
  const { uri } = request.params;

  switch (uri) {
    case "monitor://metrics":
      return {
        contents: [
          {
            uri,
            mimeType: "application/json",
            text: JSON.stringify(metricsStore, null, 2),
          },
        ],
      };

    case "monitor://alerts":
      return {
        contents: [
          {
            uri,
            mimeType: "application/json",
            text: JSON.stringify(alertsStore, null, 2),
          },
        ],
      };

    default:
      throw new Error(`Unknown resource: ${uri}`);
  }
});

// Main function to start the server
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("Monitor MCP Server running on stdio");
}

// Start the server
main().catch((error) => {
  console.error("Fatal error in main():", error);
  process.exit(1);
});
