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
    name: 'autogen-mcp',
    version: '0.1.0',
  },
  {
    capabilities: {
      tools: {},
      resources: {},
    },
  }
);

// Store for autogen data
let agentsStore = [];
let conversationsStore = [];
let workflowsStore = [];

// Define input schemas
const CreateAgentSchema = z.object({
  name: z.string().describe('Agent name'),
  role: z.string().describe('Agent role (e.g., assistant, user, critic)'),
  systemMessage: z.string().describe('System message for the agent'),
  model: z.string().optional().describe('Model to use for this agent'),
  capabilities: z.array(z.string()).optional().describe('Agent capabilities'),
});

const StartConversationSchema = z.object({
  agents: z.array(z.string()).describe('List of agent names to include'),
  topic: z.string().describe('Conversation topic or initial message'),
  maxRounds: z.number().optional().describe('Maximum conversation rounds'),
  moderator: z.string().optional().describe('Moderator agent name'),
});

const CreateWorkflowSchema = z.object({
  name: z.string().describe('Workflow name'),
  description: z.string().describe('Workflow description'),
  steps: z.array(z.object({
    agent: z.string().describe('Agent responsible for this step'),
    task: z.string().describe('Task description'),
    dependencies: z.array(z.string()).optional().describe('Dependencies on other steps'),
  })).describe('Workflow steps'),
  parallel: z.boolean().optional().describe('Allow parallel execution'),
});

const ExecuteWorkflowSchema = z.object({
  workflowName: z.string().describe('Name of workflow to execute'),
  inputs: z.record(z.any()).optional().describe('Input parameters for workflow'),
  timeout: z.number().optional().describe('Execution timeout in seconds'),
});

// List available tools
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: 'create_agent',
        description: 'Create a new AutoGen agent with specific role and capabilities',
        inputSchema: CreateAgentSchema,
      },
      {
        name: 'start_conversation',
        description: 'Start a multi-agent conversation on a specific topic',
        inputSchema: StartConversationSchema,
      },
      {
        name: 'create_workflow',
        description: 'Create a multi-agent workflow with defined steps',
        inputSchema: CreateWorkflowSchema,
      },
      {
        name: 'execute_workflow',
        description: 'Execute a predefined workflow with given inputs',
        inputSchema: ExecuteWorkflowSchema,
      },
    ],
  };
});

// Handle tool calls
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  switch (name) {
    case 'create_agent': {
      const { name: agentName, role, systemMessage, model, capabilities } = CreateAgentSchema.parse(args);
      
      // Check if agent already exists
      const existingAgent = agentsStore.find(a => a.name === agentName);
      if (existingAgent) {
        throw new Error(`Agent '${agentName}' already exists.`);
      }
      
      const agent = {
        id: Date.now(),
        name: agentName,
        role,
        systemMessage,
        model: model || 'gpt-4',
        capabilities: capabilities || [],
        status: 'active',
        createdAt: new Date().toISOString(),
        conversationCount: 0,
        lastActive: new Date().toISOString(),
      };
      
      agentsStore.push(agent);
      
      return {
        content: [
          {
            type: 'text',
            text: `Agent created successfully:\n\nName: ${agentName}\nRole: ${role}\nModel: ${agent.model}\nCapabilities: ${agent.capabilities.join(', ') || 'None'}\nSystem Message: ${systemMessage}\n\nAgent ID: ${agent.id}`,
          },
        ],
      };
    }

    case 'start_conversation': {
      const { agents, topic, maxRounds, moderator } = StartConversationSchema.parse(args);
      
      // Validate agents exist
      const missingAgents = agents.filter(name => !agentsStore.find(a => a.name === name));
      if (missingAgents.length > 0) {
        throw new Error(`Agents not found: ${missingAgents.join(', ')}`);
      }
      
      const conversation = {
        id: Date.now(),
        agents,
        topic,
        maxRounds: maxRounds || 10,
        moderator: moderator || agents[0],
        status: 'active',
        startTime: new Date().toISOString(),
        currentRound: 1,
        messages: [
          {
            round: 1,
            agent: moderator || agents[0],
            message: `Starting conversation on topic: ${topic}`,
            timestamp: new Date().toISOString(),
          }
        ],
      };
      
      conversationsStore.push(conversation);
      
      // Update agent conversation counts
      agents.forEach(agentName => {
        const agent = agentsStore.find(a => a.name === agentName);
        if (agent) {
          agent.conversationCount++;
          agent.lastActive = new Date().toISOString();
        }
      });
      
      // Simulate conversation progress
      setTimeout(() => {
        conversation.status = 'completed';
        conversation.endTime = new Date().toISOString();
        conversation.currentRound = conversation.maxRounds;
      }, 5000);
      
      return {
        content: [
          {
            type: 'text',
            text: `Conversation started:\n\nTopic: ${topic}\nParticipants: ${agents.join(', ')}\nModerator: ${conversation.moderator}\nMax Rounds: ${conversation.maxRounds}\nConversation ID: ${conversation.id}\n\nStatus: ${conversation.status}`,
          },
        ],
      };
    }

    case 'create_workflow': {
      const { name: workflowName, description, steps, parallel } = CreateWorkflowSchema.parse(args);
      
      // Validate agents in steps exist
      const requiredAgents = [...new Set(steps.map(step => step.agent))];
      const missingAgents = requiredAgents.filter(name => !agentsStore.find(a => a.name === name));
      if (missingAgents.length > 0) {
        throw new Error(`Agents not found for workflow: ${missingAgents.join(', ')}`);
      }
      
      const workflow = {
        id: Date.now(),
        name: workflowName,
        description,
        steps,
        parallel: parallel || false,
        status: 'ready',
        createdAt: new Date().toISOString(),
        executionCount: 0,
        lastExecuted: null,
      };
      
      workflowsStore.push(workflow);
      
      return {
        content: [
          {
            type: 'text',
            text: `Workflow created successfully:\n\nName: ${workflowName}\nDescription: ${description}\nSteps: ${steps.length}\nParallel Execution: ${workflow.parallel}\nRequired Agents: ${requiredAgents.join(', ')}\n\nWorkflow ID: ${workflow.id}`,
          },
        ],
      };
    }

    case 'execute_workflow': {
      const { workflowName, inputs, timeout } = ExecuteWorkflowSchema.parse(args);
      
      const workflow = workflowsStore.find(w => w.name === workflowName);
      if (!workflow) {
        throw new Error(`Workflow '${workflowName}' not found.`);
      }
      
      const execution = {
        id: Date.now(),
        workflowId: workflow.id,
        workflowName,
        inputs: inputs || {},
        timeout: timeout || 300,
        status: 'running',
        startTime: new Date().toISOString(),
        currentStep: 0,
        completedSteps: [],
        results: {},
      };
      
      workflow.executionCount++;
      workflow.lastExecuted = new Date().toISOString();
      
      // Simulate workflow execution
      const executionTime = Math.random() * 10000 + 2000;
      setTimeout(() => {
        execution.status = 'completed';
        execution.endTime = new Date().toISOString();
        execution.currentStep = workflow.steps.length;
        execution.completedSteps = workflow.steps.map((step, index) => ({
          stepIndex: index,
          agent: step.agent,
          task: step.task,
          result: `Step ${index + 1} completed by ${step.agent}`,
          completedAt: new Date().toISOString(),
        }));
      }, Math.min(executionTime, 8000));
      
      return {
        content: [
          {
            type: 'text',
            text: `Workflow execution started:\n\nWorkflow: ${workflowName}\nSteps: ${workflow.steps.length}\nParallel: ${workflow.parallel}\nTimeout: ${execution.timeout}s\nInputs: ${JSON.stringify(execution.inputs)}\n\nExecution ID: ${execution.id}\nStatus: ${execution.status}`,
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
        uri: "autogen://agents",
        name: "Agent Registry",
        description: "All created AutoGen agents",
        mimeType: "application/json",
      },
      {
        uri: "autogen://conversations",
        name: "Conversation History",
        description: "History of all multi-agent conversations",
        mimeType: "application/json",
      },
      {
        uri: "autogen://workflows",
        name: "Workflow Registry",
        description: "All created workflows and their execution history",
        mimeType: "application/json",
      },
    ],
  };
});

// Handle resource reads
server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
  const { uri } = request.params;

  switch (uri) {
    case "autogen://agents":
      return {
        contents: [
          {
            uri,
            mimeType: "application/json",
            text: JSON.stringify({
              agents: agentsStore,
              summary: {
                totalAgents: agentsStore.length,
                activeAgents: agentsStore.filter(a => a.status === 'active').length,
                totalConversations: agentsStore.reduce((sum, a) => sum + a.conversationCount, 0),
              }
            }, null, 2),
          },
        ],
      };

    case "autogen://conversations":
      return {
        contents: [
          {
            uri,
            mimeType: "application/json",
            text: JSON.stringify({
              conversations: conversationsStore,
              summary: {
                totalConversations: conversationsStore.length,
                activeConversations: conversationsStore.filter(c => c.status === 'active').length,
                completedConversations: conversationsStore.filter(c => c.status === 'completed').length,
              }
            }, null, 2),
          },
        ],
      };

    case "autogen://workflows":
      return {
        contents: [
          {
            uri,
            mimeType: "application/json",
            text: JSON.stringify({
              workflows: workflowsStore,
              summary: {
                totalWorkflows: workflowsStore.length,
                totalExecutions: workflowsStore.reduce((sum, w) => sum + w.executionCount, 0),
              }
            }, null, 2),
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
  console.error('AutoGen MCP server running on stdio');
}

// Start the server
main().catch(console.error);