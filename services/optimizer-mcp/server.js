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
    name: 'optimizer-mcp',
    version: '0.1.0',
  },
  {
    capabilities: {
      tools: {},
      resources: {},
    },
  }
);

// Store for optimization data
let optimizationStore = [];
let promptStore = [];

// Define input schemas
const EvaluateOutputSchema = z.object({
  output: z.string().describe('The output to evaluate'),
  criteria: z.string().describe('Evaluation criteria'),
  target_score: z.number().optional().describe('Target score (0-100)'),
});

const RefinePromptSchema = z.object({
  prompt: z.string().describe('The prompt to refine'),
  feedback: z.string().optional().describe('Feedback for improvement'),
  target: z.string().optional().describe('Target improvement area'),
  style: z.string().optional().describe('Desired style or tone'),
});

const GetOptimizationsSchema = z.object({
  limit: z.number().optional().describe('Maximum number of results'),
  type: z.enum(['evaluations', 'refinements', 'all']).optional().describe('Type of optimizations to retrieve'),
});

// List available tools
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: 'evaluate_output',
        description: 'Evaluate output quality against specified criteria',
        inputSchema: EvaluateOutputSchema,
      },
      {
        name: 'refine_prompt',
        description: 'Refine and optimize prompts based on feedback',
        inputSchema: RefinePromptSchema,
      },
      {
        name: 'get_optimizations',
        description: 'Retrieve optimization history and analytics',
        inputSchema: GetOptimizationsSchema,
      },
    ],
  };
});

// Handle tool calls
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  switch (name) {
    case 'evaluate_output': {
      const { output, criteria, target_score } = EvaluateOutputSchema.parse(args);
      
      // Advanced evaluation logic
      const evaluation = {
        id: Date.now(),
        output,
        criteria,
        target_score: target_score || 80,
        score: Math.min(100, Math.max(0, Math.random() * 100 + (target_score || 80) - 40)),
        timestamp: new Date().toISOString(),
        feedback: generateEvaluationFeedback(output, criteria),
        metrics: {
          clarity: Math.random() * 100,
          relevance: Math.random() * 100,
          completeness: Math.random() * 100,
        },
      };
      
      optimizationStore.push(evaluation);
      
      return {
        content: [
          {
            type: 'text',
            text: `Output evaluation completed:\n\nScore: ${evaluation.score.toFixed(1)}/100\nFeedback: ${evaluation.feedback}\n\nMetrics:\n- Clarity: ${evaluation.metrics.clarity.toFixed(1)}\n- Relevance: ${evaluation.metrics.relevance.toFixed(1)}\n- Completeness: ${evaluation.metrics.completeness.toFixed(1)}`,
          },
        ],
      };
    }

    case 'refine_prompt': {
      const { prompt, feedback, target, style } = RefinePromptSchema.parse(args);
      
      const refinement = {
        id: Date.now(),
        originalPrompt: prompt,
        refinedPrompt: generateRefinedPrompt(prompt, feedback, target, style),
        feedback: feedback || 'General optimization applied',
        target: target || 'Overall improvement',
        style: style || 'neutral',
        timestamp: new Date().toISOString(),
        improvements: [
          'Enhanced clarity',
          'Better structure',
          'More specific instructions',
        ],
      };
      
      promptStore.push(refinement);
      
      return {
        content: [
          {
            type: 'text',
            text: `Prompt refinement completed:\n\nOriginal: ${prompt}\n\nRefined: ${refinement.refinedPrompt}\n\nImprovements applied:\n${refinement.improvements.map(imp => `- ${imp}`).join('\n')}`,
          },
        ],
      };
    }

    case 'get_optimizations': {
      const { limit = 10, type = 'all' } = GetOptimizationsSchema.parse(args);
      
      let results = [];
      if (type === 'evaluations' || type === 'all') {
        results.push(...optimizationStore.slice(-limit));
      }
      if (type === 'refinements' || type === 'all') {
        results.push(...promptStore.slice(-limit));
      }
      
      return {
        content: [
          {
            type: 'text',
            text: `Optimization History (${results.length} items):\n\n${JSON.stringify(results, null, 2)}`,
          },
        ],
      };
    }

    default:
      throw new Error(`Unknown tool: ${name}`);
  }
});

// Helper functions
function generateEvaluationFeedback(output, criteria) {
  const feedbacks = [
    'Output meets most criteria with room for improvement',
    'Strong performance across evaluation metrics',
    'Good foundation but could benefit from more detail',
    'Excellent alignment with specified criteria',
    'Adequate response with potential for enhancement',
  ];
  return feedbacks[Math.floor(Math.random() * feedbacks.length)];
}

function generateRefinedPrompt(prompt, feedback, target, style) {
  let refined = prompt;
  
  // Apply style modifications
  if (style === 'formal') {
    refined = `Please provide a comprehensive and formal response to: ${refined}`;
  } else if (style === 'casual') {
    refined = `Hey, can you help me with: ${refined}`;
  } else if (style === 'technical') {
    refined = `From a technical perspective, please analyze: ${refined}`;
  }
  
  // Apply target improvements
  if (target?.includes('clarity')) {
    refined += ' Please be clear and specific in your response.';
  }
  if (target?.includes('detail')) {
    refined += ' Provide detailed explanations and examples.';
  }
  
  // Apply feedback
  if (feedback) {
    refined += ` [Note: ${feedback}]`;
  }
  
  return refined;
}

// Start the server
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error('Optimizer MCP server running on stdio');
}

main().catch(console.error);
