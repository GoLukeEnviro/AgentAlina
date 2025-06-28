import { AgentChatProvider } from '@agno-agi/agent-ui';

const AgentDashboard = () => (
  <AgentChatProvider 
    apiEndpoint="http://localhost:8000/agent-api"
    multimodalInputs={['text', 'voice', 'image']}
  />
);